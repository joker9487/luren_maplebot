"""Embedding（Ollama）與向量檢索（Chroma）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import threading
import time
from pathlib import Path

import chromadb
import httpx
from chromadb.api.client import SharedSystemClient

from chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    category: str
    title: str
    display_text: str
    score: float  # 1 - cosine distance
    content_hash: str = ""  # 這塊的 sha256，資料改了就會變（快取用）


async def embed(
    texts: list[str],
    *,
    url: str,
    model: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = 60.0,
    keep_alive: str | None = None,
) -> list[list[float]]:
    """呼叫 Ollama `/api/embed`，整個 list 一次送。

    keep_alive：模型閒置多久後從記憶體卸載（Ollama 預設 5 分鐘），例如 "24h"。
    卸載後下一題要重新載入，會多等一兩秒。
    """
    if not texts:
        return []
    owned = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        payload = {"model": model, "input": texts}
        if keep_alive:
            payload["keep_alive"] = keep_alive
        resp = await client.post(f"{url}/api/embed", json=payload)
        resp.raise_for_status()
        vectors = resp.json()["embeddings"]
    finally:
        if owned:
            await client.aclose()
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedding 數量不符：送 {len(texts)} 筆，回 {len(vectors)} 筆")
    return vectors


class ChromaStore:
    """包一層 Chroma collection。向量自己算好傳進來，不用 Chroma 內建 embedding。

    collection metadata 的 `embed_model` 記錄建庫時用的模型。
    Chroma 的呼叫是同步的，在 async 程式裡要用 asyncio.to_thread 包。
    """

    STAMP = "ingest_stamp"

    def __init__(self, path: str, name: str):
        self._path = path
        self._name = name
        self._stamp_path = Path(path) / self.STAMP
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        self._client = chromadb.PersistentClient(
            path=self._path, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._col = self._get_existing()
        self._stamp = self._read_stamp()

    def _read_stamp(self) -> int | None:
        try:
            return self._stamp_path.stat().st_mtime_ns
        except OSError:
            return None

    def _refresh(self) -> None:
        """另一個行程（ingest.py）改過資料就重新連線。

        Chroma 在同一個行程裡會快取讀過的索引：app 開著時另外跑 ingest，
        新增的資料搜不到、重建後甚至會出錯。ingest 結束時會更新 ingest_stamp，
        這裡每次讀取前檢查一下（只是看檔案時間，很便宜），變了就整個重連。
        """
        stamp = self._read_stamp()
        if stamp == self._stamp and self._col is not None:
            return
        with self._lock:
            if self._read_stamp() == self._stamp and self._col is not None:
                return
            SharedSystemClient.clear_system_cache()
            self._connect()

    def mark_updated(self) -> None:
        """ingest 寫完後呼叫，通知正在跑的 app 重新連線。"""
        self._stamp_path.parent.mkdir(parents=True, exist_ok=True)
        self._stamp_path.write_text(str(time.time_ns()), encoding="utf-8")
        self._stamp = self._read_stamp()

    def _get_existing(self):
        try:
            return self._client.get_collection(self._name)
        except Exception:  # 不存在（不同版本丟的例外型別不同）
            return None

    @property
    def embed_model(self) -> str | None:
        self._refresh()
        if self._col is None:
            return None
        return (self._col.metadata or {}).get("embed_model")

    def reset(self, embed_model: str) -> None:
        """刪掉整個 collection，用新的模型名稱重建。"""
        if self._col is not None:
            self._client.delete_collection(self._name)
        self._col = self._client.create_collection(
            self._name, metadata={"hnsw:space": "cosine", "embed_model": embed_model}
        )

    def existing_hashes(self) -> dict[str, str]:
        self._refresh()
        if self._col is None:
            return {}
        got = self._col.get(include=["metadatas"])
        return {i: (m or {}).get("hash", "") for i, m in zip(got["ids"], got["metadatas"])}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            metadatas=[
                {
                    "category": c.category,
                    "title": c.title,
                    "source": c.source,
                    "hash": c.content_hash,
                    "display_text": c.display_text,
                }
                for c in chunks
            ],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._col.delete(ids=list(ids))

    def count(self) -> int:
        self._refresh()
        return 0 if self._col is None else self._col.count()

    def list_entries(self) -> list[tuple[str, str]]:
        """庫裡有哪些條目：[(類別, 標題)]，切成多塊的條目只算一次。"""
        self._refresh()
        if self._col is None:
            return []
        metas = self._col.get(include=["metadatas"])["metadatas"]
        return list(dict.fromkeys((m["category"], m["title"]) for m in metas if m))

    def search(self, query_vec: Sequence[float], k: int) -> list[Hit]:
        """回傳最相近的 k 塊，分數由高到低；這裡不做門檻判斷。"""
        self._refresh()
        if self._col is None or self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[list(query_vec)],
            n_results=min(k, self._col.count()),
            include=["metadatas", "distances"],
        )
        return [
            Hit(
                chunk_id=cid,
                category=meta["category"],
                title=meta["title"],
                display_text=meta["display_text"],
                score=1.0 - dist,
                content_hash=meta.get("hash", ""),
            )
            for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0])
        ]
