"""把 data/*.md 匯入 Chroma。

用法：python ingest.py [--dry-run] [--rebuild] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, Sequence

from chunking import Chunk, load_chunks

EMBED_BATCH = 32

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class Store(Protocol):
    embed_model: str | None

    def reset(self, embed_model: str) -> None: ...
    def existing_hashes(self) -> dict[str, str]: ...
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def delete(self, ids: Sequence[str]) -> None: ...
    def mark_updated(self) -> None: ...


@dataclass
class SyncPlan:
    add: list[Chunk] = field(default_factory=list)
    update: list[Chunk] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)
    rebuild: bool = False

    def summary(self) -> str:
        return f"新增 {len(self.add)}、更新 {len(self.update)}、刪除 {len(self.delete)}、跳過 {len(self.skip)}"


def plan_sync(chunks: Sequence[Chunk], existing: dict[str, str]) -> SyncPlan:
    """比對檔案切出的塊和庫裡現有的 {id: hash}（純函式）。"""
    plan = SyncPlan()
    new_ids = set()
    for c in chunks:
        new_ids.add(c.id)
        if c.id not in existing:
            plan.add.append(c)
        elif existing[c.id] != c.content_hash:
            plan.update.append(c)
        else:
            plan.skip.append(c.id)
    plan.delete = sorted(i for i in existing if i not in new_ids)
    return plan


async def run_ingest(
    chunks: Sequence[Chunk],
    store: Store,
    embed_fn: EmbedFn,
    embed_model: str,
    *,
    dry_run: bool = False,
    rebuild: bool = False,
) -> SyncPlan:
    # 換了 embedding 模型（或指定 --rebuild）：舊向量不能用，整批重建
    rebuild = rebuild or (store.embed_model is not None and store.embed_model != embed_model)
    existing = store.existing_hashes()
    plan = plan_sync(chunks, {} if rebuild else existing)
    plan.rebuild = rebuild
    if rebuild:
        new_ids = {c.id for c in chunks}
        plan.delete = sorted(i for i in existing if i not in new_ids)

    if dry_run:
        return plan

    if rebuild or store.embed_model is None:
        store.reset(embed_model)
    else:
        store.delete(plan.delete)

    todo = plan.add + plan.update
    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start : start + EMBED_BATCH]
        vectors = await embed_fn([c.embed_text for c in batch])
        store.upsert(batch, vectors)
    # 通知正在跑的 app：資料變了，要重新連線
    store.mark_updated()
    return plan


def main(argv: list[str] | None = None) -> int:
    from config import get_settings
    from retrieval import ChromaStore, embed

    s = get_settings()
    parser = argparse.ArgumentParser(description="把知識庫 md 匯入 Chroma")
    parser.add_argument("--dry-run", action="store_true", help="只印出會做什麼，不改資料庫")
    parser.add_argument("--rebuild", action="store_true", help="刪掉整個 collection 重建")
    parser.add_argument("--data-dir", default=s.data_dir, help=f"知識庫資料夾（預設 {s.data_dir}）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        chunks = load_chunks(args.data_dir)
    except ValueError as e:
        print(f"資料格式錯誤：{e}", file=sys.stderr)
        return 1
    print(f"讀到 {len(chunks)} 塊（{args.data_dir}）")

    store = ChromaStore(s.chroma_dir, s.chroma_collection)

    async def embed_fn(texts: list[str]) -> list[list[float]]:
        return await embed(texts, url=s.ollama_url, model=s.embed_model)

    plan = asyncio.run(
        run_ingest(chunks, store, embed_fn, s.embed_model, dry_run=args.dry_run, rebuild=args.rebuild)
    )
    if plan.rebuild:
        print(f"整批重建（embedding 模型：{s.embed_model}）")
    prefix = "[dry-run] 預計 " if args.dry_run else ""
    print(prefix + plan.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
