"""互動式檢索：輸入問題 → 印前 k 塊的分數、類別、標題、內容前 50 字。用來觀察分數、定門檻。

用法：python search_cli.py [--k 4]
"""

from __future__ import annotations

import argparse
import asyncio

from config import get_settings
from retrieval import ChromaStore, embed


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="互動式檢索")
    parser.add_argument("--k", type=int, default=s.top_k)
    args = parser.parse_args()

    store = ChromaStore(s.chroma_dir, s.chroma_collection)
    if store.count() == 0:
        print("知識庫是空的，先跑 python ingest.py")
        return
    if store.embed_model != s.embed_model:
        print(f"注意：庫是用 {store.embed_model} 建的，現在 EMBED_MODEL={s.embed_model}，請重跑 ingest.py")
    print(f"共 {store.count()} 塊，門檻 {s.score_threshold}。輸入問題（空白或 Ctrl+C 離開）")

    while True:
        try:
            q = input("\n問題> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        vec = asyncio.run(embed([q], url=s.ollama_url, model=s.embed_model))[0]
        for rank, h in enumerate(store.search(vec, args.k), 1):
            mark = "  " if h.score >= s.score_threshold else "↓ "  # ↓ = 低於門檻
            body = h.display_text.split("\n", 1)[-1].replace("\n", " ")
            print(f"{mark}{rank}. {h.score:.3f}  [{h.category}] {h.title}｜{body[:50]}")


if __name__ == "__main__":
    main()
