"""把 Discord 討論整理成知識庫草稿。

用法：
  python scripts/draft_from_discord.py 討論.txt
  python scripts/draft_from_discord.py 匯出.json --model gpt-5-mini
  python scripts/draft_from_discord.py 討論.txt --no-dedupe --max-chars 4000

輸入：
- 直接從 Discord 複製貼上存成 .txt（UTF-8）
- 或用 DiscordChatExporter 匯出的 .json

輸出：data/_drafts/<檔名>-<時間>/<類別>.md（不會被 ingest 讀到，也不進 Git）
審核完把要的條目搬到 data/<類別>.md，再跑 python ingest.py。

用的是 .env 的 LLM_PROVIDER / LLM_MODEL（可用 --model 換）。討論內容會送到那個模型，
用 OpenAI 時等於把社群的對話送給 OpenAI，貼之前想一下內容適不適合。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from config import get_settings  # noqa: E402
from drafting import (  # noqa: E402
    draft_entries,
    flag_names,
    load_discussion,
    make_windows,
    mark_duplicates,
    speaker_names,
    write_drafts,
)
from llm import get_llm  # noqa: E402

DUP_THRESHOLD = 0.8


def main() -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Discord 討論 → 知識庫草稿")
    parser.add_argument("input", help="討論紀錄 .txt 或 DiscordChatExporter .json")
    parser.add_argument("--model", help=f"整理用的模型（預設 {s.llm_model}）")
    parser.add_argument("--source", help="來源說明（預設從檔名或頻道名稱產生）")
    parser.add_argument("--max-chars", type=int, default=6000, help="每段最多幾個字（預設 6000）")
    parser.add_argument("--no-dedupe", action="store_true", help="不跟現有知識庫比對（不需要 Ollama）")
    parser.add_argument("--out", default="data/_drafts", help="輸出資料夾")
    parser.add_argument("--prompt", default="prompts/draft_v1.txt")
    args = parser.parse_args()

    lines, source = load_discussion(args.input)
    source = args.source or source
    if not lines:
        print("檔案裡沒有內容")
        return 1
    windows = make_windows(lines, args.max_chars)
    categories = sorted(p.stem for p in Path(s.data_dir).glob("*.md"))
    print(f"讀到 {len(lines)} 則訊息，分成 {len(windows)} 段；現有類別：{'、'.join(categories) or '無'}")

    # 整理比回答需要更多輸出，推理模型也讓它多想一點
    ds = replace(
        s,
        llm_model=args.model or s.llm_model,
        llm_max_tokens=max(s.llm_max_tokens, 6000),
        llm_temperature=0.1,
        openai_reasoning_effort="low",
    )
    system = Path(args.prompt).read_text(encoding="utf-8")

    async def run():
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
            llm = get_llm(ds, client)
            print(f"用 {llm.provider} / {llm.model} 整理中…")
            entries = await draft_entries(
                windows, llm, system, categories, on_error=lambda i, e: print(f"  第 {i} 段失敗，跳過：{e}")
            )
            if entries and not args.no_dedupe:
                from retrieval import ChromaStore, embed

                store = ChromaStore(s.chroma_dir, s.chroma_collection)

                async def find(texts):
                    vecs = await embed(texts, url=s.ollama_url, model=s.embed_model, client=client)
                    out = []
                    for v in vecs:
                        hits = store.search(v, 1)
                        out.append((hits[0].chunk_id, hits[0].score) if hits else None)
                    return out

                try:
                    await mark_duplicates(entries, find, DUP_THRESHOLD)
                except Exception as e:  # noqa: BLE001
                    print(f"  查重失敗（跳過）：{e}")
            return entries

    entries = asyncio.run(run())
    if not entries:
        print("沒有整理出任何條目")
        return 0
    flag_names(entries, speaker_names(lines))

    out_dir = Path(args.out) / f"{Path(args.input).stem}-{datetime.now():%Y%m%d-%H%M}"
    paths = write_drafts(entries, out_dir, source, f"{datetime.now():%Y/%m/%d}")
    low = sum(e.confidence == "低" for e in entries)
    dup = sum(e.duplicate_of is not None for e in entries)
    named = sum("玩家名字" in e.notes for e in entries)
    print(f"\n整理出 {len(entries)} 條（信心低 {low}、可能重複 {dup}、出現玩家名字 {named}）：")
    for p in paths:
        print(f"  {p}")
    print("審核後把要的條目搬到 data/，再跑 python ingest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
