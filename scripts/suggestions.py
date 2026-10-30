"""看玩家許願和「找不到答案」的問題 = 下一批要整理的資料。

用法：
  python scripts/suggestions.py
  python scripts/suggestions.py --since 2026-11-01 --top 50
  python scripts/suggestions.py --db storage/qa.db

同樣的問題（正規化後一樣）會合併計數，依次數排序。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from ratelimit import TAIPEI  # noqa: E402

SOURCE_NAMES = {"footer": "頁尾", "no_match": "找不到", "not_found": "沒答案", "feedback": "👎"}


def taipei_day(iso: str) -> str:
    """資料庫存 UTC，顯示成台灣日期。"""
    try:
        return datetime.fromisoformat(iso).astimezone(TAIPEI).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return (iso or "")[:10]


def top_suggestions(conn: sqlite3.Connection, since: str, limit: int) -> list[tuple]:
    """[(次數, 代表文字, 來源們, 最後一次時間)]"""
    rows = conn.execute(
        "SELECT COUNT(*), MAX(text), GROUP_CONCAT(DISTINCT source), MAX(created_at) FROM suggestions"
        " WHERE created_at >= ? GROUP BY text_norm ORDER BY COUNT(*) DESC, MAX(created_at) DESC LIMIT ?",
        (since, limit),
    ).fetchall()
    return [
        (n, text, ",".join(SOURCE_NAMES.get(s, s) for s in (srcs or "").split(",")), taipei_day(last))
        for n, text, srcs, last in rows
    ]


def top_unanswered(conn: sqlite3.Connection, since: str, limit: int) -> list[tuple]:
    """[(次數, 代表問題, 狀態們, 最高分)]：檢索不到（no_match）或模型說沒答案（not_found）。"""
    rows = conn.execute(
        "SELECT COUNT(*), MAX(question), GROUP_CONCAT(DISTINCT status), MAX(top_score) FROM qa_logs"
        " WHERE status IN ('no_match', 'not_found') AND created_at >= ?"
        " GROUP BY question_norm ORDER BY COUNT(*) DESC, MAX(created_at) DESC LIMIT ?",
        (since, limit),
    ).fetchall()
    return [(n, q, statuses, score) for n, q, statuses, score in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="玩家許願 + 找不到答案的問題")
    parser.add_argument("--db", default=get_settings().db_path)
    parser.add_argument("--since", default="", help="只看這天之後（例如 2026-11-01，UTC）")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"找不到資料庫：{args.db}")
        return 1
    conn = sqlite3.connect(args.db)
    try:
        has_table = conn.execute("SELECT 1 FROM sqlite_master WHERE name='suggestions'").fetchone()
        print("== 玩家許願 ==")
        rows = top_suggestions(conn, args.since, args.top) if has_table else []
        if not rows:
            print("（還沒有）")
        for n, text, sources, last in rows:
            print(f"{n:4d}  {text}　［{sources}｜{last}］")

        print("\n== 找不到答案的問題 ==")
        rows = top_unanswered(conn, args.since, args.top)
        if not rows:
            print("（還沒有）")
        for n, q, statuses, score in rows:
            label = "、".join({"no_match": "檢索不到", "not_found": "模型說沒答案"}[s] for s in statuses.split(","))
            print(f"{n:4d}  {q}　［{label}｜最高分 {score or 0:.2f}］")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
