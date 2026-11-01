"""看最近的問答、被按 👎 的回答、今天的統計（唯讀，不會寫入資料庫）。

用法（建議在容器裡跑，跟 app 讀同一個檔案系統）：
  docker compose exec app python scripts/recent.py
  docker compose exec app python scripts/recent.py --top 50 --answers
  docker compose exec app python scripts/recent.py --since 2026-10-30
  docker compose exec app python scripts/recent.py --down        # 只看 👎

日期一律是台灣時間。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from ratelimit import TAIPEI  # noqa: E402

STATUS_LABEL = {
    "answered": "回答",
    "not_found": "沒答案",
    "no_match": "找不到",
    "error": "錯誤",
    "aborted": "中斷",
    "pending": "進行中",
}


def taipei_day_start_utc(day: str) -> str:
    """台灣時間某天 0 點 → UTC ISO 字串（跟資料庫 created_at 同格式，可以直接比大小）。"""
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TAIPEI)
    return start.astimezone(timezone.utc).isoformat(timespec="seconds")


def taipei_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(TAIPEI).strftime("%m/%d %H:%M")
    except (TypeError, ValueError):
        return (iso or "")[:16]


def clip(text: str | None, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def connect_readonly(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def summary(conn: sqlite3.Connection, since: str) -> dict:
    cached_expr = "SUM(cached)" if has_column(conn, "qa_logs", "cached") else "0"
    row = conn.execute(
        f"SELECT COUNT(*) total, {cached_expr} cached,"
        " SUM(helpful = 1) up, SUM(helpful = 0) down,"
        " AVG(CASE WHEN status = 'answered' THEN first_token_ms END) avg_first,"
        " AVG(CASE WHEN status = 'answered' THEN latency_ms END) avg_total"
        " FROM qa_logs WHERE created_at >= ? AND status != 'pending'",
        (since,),
    ).fetchone()
    by_status = dict(
        conn.execute(
            "SELECT status, COUNT(*) FROM qa_logs WHERE created_at >= ? GROUP BY status ORDER BY COUNT(*) DESC",
            (since,),
        ).fetchall()
    )
    return {
        "total": row["total"] or 0,
        "cached": row["cached"] or 0,
        "up": row["up"] or 0,
        "down": row["down"] or 0,
        "avg_first_ms": row["avg_first"],
        "avg_total_ms": row["avg_total"],
        "by_status": by_status,
    }


def recent(conn: sqlite3.Connection, since: str, limit: int, only_down: bool = False) -> list[sqlite3.Row]:
    cached_col = "cached" if has_column(conn, "qa_logs", "cached") else "0 AS cached"
    where = "created_at >= ?" + (" AND helpful = 0" if only_down else "")
    return conn.execute(
        f"SELECT id, created_at, question, status, top_score, category, answer, helpful, feedback_reason, {cached_col}"
        f" FROM qa_logs WHERE {where} ORDER BY id DESC LIMIT ?",
        (since, limit),
    ).fetchall()


def print_summary(s: dict, title: str) -> None:
    print(f"== {title} ==")
    if not s["total"]:
        print("（還沒有問答）\n")
        return
    parts = "、".join(f"{STATUS_LABEL.get(k, k)} {v}" for k, v in s["by_status"].items())
    print(f"共 {s['total']} 題：{parts}")
    rated = s["up"] + s["down"]
    rate = f"（👎 比例 {s['down'] / rated:.0%}）" if rated else ""
    print(f"快取命中 {s['cached']}｜👍 {s['up']}  👎 {s['down']}{rate}")
    if s["avg_first_ms"] is not None:
        print(f"回答平均：首字 {s['avg_first_ms'] / 1000:.1f}s、總共 {s['avg_total_ms'] / 1000:.1f}s")
    print()


def print_rows(rows: list[sqlite3.Row], title: str, show_answer: bool) -> None:
    print(f"== {title} ==")
    if not rows:
        print("（沒有）\n")
        return
    for r in rows:
        fb = {1: "👍", 0: "👎"}.get(r["helpful"], "  ")
        mark = "快取" if r["cached"] else STATUS_LABEL.get(r["status"], r["status"])
        score = f"{r['top_score']:.2f}" if r["top_score"] is not None else " -  "
        line = f"#{r['id']:<5} {taipei_time(r['created_at'])} {fb} [{mark}] {score}  {clip(r['question'], 40)}"
        if r["helpful"] == 0 and r["feedback_reason"]:
            line += f"　← {r['feedback_reason']}"
        print(line)
        if show_answer or r["helpful"] == 0:
            print(f"        {clip(r['category'], 10)}｜{clip(r['answer'], 80)}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="最近的問答、👎、統計（唯讀）")
    parser.add_argument("--db", default=get_settings().db_path)
    parser.add_argument("--top", type=int, default=20, help="最近幾題（預設 20）")
    parser.add_argument("--since", help="只看這天（台灣時間）之後，例如 2026-10-30")
    parser.add_argument("--answers", action="store_true", help="最近的問答也印出回答開頭")
    parser.add_argument("--down", action="store_true", help="只看被按 👎 的")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"找不到資料庫：{args.db}")
        return 1
    today = datetime.now(TAIPEI).strftime("%Y-%m-%d")
    since = taipei_day_start_utc(args.since) if args.since else ""

    conn = connect_readonly(args.db)
    try:
        print_summary(summary(conn, taipei_day_start_utc(today)), f"今天（{today}）")
        if args.since:
            print_summary(summary(conn, since), f"{args.since} 之後")
        if not args.down:
            print_rows(recent(conn, since, args.top), f"最近 {args.top} 題", args.answers)
        print_rows(recent(conn, since, args.top, only_down=True), "被按 👎 的", True)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
