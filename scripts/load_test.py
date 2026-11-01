"""壓力測試：模擬多人同時問問題，看首字時間、總時間、429 / 503 數量。

用法：
  python scripts/load_test.py --users 5 --per-user 3
  python scripts/load_test.py --users 10 --per-user 2 --questions eval/questions.jsonl

每個請求帶不同的假 IP（CF-Connecting-IP），避開單一 IP 的冷卻和每小時上限。
注意：
- 一定要關掉快取（CACHE_ENABLED=false）。內建題目只有 10 題，開著快取的話只有第一輪會真的問模型，
  之後全部是快取（幾毫秒），數字會好看得不真實。結果裡有快取命中會跳警告
- 每題都會寫進伺服器的 qa.db、也會算進每日上限，建議用另一個資料庫、調高每日上限

啟動伺服器（PowerShell）：
  $env:CACHE_ENABLED="false"; $env:DB_PATH="storage/loadtest.db"; $env:DAILY_LIMIT="100000"; uvicorn app:app
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_QUESTIONS = [
    "影武者幾等轉職？",
    "30 等去哪練等？",
    "交易被封鎖怎麼辦？",
    "冒險家四轉要準備什麼？",
    "英雄四轉技能怎麼點？",
    "萬聖節幽靈在哪？",
    "三轉要去哪裡找長老？",
    "帳號被封鎖 3000 年怎麼辦？",
    "火毒技能怎麼配？",
    "100 等去哪練功？",
]

_ip_counter = itertools.count(1)


def fake_ip() -> str:
    n = next(_ip_counter)
    return f"10.{(n >> 16) & 255}.{(n >> 8) & 255}.{n & 255}"


@dataclass
class Result:
    http_status: int | None  # None = 連線錯誤
    status: str | None = None  # SSE done 的 status
    first_token: float | None = None  # 秒
    total: float | None = None  # 秒
    cached: bool = False  # 這題是快取回答的（沒有真的呼叫模型）


def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if path.endswith(".jsonl"):
        return [json.loads(l)["q"] for l in lines]
    return lines


def percentile(values: list[float], p: float) -> float:
    """最近排名法（nearest-rank）。"""
    s = sorted(values)
    k = max(1, math.ceil(p / 100 * len(s)))
    return s[k - 1]


def summarize(values: list[float]) -> str:
    if not values:
        return "（沒有資料）"
    return (
        f"平均 {statistics.mean(values):.2f}s  P50 {percentile(values, 50):.2f}s  "
        f"P95 {percentile(values, 95):.2f}s  最大 {max(values):.2f}s"
    )


async def ask_once(client: httpx.AsyncClient, url: str, question: str) -> Result:
    started = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{url}/api/ask", json={"question": question}, headers={"CF-Connecting-IP": fake_ip()}
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return Result(resp.status_code, total=time.perf_counter() - started)
            result = Result(200)
            event = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    if event == "token" and result.first_token is None:
                        result.first_token = time.perf_counter() - started
                    elif event == "done":
                        done = json.loads(line[5:])
                        result.status = done["status"]
                        result.cached = bool(done.get("cached"))
            result.total = time.perf_counter() - started
            return result
    except httpx.HTTPError as e:
        print(f"  連線錯誤：{e!r}", file=sys.stderr)
        return Result(None, total=time.perf_counter() - started)


async def user(client: httpx.AsyncClient, url: str, questions: list[str], start: int, n: int) -> list[Result]:
    # 一個使用者依序問 n 題
    return [await ask_once(client, url, questions[(start + i) % len(questions)]) for i in range(n)]


async def run(url: str, users: int, per_user: int, questions: list[str]) -> list[Result]:
    timeout = httpx.Timeout(180, connect=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        batches = await asyncio.gather(*(user(client, url, questions, i, per_user) for i in range(users)))
    return [r for batch in batches for r in batch]


def report(results: list[Result], elapsed: float) -> None:
    ok = [r for r in results if r.http_status == 200]
    count = lambda code: sum(r.http_status == code for r in results)  # noqa: E731
    print(f"\n共 {len(results)} 題，花了 {elapsed:.1f} 秒")
    print(f"成功 {len(ok)}、429 {count(429)}、503 {count(503)}、錯誤 {len(results) - len(ok) - count(429) - count(503)}")
    statuses: dict[str, int] = {}
    for r in ok:
        statuses[r.status or "（沒收到 done）"] = statuses.get(r.status or "（沒收到 done）", 0) + 1
    print("回答狀態：" + "、".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    print("首字時間：" + summarize([r.first_token for r in ok if r.first_token is not None and r.status != "no_match"]))
    print("總時間：  " + summarize([r.total for r in ok if r.total is not None]))
    cached = sum(r.cached for r in ok)
    if cached:
        print()
        print(f"⚠️  有 {cached} 題是快取回答的（沒有真的呼叫模型），上面的時間會偏快、不準。")
        print("   請用 CACHE_ENABLED=false 重新啟動伺服器再測。")


def main() -> None:
    parser = argparse.ArgumentParser(description="問答 API 壓力測試")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=5, help="同時人數")
    parser.add_argument("--per-user", type=int, default=2, help="每人題數（依序問）")
    parser.add_argument("--questions", help="題目檔：.jsonl（取 q 欄位）或一行一題的文字檔；不給就用內建題目")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"{args.users} 人同時問，每人 {args.per_user} 題 → {args.url}")
    started = time.perf_counter()
    results = asyncio.run(run(args.url.rstrip("/"), args.users, args.per_user, questions))
    report(results, time.perf_counter() - started)


if __name__ == "__main__":
    main()
