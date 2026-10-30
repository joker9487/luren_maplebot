"""問答紀錄（SQLite，標準函式庫 sqlite3，用 asyncio.to_thread 包成 async）。

不存使用者 IP。
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    question        TEXT NOT NULL,
    question_norm   TEXT NOT NULL,
    retrieved       TEXT,
    top_score       REAL,
    category        TEXT,
    answer          TEXT,
    status          TEXT NOT NULL,
    provider        TEXT,
    model           TEXT,
    prompt_version  TEXT,
    latency_ms      INTEGER,
    first_token_ms  INTEGER,
    helpful         INTEGER,
    feedback_reason TEXT,
    feedback_at     TEXT,
    cache_key       TEXT,
    cached          INTEGER NOT NULL DEFAULT 0,
    feedback_token  TEXT
);
CREATE INDEX IF NOT EXISTS idx_qa_logs_created_at ON qa_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_qa_logs_question_norm ON qa_logs(question_norm);

-- 玩家許願：想問但目前沒答案的問題（不存 IP）
CREATE TABLE IF NOT EXISTS suggestions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    text        TEXT NOT NULL,
    text_norm   TEXT NOT NULL,
    source      TEXT NOT NULL,   -- footer（頁尾連結）/ no_match / not_found / feedback（按 👎 資料沒有）
    log_id      INTEGER          -- 從哪一題延伸出來的（可以是 NULL）
);
CREATE INDEX IF NOT EXISTS idx_suggestions_text_norm ON suggestions(text_norm);
"""

# 舊版資料庫沒有的欄位：啟動時補上（SQLite 只能加欄位，不會動到舊資料）
MIGRATIONS = {
    "cache_key": "ALTER TABLE qa_logs ADD COLUMN cache_key TEXT",
    "cached": "ALTER TABLE qa_logs ADD COLUMN cached INTEGER NOT NULL DEFAULT 0",
    "feedback_token": "ALTER TABLE qa_logs ADD COLUMN feedback_token TEXT",
}
POST_MIGRATION = "CREATE INDEX IF NOT EXISTS idx_qa_logs_cache_key ON qa_logs(cache_key)"

_SPACES_RE = re.compile(r"\s+")
_TRAILING_PUNCT = "?？!！。.,，、~～…;；:："


def normalize_question(q: str) -> str:
    """統計常問問題用：去頭尾空白、全形轉半形、英文轉小寫、連續空白合一、去掉結尾的問號和標點。"""
    q = unicodedata.normalize("NFKC", q)  # 全形英數、符號、空白 → 半形
    q = _SPACES_RE.sub(" ", q.lower()).strip()
    return q.rstrip(_TRAILING_PUNCT + " ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class QADB:
    def __init__(self, path: str | Path):
        self.path = str(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """每次呼叫開一條連線：成功就 commit、失敗 rollback，最後一定關掉。"""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- 同步版本（在 thread 裡跑） ----

    def init_sync(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(qa_logs)")}
            for name, sql in MIGRATIONS.items():
                if name not in columns:
                    conn.execute(sql)
            conn.execute(POST_MIGRATION)

    def create_log_sync(
        self, question: str, provider: str, model: str, prompt_version: str, feedback_token: str | None = None
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO qa_logs (created_at, question, question_norm, status, provider, model, prompt_version,"
                " feedback_token) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
                (_now(), question, normalize_question(question), provider, model, prompt_version, feedback_token),
            )
            return cur.lastrowid

    def finish_log_sync(
        self,
        log_id: int,
        *,
        status: str,
        answer: str,
        retrieved: list[dict],
        top_score: float | None,
        category: str | None,
        latency_ms: int,
        first_token_ms: int | None,
        cache_key: str | None = None,
        cached: bool = False,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE qa_logs SET status=?, answer=?, retrieved=?, top_score=?, category=?,"
                " latency_ms=?, first_token_ms=?, cache_key=?, cached=? WHERE id=?",
                (
                    status,
                    answer,
                    json.dumps(retrieved, ensure_ascii=False),
                    top_score,
                    category,
                    latency_ms,
                    first_token_ms,
                    cache_key,
                    int(cached),
                    log_id,
                ),
            )

    def find_cached_answer_sync(self, cache_key: str, ttl_hours: float) -> str | None:
        """同一個 key、TTL 內由模型正常回答過（不是快取複製出來的）、而且這段時間內沒有人按 👎。"""
        since = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT answer FROM qa_logs"
                " WHERE cache_key=? AND status='answered' AND cached=0 AND created_at>=?"
                "   AND NOT EXISTS (SELECT 1 FROM qa_logs b WHERE b.cache_key=? AND b.helpful=0 AND b.created_at>=?)"
                " ORDER BY id DESC LIMIT 1",
                (cache_key, since, cache_key, since),
            ).fetchone()
            return row["answer"] if row else None

    def set_feedback_sync(self, log_id: int, helpful: bool, reason: str | None, token: str | None = None) -> str:
        """回傳 ok / not_found / duplicate（已有回饋就拒絕）。

        token 要跟建立紀錄時發給前端的一樣，別人猜得到 log_id 也沒辦法亂按 👎。
        token 不對一律當成 not_found，不透露這筆存不存在。
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE qa_logs SET helpful=?, feedback_reason=?, feedback_at=?"
                " WHERE id=? AND feedback_token IS ? AND helpful IS NULL",
                (int(helpful), reason, _now(), log_id, token),
            )
            if cur.rowcount:
                return "ok"
            exists = conn.execute(
                "SELECT 1 FROM qa_logs WHERE id=? AND feedback_token IS ?", (log_id, token)
            ).fetchone()
            return "duplicate" if exists else "not_found"

    def add_suggestion_sync(self, text: str, source: str, log_id: int | None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO suggestions (created_at, text, text_norm, source, log_id) VALUES (?, ?, ?, ?, ?)",
                (_now(), text, normalize_question(text), source, log_id),
            )
            return cur.lastrowid

    def get_log_sync(self, log_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM qa_logs WHERE id=?", (log_id,)).fetchone()
            return dict(row) if row else None

    # ---- async 版本 ----

    async def init(self) -> None:
        await asyncio.to_thread(self.init_sync)

    async def create_log(
        self, question: str, provider: str, model: str, prompt_version: str, feedback_token: str | None = None
    ) -> int:
        return await asyncio.to_thread(
            self.create_log_sync, question, provider, model, prompt_version, feedback_token
        )

    async def finish_log(self, log_id: int, **kwargs) -> None:
        await asyncio.to_thread(lambda: self.finish_log_sync(log_id, **kwargs))

    async def find_cached_answer(self, cache_key: str, ttl_hours: float) -> str | None:
        return await asyncio.to_thread(self.find_cached_answer_sync, cache_key, ttl_hours)

    async def set_feedback(self, log_id: int, helpful: bool, reason: str | None, token: str | None = None) -> str:
        return await asyncio.to_thread(self.set_feedback_sync, log_id, helpful, reason, token)

    async def add_suggestion(self, text: str, source: str, log_id: int | None) -> int:
        return await asyncio.to_thread(self.add_suggestion_sync, text, source, log_id)

    async def get_log(self, log_id: int) -> dict | None:
        return await asyncio.to_thread(self.get_log_sync, log_id)
