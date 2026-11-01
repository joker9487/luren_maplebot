import sqlite3

import pytest

from db import QADB
from scripts.recent import clip, connect_readonly, recent, summary, taipei_day_start_utc, taipei_time


def make_db(tmp_path):
    db = QADB(tmp_path / "qa.db")
    db.init_sync()

    def add(question, status, helpful=None, reason=None, cached=False, first=500):
        log_id = db.create_log_sync(question, "openai", "m", "v2", feedback_token="tok-123456")
        db.finish_log_sync(
            log_id, status=status, answer=f"{question}的回答", retrieved=[], top_score=0.6, category="轉職任務",
            latency_ms=first + 1000, first_token_ms=first, cached=cached,
        )
        if helpful is not None:
            db.set_feedback_sync(log_id, helpful, reason, "tok-123456")
        return log_id

    add("影武者幾等轉職", "answered", helpful=True)
    add("刀賊怎麼點", "answered", helpful=False, reason="答錯")
    add("影武者幾等轉職", "answered", cached=True, first=10)
    add("今天天氣", "no_match")
    return tmp_path / "qa.db"


def test_summary_and_recent(tmp_path):
    conn = connect_readonly(str(make_db(tmp_path)))
    s = summary(conn, "")
    assert s["total"] == 4 and s["cached"] == 1 and s["up"] == 1 and s["down"] == 1
    assert s["by_status"] == {"answered": 3, "no_match": 1}

    rows = recent(conn, "", 10)
    assert [r["question"] for r in rows][0] == "今天天氣"  # 新的在前
    down = recent(conn, "", 10, only_down=True)
    assert [(r["question"], r["feedback_reason"]) for r in down] == [("刀賊怎麼點", "答錯")]


def test_readonly_connection_cannot_write(tmp_path):
    conn = connect_readonly(str(make_db(tmp_path)))
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM qa_logs")


def test_since_filters_by_taipei_day(tmp_path):
    conn = connect_readonly(str(make_db(tmp_path)))
    assert summary(conn, taipei_day_start_utc("2999-01-01"))["total"] == 0


def test_time_helpers():
    assert taipei_day_start_utc("2026-10-30") == "2026-10-29T16:00:00+00:00"
    assert taipei_time("2026-10-29T16:30:00+00:00") == "10/30 00:30"
    assert clip("一二三四五", 3) == "一二…" and clip(None, 3) == ""
