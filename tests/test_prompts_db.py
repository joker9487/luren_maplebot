import pytest

from db import QADB, normalize_question
from prompts import NOT_FOUND, build_messages, classify_answer, load_prompt
from retrieval import Hit


# ---- question_norm ----

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  東西不能交易怎麼辦？  ", "東西不能交易怎麼辦"),
        ("ＥＸＰ卷怎麼拿", "exp卷怎麼拿"),  # 全形英文 → 半形小寫
        ("二轉　任務   在哪", "二轉 任務 在哪"),  # 全形空白、連續空白
        ("怎麼練等??!!", "怎麼練等"),
        ("怎麼練等。", "怎麼練等"),
        ("怎麼練等 ？", "怎麼練等"),
        ("ＧＭ在嗎～", "gm在嗎"),
        ("１２３等去哪", "123等去哪"),
        ("中間的？不要動", "中間的?不要動"),  # 只去掉結尾的標點
        ("", ""),
    ],
)
def test_normalize_question(raw, expected):
    assert normalize_question(raw) == expected


# ---- prompts ----

@pytest.mark.parametrize("version", ["answer_v2", "answer_v3"])
def test_load_prompt_version_and_content(version):
    p = load_prompt(f"prompts/{version}.txt")
    assert p.version == version
    assert NOT_FOUND in p.system


def test_load_prompt_requires_not_found_sentence(tmp_path):
    f = tmp_path / "answer_v9.txt"
    f.write_text("沒有固定句的 prompt", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt(f)


def test_build_messages():
    hits = [Hit("a#0", "c", "標題一", "標題一\n內容一", 0.9), Hit("b#0", "c", "標題二", "標題二\n內容二", 0.8)]
    msgs = build_messages("SYS", "怎麼辦", hits)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {
        "role": "user",
        "content": "<資料>\n[1] 標題一\n內容一\n[2] 標題二\n內容二\n</資料>\n\n問題：怎麼辦",
    }


@pytest.mark.parametrize(
    "answer",
    [
        NOT_FOUND,
        "目前資料裡沒有這題的答案",
        "「目前資料裡沒有這題的答案。」",
        " 目前資料裡沒有這題的答案。\n",
        "目前资料里没有这题的答案。",
        # 固定句後面又自己補了建議
        "目前資料裡沒有這題的答案。不過你可以到 Discord 問。",
        # 沒照固定句、自己寫了一段（#683）
        "結論：目前資料裡沒有提到「洗9 6磁條」相關內容，無法給出明確答案。\n\n步驟：\n- 確認你在問的版本",
        "資料中沒有皇家騎士團的等級上限。",
    ],
)
def test_classify_not_found(answer):
    assert classify_answer(answer) == "not_found"


@pytest.mark.parametrize(
    "answer",
    [
        "到 Discord 回報區開單。",
        "",
        # 只回答一部分：開頭是答案，最後才說其他部分沒有
        "130 等可以去神木村練。\n其他部分目前資料裡沒有。",
        "資料整理的路線：10～15 等打綠菇菇。",
    ],
)
def test_classify_answered(answer):
    assert classify_answer(answer) == "answered"


# ---- db ----

def test_db_log_lifecycle(tmp_path):
    db = QADB(tmp_path / "sub" / "qa.db")
    db.init_sync()
    log_id = db.create_log_sync("怎麼練等？", "ollama", "qwen", "answer_v2")
    log = db.get_log_sync(log_id)
    assert log["status"] == "pending" and log["question_norm"] == "怎麼練等"
    assert log["created_at"].endswith("+00:00")

    db.finish_log_sync(
        log_id, status="answered", answer="去新手村", retrieved=[{"chunk": "練等/x#0", "score": 0.7}],
        top_score=0.7, category="練等", latency_ms=1200, first_token_ms=300,
    )
    log = db.get_log_sync(log_id)
    assert log["status"] == "answered" and log["latency_ms"] == 1200 and log["helpful"] is None

    assert db.set_feedback_sync(log_id, True, None) == "ok"
    assert db.set_feedback_sync(log_id, False, "答錯") == "duplicate"
    assert db.set_feedback_sync(log_id + 1, True, None) == "not_found"
    assert db.get_log_sync(log_id)["helpful"] == 1


def test_db_init_is_idempotent(tmp_path):
    db = QADB(tmp_path / "qa.db")
    db.init_sync()
    db.init_sync()


def test_db_migrates_old_schema(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE qa_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, question TEXT NOT NULL,"
        " question_norm TEXT NOT NULL, retrieved TEXT, top_score REAL, category TEXT, answer TEXT, status TEXT NOT NULL,"
        " provider TEXT, model TEXT, prompt_version TEXT, latency_ms INTEGER, first_token_ms INTEGER, helpful INTEGER,"
        " feedback_reason TEXT, feedback_at TEXT)"
    )
    conn.execute("INSERT INTO qa_logs (created_at, question, question_norm, status) VALUES ('t', '舊的', '舊的', 'answered')")
    conn.commit()
    conn.close()

    db = QADB(path)
    db.init_sync()
    old = db.get_log_sync(1)
    assert old["question"] == "舊的" and old["cached"] == 0 and old["cache_key"] is None


def test_cache_key():
    from cache import cache_key

    a = Hit("a#0", "c", "t", "d", 0.9, "h1")
    b = Hit("b#0", "c", "t", "d", 0.8, "h2")
    base = cache_key("q", "v2", "openai", "m", [a, b])
    assert cache_key("q", "v2", "openai", "m", [b, a]) == base  # 順序不影響
    assert cache_key("q", "v2", "openai", "m", [a]) != base
    assert cache_key("q", "v2", "openai", "m", [a, Hit("b#0", "c", "t", "d", 0.8, "改了")]) != base
    assert cache_key("q2", "v2", "openai", "m", [a, b]) != base
