import sqlite3

from chunking import parse_intro
from db import QADB
from scripts.suggestions import top_suggestions, top_unanswered
from topics import build_topics, read_md_info


def test_parse_intro():
    text = "# 轉職任務\n轉職任務，目前僅收錄冒險家\n<!-- 註解 -->\n\n其餘照任務指引\n## 第一條\n內容\n"
    assert parse_intro(text) == "轉職任務，目前僅收錄冒險家\n其餘照任務指引"
    assert parse_intro("## 直接開始\n內容") == ""


def test_build_topics_orders_by_file_and_dedupes():
    entries = [("練等", "B"), ("轉職", "二轉"), ("轉職", "一轉"), ("練等", "A"), ("新類別", "X"), ("轉職", "舊標題")]
    md_info = {"轉職": ("轉職說明", ["一轉", "二轉"]), "練等": ("", ["A", "B"])}
    result = build_topics(entries, md_info)
    assert [c["name"] for c in result["categories"]] == ["轉職", "練等", "新類別"]
    assert result["categories"][0] == {"name": "轉職", "intro": "轉職說明", "titles": ["一轉", "二轉", "舊標題"]}
    assert result["total"] == 6


def test_read_md_info_survives_bad_file(tmp_path):
    (tmp_path / "好.md").write_text("# 好\n說明\n## a\n內容\n", encoding="utf-8")
    (tmp_path / "壞.md").write_text("壞的說明\n## 重複\n內容\n## 重複\n內容\n", encoding="utf-8")
    info = read_md_info(tmp_path)
    assert info["好"] == ("說明", ["a"])
    assert info["壞"] == ("壞的說明", [])


def test_suggestion_and_unanswered_reports(tmp_path):
    db = QADB(tmp_path / "qa.db")
    db.init_sync()
    db.add_suggestion_sync("夜使者怎麼點？", "footer", None)
    db.add_suggestion_sync("夜使者怎麼點", "no_match", 3)
    db.add_suggestion_sync("拳霸配點", "footer", None)
    for q, status in [("怎麼賺錢", "no_match"), ("怎麼賺錢？", "no_match"), ("組隊任務", "not_found"), ("影武者", "answered")]:
        log_id = db.create_log_sync(q, "fake", "m", "v")
        db.finish_log_sync(log_id, status=status, answer="", retrieved=[], top_score=0.3, category=None,
                           latency_ms=1, first_token_ms=None)

    conn = sqlite3.connect(tmp_path / "qa.db")
    sugg = top_suggestions(conn, "", 10)
    assert sugg[0][0] == 2 and sugg[0][1].startswith("夜使者怎麼點")
    assert set(sugg[0][2].split(",")) == {"頁尾", "找不到"}
    unanswered = top_unanswered(conn, "", 10)
    assert [(n, s) for n, _, s, _ in unanswered] == [(2, "no_match"), (1, "not_found")]
