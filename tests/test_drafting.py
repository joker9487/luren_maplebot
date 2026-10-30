import asyncio
import json

import pytest

from chunking import parse_file
from drafting import (
    DraftEntry,
    draft_entries,
    extract_json,
    load_discussion,
    make_windows,
    mark_duplicates,
    merge_entries,
    normalize_entries,
    parse_exporter_json,
    write_drafts,
)
from llm import FakeLLM

EXPORT = {
    "channel": {"name": "新手問答"},
    "messages": [
        {"timestamp": "2026-10-01T10:00:00+08:00", "author": {"name": "a1", "nickname": "小明"}, "content": "影武者幾等轉?"},
        {"timestamp": "2026-10-01T10:01:00+08:00", "author": {"name": "b2"}, "content": "20等下忍\n30等中忍"},
        {"timestamp": "2026-10-02T09:00:00+08:00", "author": {"name": "c3"}, "content": ""},  # 只有圖片
    ],
}


# ---- 讀取 ----

def test_parse_exporter_json():
    lines, source = parse_exporter_json(EXPORT)
    assert lines == ["[2026-10-01] 小明：影武者幾等轉?", "[2026-10-01] b2：20等下忍 ／ 30等中忍"]
    assert source == "Discord #新手問答 2026-10-01～2026-10-01"


def test_load_discussion_txt_and_json(tmp_path):
    txt = tmp_path / "討論.txt"
    txt.write_text("﻿第一行\n\n第二行\n", encoding="utf-8")
    assert load_discussion(txt) == (["第一行", "第二行"], "Discord 討論（討論）")

    js = tmp_path / "export.json"
    js.write_text(json.dumps(EXPORT, ensure_ascii=False), encoding="utf-8")
    assert len(load_discussion(js)[0]) == 2


# ---- 切段 ----

def test_make_windows_respects_limit_and_overlaps():
    lines = [f"第{i:02d}行" + "字" * 10 for i in range(30)]  # 每行 15 字
    windows = make_windows(lines, max_chars=100, overlap=2)
    assert all(len(w) <= 100 for w in windows)
    assert windows[0].startswith("第00行") and windows[-1].endswith(lines[-1])
    # 下一段開頭 = 上一段最後兩行
    first, second = windows[0].split("\n"), windows[1].split("\n")
    assert second[:2] == first[-2:]


def test_make_windows_single_and_long_line():
    assert make_windows(["短"], 100) == ["短"]
    assert make_windows(["超" * 500, "短"], 100, overlap=0) == ["超" * 500, "短"]  # 一行太長也要往前走


# ---- 解析模型回應 ----

def test_extract_json_handles_fences_and_chatter():
    text = '好的，以下是結果：\n```json\n{"entries": [{"title": "t"}]}\n```\n希望有幫助'
    assert extract_json(text) == {"entries": [{"title": "t"}]}
    with pytest.raises(ValueError):
        extract_json("沒有 JSON")


def test_normalize_entries_cleans_fields():
    data = {
        "entries": [
            {"category": "轉職/任務", "title": "## 影武者幾等轉職", "keywords": "雙刀、DB", "content": "20 等\n## 下一段", "confidence": "超高"},
            {"category": "練等", "title": "", "content": "沒標題"},
            {"category": "練等", "title": "沒內容", "content": " "},
            "不是物件",
        ]
    }
    [e] = normalize_entries(data)
    assert e.category == "轉職任務"
    assert e.title == "影武者幾等轉職"
    assert e.keywords == ["雙刀", "DB"]
    assert e.content == "20 等\n### 下一段"  # ## 降成 ###，不會被當成新條目
    assert e.confidence == "中"
    assert normalize_entries({}) == []


def test_merge_entries_keeps_longest_and_unions_keywords():
    a = DraftEntry("轉職任務", "影武者幾等轉職", "短", ["雙刀"], notes="n1")
    b = DraftEntry("轉職任務", "影武者 幾等轉職", "比較長的內容", ["DB", "雙刀"], notes="n2")
    c = DraftEntry("練等", "影武者幾等轉職", "別的類別", [])
    merged = merge_entries([a, b, c])
    assert len(merged) == 2
    assert merged[0].content == "比較長的內容"
    assert merged[0].keywords == ["雙刀", "DB"]
    assert merged[0].notes == "n1；n2"


# ---- 整個流程 ----

def test_draft_entries_skips_failed_windows():
    good = json.dumps({"entries": [{"category": "轉職任務", "title": "影武者幾等轉職", "content": "20 等下忍"}]}, ensure_ascii=False)

    class ScriptedLLM(FakeLLM):
        def __init__(self, replies):
            super().__init__()
            self.replies = list(replies)

        async def stream(self, messages):
            self.calls.append(messages)
            yield self.replies.pop(0)

    llm = ScriptedLLM([good, "壞掉的回應", good])
    errors = []
    entries = asyncio.run(
        draft_entries(["段1", "段2", "段3"], llm, "類別：{categories}", ["轉職任務"], on_error=lambda i, e: errors.append(i))
    )
    assert len(entries) == 1  # 第 1、3 段一樣，合併成一條
    assert errors == [2]
    assert llm.calls[0][0]["content"] == "類別：轉職任務"
    assert "第 2/3 段" in llm.calls[1][1]["content"] and "段2" in llm.calls[1][1]["content"]


def test_mark_duplicates():
    entries = [DraftEntry("c", "a", "內容a"), DraftEntry("c", "b", "內容b"), DraftEntry("c", "x", "內容x")]

    async def find(texts):
        assert len(texts) == 3 and texts[0].startswith("c\na")
        return [("轉職任務/冒險家四轉#0", 0.91), ("練等/x#0", 0.5), None]

    asyncio.run(mark_duplicates(entries, find, 0.8))
    assert entries[0].duplicate_of == ("轉職任務/冒險家四轉#0", 0.91)
    assert entries[1].duplicate_of is None and entries[2].duplicate_of is None


def test_write_drafts_roundtrip_through_real_parser(tmp_path):
    entries = [
        DraftEntry("轉職任務", "影武者幾等轉職", "1. 20 等下忍\n2. 30 等中忍", ["雙刀", "DB"], "高", "有人說 25 等，後來更正",
                   duplicate_of=("轉職任務/影武者轉職總覽#0", 0.86)),
        DraftEntry("新類別", "怎麼賺錢", "打王賣裝", [], "低"),
    ]
    paths = write_drafts(entries, tmp_path / "run", "Discord #新手問答", "2026/10/30")
    assert [p.name for p in paths] == ["新類別.md", "轉職任務.md"]

    text = (tmp_path / "run" / "轉職任務.md").read_text(encoding="utf-8")
    assert "<!-- 草稿｜信心：高｜可能重複：轉職任務/影武者轉職總覽（0.86）｜備註：有人說 25 等，後來更正 -->" in text

    [e] = parse_file(tmp_path / "run" / "轉職任務.md")
    assert e.title == "影武者幾等轉職"
    assert e.keywords == ["雙刀", "DB"]
    assert e.source == "Discord #新手問答（草稿）"
    assert e.content == "1. 20 等下忍\n2. 30 等中忍"  # 審核註解不會混進內容


def test_speaker_names_and_flag():
    from drafting import flag_names, speaker_names

    lines = ["[2026-10-20] 老玩家A：答案", "[2026-10-20] 阿貓: 問題", "沒有格式的行", "[2026-10-20] X：太短的名字不抓"]
    assert speaker_names(lines) == {"老玩家A", "阿貓"}

    clean = DraftEntry("c", "標題", "內容沒有名字")
    dirty = DraftEntry("c", "標題2", "老玩家A 說要去找雪姬", notes="原本的備註")
    flag_names([clean, dirty], {"老玩家A", "阿貓"})
    assert clean.notes == ""
    assert dirty.notes == "⚠ 內容出現玩家名字：老玩家A，請刪掉；原本的備註"
