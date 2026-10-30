import hashlib
import logging

import pytest

from chunking import DuplicateTitleError, Entry, load_chunks, parse_file, parse_text, split_entry

SAMPLE = """# 誤鎖

這段在第一個 ## 之前，要忽略。

## 物品被誤鎖怎麼回報
關鍵字：鎖住、不能交易，不能丟
來源：Discord 回報區置頂
1. 先截圖物品和角色名稱
2. 到 Discord 回報區開單

## 沒有選填欄位
直接就是內容。
"""


def test_parse_normal_entry_with_fields():
    entries = parse_text(SAMPLE, "誤鎖")
    assert [e.title for e in entries] == ["物品被誤鎖怎麼回報", "沒有選填欄位"]
    e = entries[0]
    assert e.category == "誤鎖"
    assert e.keywords == ["鎖住", "不能交易", "不能丟"]
    assert e.source == "Discord 回報區置頂"
    assert e.content == "1. 先截圖物品和角色名稱\n2. 到 Discord 回報區開單"


def test_parse_entry_without_fields():
    e = parse_text(SAMPLE, "誤鎖")[1]
    assert e.keywords == []
    assert e.source == ""
    assert e.content == "直接就是內容。"


def test_fields_only_right_below_title():
    text = "## 標題\n內容第一行\n關鍵字：這行是內容\n"
    e = parse_text(text, "c")[0]
    assert e.keywords == []
    assert "關鍵字：這行是內容" in e.content


def test_fields_any_order_halfwidth_colon_and_blank_lines():
    text = "## 標題\n\n來源: 某處\n關鍵字: a, b\n\n內容\n"
    e = parse_text(text, "c")[0]
    assert e.source == "某處"
    assert e.keywords == ["a", "b"]
    assert e.content == "內容"


def test_duplicate_title_raises():
    with pytest.raises(DuplicateTitleError, match="標題重複"):
        parse_text("## A\n內容\n## A\n內容2\n", "c")


def test_empty_content_skipped_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        entries = parse_text("## 空的\n關鍵字：x\n\n## 有內容\n內容\n", "c")
    assert [e.title for e in entries] == ["有內容"]
    assert "內容空白" in caplog.text


def test_html_comment_lines_are_ignored():
    text = "## 標題\n關鍵字：a\n<!-- 草稿｜信心：中 -->\n內容\n<!-- 另一個備註 -->\n第二行\n"
    e = parse_text(text, "c")[0]
    assert e.keywords == ["a"]
    assert e.content == "內容\n第二行"


def test_headings_only_counts_as_empty(caplog):
    with caplog.at_level(logging.WARNING):
        entries = parse_text("## 英雄\n\n### 英雄一轉\n\n### 英雄二轉\n", "c")
    assert entries == []
    assert "內容空白" in caplog.text


def test_short_entry_is_one_chunk():
    e = Entry("誤鎖", "標題", "內容", ["鎖住", "被鎖"], "Discord")
    [c] = split_entry(e)
    assert c.id == "誤鎖/標題#0"
    assert c.display_text == "標題\n內容"
    # embed_text 含類別、標題、關鍵字、內容；display_text 不含關鍵字、來源
    assert c.embed_text == "誤鎖\n標題\n關鍵字：鎖住、被鎖\n內容"
    assert "鎖住" not in c.display_text and "Discord" not in c.display_text
    assert c.content_hash == hashlib.sha256(c.embed_text.encode("utf-8")).hexdigest()


def test_long_entry_split_by_subheadings():
    content = "開頭說明\n### 第一步\n" + "甲" * 500 + "\n### 第二步\n" + "乙" * 500
    chunks = split_entry(Entry("任務", "二轉", content))
    assert [c.id for c in chunks] == ["任務/二轉#0", "任務/二轉#1", "任務/二轉#2"]
    assert chunks[0].display_text == "二轉\n開頭說明"
    assert chunks[1].display_text.startswith("二轉 > 第一步\n甲")
    assert chunks[2].display_text.startswith("二轉 > 第二步\n乙")
    assert chunks[1].embed_text.startswith("任務\n二轉 > 第一步\n")


def test_long_entry_without_subheadings_still_split():
    paras = "\n\n".join("字" * 300 for _ in range(4))  # 約 1200 字、沒有 ###
    chunks = split_entry(Entry("c", "t", paras))
    assert len(chunks) > 1
    for c in chunks:
        body = c.display_text.split("\n", 1)[1]
        assert len(body) <= 800


def test_exactly_800_is_one_chunk():
    assert len(split_entry(Entry("c", "t", "字" * 800))) == 1
    assert len(split_entry(Entry("c", "t", "字" * 801))) == 2


def test_parse_file_uses_filename_as_category(tmp_path):
    p = tmp_path / "練等.md"
    p.write_text("## 去哪練\n內容\n", encoding="utf-8")
    assert parse_file(p)[0].category == "練等"


def test_load_chunks_ignores_subfolders(tmp_path):
    (tmp_path / "a.md").write_text("## x\n內容\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("## y\n內容\n", encoding="utf-8")
    assert [c.id for c in load_chunks(tmp_path)] == ["a/x#0"]


def test_example_data_parses():
    chunks = load_chunks("data/examples")
    assert {c.category for c in chunks} == {"練等", "誤鎖", "任務"}
