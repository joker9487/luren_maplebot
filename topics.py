"""「目前收錄的內容」清單（給前端的展開式清單用）。

- 標題：從向量庫讀，也就是機器人「實際找得到」的內容；跑完 ingest 就自動更新
- 簡介：每個 md 檔第一個 `##` 之前的說明文字
- 順序：照 md 檔裡的順序；庫裡有、檔案裡已經沒有的（還沒重跑 ingest）排在最後
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from chunking import parse_intro, parse_text

logger = logging.getLogger(__name__)


def read_md_info(data_dir: str | Path) -> dict[str, tuple[str, list[str]]]:
    """{類別: (簡介, 檔案裡的標題順序)}；某個檔案格式有錯就只略過它的排序。"""
    info: dict[str, tuple[str, list[str]]] = {}
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        try:
            titles = [e.title for e in parse_text(text, path.stem, name=path.name)]
        except ValueError:
            logger.warning("%s 格式有錯，收錄清單先不照檔案排序", path.name)
            titles = []
        info[path.stem] = (parse_intro(text), titles)
    return info


def build_topics(entries: Sequence[tuple[str, str]], md_info: dict[str, tuple[str, list[str]]]) -> dict:
    """（純函式）把庫裡的 (類別, 標題) 整理成依類別分組的清單。"""
    by_cat: dict[str, list[str]] = {}
    for category, title in entries:
        by_cat.setdefault(category, []).append(title)

    cat_order = list(md_info)
    categories = []
    for category in sorted(by_cat, key=lambda c: (cat_order.index(c) if c in cat_order else len(cat_order), c)):
        intro, file_titles = md_info.get(category, ("", []))
        pos = {t: i for i, t in enumerate(file_titles)}
        titles = sorted(by_cat[category], key=lambda t: (pos.get(t, len(pos)), t))
        categories.append({"name": category, "intro": intro, "titles": titles})
    return {"categories": categories, "total": sum(len(c["titles"]) for c in categories)}
