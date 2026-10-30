"""知識庫 md 的解析與切段（純函式，不依賴 Chroma / FastAPI）。

格式：一個類別一個 md 檔（檔名 = 類別），每個 `## 標題` 是一條資料。
標題下面緊接的「關鍵字：」「來源：」是選填欄位，其餘是內容。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CHARS = 800

_ENTRY_RE = re.compile(r"^##(?!#)\s*(.*)$")
_H1_RE = re.compile(r"^#(?!#)")
_SUB_RE = re.compile(r"^###(?!#)\s*(.*)$")
_FIELD_RE = re.compile(r"^(關鍵字|來源)\s*[：:]\s*(.*)$")
_KEYWORD_SEP_RE = re.compile(r"[、,，;；]")
# 單行 HTML 註解：給人看的備註（例如草稿的審核資訊），不進檢索也不給模型
_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")


class DuplicateTitleError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    category: str
    title: str
    content: str
    keywords: list[str] = field(default_factory=list)
    source: str = ""


@dataclass(frozen=True)
class Chunk:
    id: str
    category: str
    title: str
    source: str
    embed_text: str
    display_text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()


def parse_text(text: str, category: str, name: str = "") -> list[Entry]:
    """把一個 md 檔的文字解析成多條 Entry。name 只用在錯誤訊息。"""
    name = name or category
    raw: list[tuple[str, list[str]]] = []  # (標題, 標題下的行)
    for line in text.splitlines():
        m = _ENTRY_RE.match(line)
        if m:
            title = m.group(1).strip()
            if not title:
                raise ValueError(f"{name}: 有 `##` 沒寫標題")
            raw.append((title, []))
        elif _H1_RE.match(line) or _COMMENT_RE.match(line):
            continue
        elif raw:
            raw[-1][1].append(line)

    entries: list[Entry] = []
    seen: set[str] = set()
    for title, lines in raw:
        if title in seen:
            raise DuplicateTitleError(f"{name}: 標題重複「{title}」")
        seen.add(title)

        keywords: list[str] = []
        source = ""
        i = 0
        # 標題下緊接的選填欄位（中間可以有空行）
        while i < len(lines):
            stripped = lines[i].strip()
            m = _FIELD_RE.match(stripped)
            if not stripped:
                i += 1
            elif m:
                if m.group(1) == "關鍵字":
                    keywords = [k.strip() for k in _KEYWORD_SEP_RE.split(m.group(2)) if k.strip()]
                else:
                    source = m.group(2).strip()
                i += 1
            else:
                break

        content = "\n".join(lines[i:]).strip()
        # 只有 ### 小標、沒有內文也算空白
        if not any(l.strip() and not _SUB_RE.match(l) for l in lines[i:]):
            logger.warning("%s: 「%s」內容空白，跳過", name, title)
            continue
        entries.append(Entry(category, title, content, keywords, source))
    return entries


def parse_file(path: str | Path) -> list[Entry]:
    path = Path(path)
    return parse_text(path.read_text(encoding="utf-8"), category=path.stem, name=path.name)


def parse_intro(text: str) -> str:
    """第一個 `##` 之前的說明文字（不含 `#` 標題、註解），拿來當類別簡介。不進檢索。"""
    lines = []
    for line in text.splitlines():
        if _ENTRY_RE.match(line):
            break
        if _H1_RE.match(line) or _COMMENT_RE.match(line) or not line.strip():
            continue
        lines.append(line.strip())
    return "\n".join(lines)


def _split_sections(content: str) -> list[tuple[str | None, str]]:
    """照 `###` 小標切成 (小標, 內容)；第一個小標前的文字小標為 None。"""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in content.splitlines():
        m = _SUB_RE.match(line)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(sub, "\n".join(lines).strip()) for sub, lines in sections if "\n".join(lines).strip()]


def _pack(text: str, limit: int) -> list[str]:
    """把太長的文字依段落 → 行 → 字數切開，再盡量合併成 ≤ limit 的塊。"""
    if len(text) <= limit:
        return [text]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            units.append(para)
            continue
        for line in para.splitlines():
            line = line.strip()
            units.extend(line[j : j + limit] for j in range(0, len(line), limit))

    out: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > limit:
            out.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        out.append(cur)
    return out


def split_entry(entry: Entry, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """一條 ≤ max_chars 字 → 一塊；超過就照 `###` 小標切，每塊開頭補「標題 > 小標」。"""
    if len(entry.content) <= max_chars:
        pieces = [(entry.title, entry.content)]
    else:
        pieces = []
        for sub, body in _split_sections(entry.content):
            heading = f"{entry.title} > {sub}" if sub else entry.title
            pieces.extend((heading, part) for part in _pack(body, max_chars))

    chunks = []
    for i, (heading, body) in enumerate(pieces):
        embed_lines = [entry.category, heading]
        if entry.keywords:
            embed_lines.append("關鍵字：" + "、".join(entry.keywords))
        embed_lines.append(body)
        chunks.append(
            Chunk(
                id=f"{entry.category}/{entry.title}#{i}",
                category=entry.category,
                title=entry.title,
                source=entry.source,
                embed_text="\n".join(embed_lines),
                display_text=f"{heading}\n{body}",
            )
        )
    return chunks


def chunk_entries(entries: list[Entry], max_chars: int = MAX_CHARS) -> list[Chunk]:
    return [c for e in entries for c in split_entry(e, max_chars)]


def load_chunks(data_dir: str | Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """讀 data_dir 底下的 *.md（不含子資料夾）並切段。"""
    files = sorted(Path(data_dir).glob("*.md"))
    return [c for f in files for c in chunk_entries(parse_file(f), max_chars)]
