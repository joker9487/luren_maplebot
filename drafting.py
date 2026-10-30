"""Discord 討論 → 知識庫草稿（核心邏輯，LLM 和查重用注入的）。

流程：
1. 讀討論紀錄（DiscordChatExporter 匯出的 JSON，或直接複製貼上的文字）→ 一行一則訊息
2. 切成好幾段（每段不超過 max_chars，前後重疊幾行，避免一個問答被切斷）
3. 每段丟給 LLM，要它只整理「討論裡有明確答案」的資訊，回 JSON
4. 合併各段結果（同類別同標題只留一條）
5. （選用）跟現有知識庫比對，標出可能重複的條目
6. 輸出成 data/ 的格式，審核資訊寫在 <!-- --> 註解裡（ingest 會忽略）

草稿只是給人審核的起點：模型可能漏掉、誤解或過度整理，一定要人看過再搬進 data/。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from chunking import Entry, parse_text, split_entry

CONFIDENCE = ("高", "中", "低")

# 查重：給 embed_text，回 (相似的條目 id, 分數) 或 None
DupFinder = Callable[[list[str]], Awaitable[list[tuple[str, float] | None]]]


@dataclass
class DraftEntry:
    category: str
    title: str
    content: str
    keywords: list[str] = field(default_factory=list)
    confidence: str = "中"
    notes: str = ""
    duplicate_of: tuple[str, float] | None = None

    def as_entry(self) -> Entry:
        return Entry(self.category, self.title, self.content, self.keywords)


# ---- 1. 讀討論紀錄 ----


def load_discussion(path: str | Path) -> tuple[list[str], str]:
    """回傳（一行一則訊息, 來源說明）。支援 DiscordChatExporter 的 JSON 和純文字。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        return parse_exporter_json(json.loads(text))
    return [l.rstrip() for l in text.splitlines() if l.strip()], f"Discord 討論（{path.stem}）"


def parse_exporter_json(data: dict) -> tuple[list[str], str]:
    """DiscordChatExporter 的 JSON：messages[].timestamp / author.name / content。

    玩家名字只用來讓模型分辨「誰在回答誰」，prompt 會要求草稿裡不能出現名字。
    """
    lines = []
    for m in data.get("messages", []):
        content = (m.get("content") or "").strip()
        if not content:
            continue
        author = (m.get("author") or {})
        name = author.get("nickname") or author.get("name") or "?"
        day = (m.get("timestamp") or "")[:10]
        lines.append(f"[{day}] {name}：{content}".replace("\n", " ／ "))
    channel = (data.get("channel") or {}).get("name") or "Discord"
    days = sorted({l[1:11] for l in lines if l.startswith("[2")})
    span = f" {days[0]}～{days[-1]}" if days else ""
    return lines, f"Discord #{channel}{span}"


# ---- 2. 切段 ----


def make_windows(lines: Sequence[str], max_chars: int = 6000, overlap: int = 8) -> list[str]:
    """依行切成每段 ≤ max_chars 字，下一段從上一段最後 overlap 行開始。"""
    windows: list[str] = []
    start = 0
    while start < len(lines):
        size, end = 0, start
        while end < len(lines) and (end == start or size + len(lines[end]) + 1 <= max_chars):
            size += len(lines[end]) + 1
            end += 1
        windows.append("\n".join(lines[start:end]))
        if end >= len(lines):
            break
        start = max(end - overlap, start + 1)
    return windows


# ---- 3. 呼叫 LLM、解析回應 ----


def build_draft_messages(system_template: str, categories: Sequence[str], window: str, i: int, n: int) -> list[dict]:
    system = system_template.replace("{categories}", "、".join(categories) or "（目前沒有）")
    user = f"以下是 Discord 討論紀錄（第 {i}/{n} 段）：\n<討論>\n{window}\n</討論>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json(text: str) -> dict:
    """模型有時會包 ```json 或多講幾句；抓第一個 { 到最後一個 }。"""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("回應裡找不到 JSON")
    return json.loads(text[start : end + 1])


_BAD_NAME_RE = re.compile(r'[\\/:*?"<>|#]')


def normalize_entries(data: dict) -> list[DraftEntry]:
    """把模型的 JSON 轉成 DraftEntry；缺標題或內容的丟掉，欄位清乾淨。"""
    out = []
    for raw in data.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip().lstrip("#").strip()
        content = str(raw.get("content") or "").strip()
        if not title or not content:
            continue
        category = _BAD_NAME_RE.sub("", str(raw.get("category") or "").strip()) or "未分類"
        keywords = raw.get("keywords") or []
        if isinstance(keywords, str):
            keywords = re.split(r"[、,，]", keywords)
        keywords = [str(k).strip() for k in keywords if str(k).strip()]
        confidence = str(raw.get("confidence") or "中").strip()
        out.append(
            DraftEntry(
                category=category,
                title=title,
                # 內容裡如果有 ## 會被當成新條目，降成 ###
                content=re.sub(r"^##(?!#)", "###", content, flags=re.M),
                keywords=keywords,
                confidence=confidence if confidence in CONFIDENCE else "中",
                notes=str(raw.get("notes") or "").strip(),
            )
        )
    return out


def merge_entries(entries: Sequence[DraftEntry]) -> list[DraftEntry]:
    """各段重疊的部分可能整理出同一條：同類別同標題只留內容最長的，關鍵字合併。"""
    merged: dict[tuple[str, str], DraftEntry] = {}
    for e in entries:
        key = (e.category, re.sub(r"\s+", "", e.title))
        if key not in merged:
            merged[key] = e
            continue
        old = merged[key]
        best = e if len(e.content) > len(old.content) else old
        keywords = list(dict.fromkeys(old.keywords + e.keywords))
        notes = "；".join(n for n in dict.fromkeys([old.notes, e.notes]) if n)
        merged[key] = replace(best, keywords=keywords, notes=notes)
    return list(merged.values())


async def draft_entries(
    windows: Sequence[str],
    llm,
    system_template: str,
    categories: Sequence[str],
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[DraftEntry]:
    """逐段呼叫 LLM；某段失敗就跳過那段（交給 on_error 記錄），不影響其他段。"""
    found: list[DraftEntry] = []
    for i, window in enumerate(windows, 1):
        messages = build_draft_messages(system_template, categories, window, i, len(windows))
        try:
            text = "".join([t async for t in llm.stream(messages)])
            found.extend(normalize_entries(extract_json(text)))
        except Exception as e:  # noqa: BLE001
            if on_error:
                on_error(i, e)
    return merge_entries(found)


async def mark_duplicates(entries: Sequence[DraftEntry], find: DupFinder, threshold: float) -> None:
    """跟現有知識庫比對，分數 ≥ threshold 的標成可能重複。"""
    if not entries:
        return
    texts = [split_entry(e.as_entry())[0].embed_text for e in entries]
    for e, hit in zip(entries, await find(texts)):
        if hit and hit[1] >= threshold:
            e.duplicate_of = hit


_AUTHOR_RE = re.compile(r"^\[[^\]]*\]\s*(.+?)[：:]")


def speaker_names(lines: Sequence[str]) -> set[str]:
    """從「[日期] 名字：訊息」抓出所有發言者名字。"""
    return {m.group(1).strip() for l in lines if (m := _AUTHOR_RE.match(l)) and len(m.group(1).strip()) >= 2}


def flag_names(entries: Sequence[DraftEntry], names: set[str]) -> None:
    """prompt 有要求不寫玩家名字，但模型不一定照做：內容或標題出現名字就在備註警告。"""
    for e in entries:
        found = sorted(n for n in names if n in e.content or n in e.title)
        if found:
            warn = "⚠ 內容出現玩家名字：" + "、".join(found) + "，請刪掉"
            e.notes = f"{warn}；{e.notes}" if e.notes else warn


# ---- 6. 輸出 ----


def render_category(category: str, entries: Sequence[DraftEntry], source: str, generated: str) -> str:
    lines = [
        f"# {category}（草稿）",
        f"<!-- {generated} 由 {source} 自動整理。審核、修改後把要的條目搬到 data/{category}.md -->",
        "",
    ]
    for e in entries:
        lines.append(f"## {e.title}")
        if e.keywords:
            lines.append("關鍵字：" + "、".join(e.keywords))
        lines.append(f"來源：{source}（草稿）")
        review = [f"信心：{e.confidence}"]
        if e.duplicate_of:
            review.append(f"可能重複：{e.duplicate_of[0].split('#')[0]}（{e.duplicate_of[1]:.2f}）")
        if e.notes:
            review.append(f"備註：{e.notes}")
        lines.append("<!-- 草稿｜" + "｜".join(review).replace("-->", "—>").replace("\n", " ") + " -->")
        lines.append(e.content)
        lines.append("")
    return "\n".join(lines)


def write_drafts(
    entries: Sequence[DraftEntry], out_dir: str | Path, source: str, generated: str
) -> list[Path]:
    """一個類別一個檔；寫完用正式的解析器再讀一次，確定格式搬過去就能用。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[DraftEntry]] = {}
    for e in entries:
        by_cat.setdefault(e.category, []).append(e)
    paths = []
    for category, items in sorted(by_cat.items()):
        text = render_category(category, merge_entries(items), source, generated)
        parse_text(text, category, name=f"草稿 {category}")  # 格式有問題會在這裡報錯
        path = out_dir / f"{category}.md"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
