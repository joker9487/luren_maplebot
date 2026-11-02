"""Prompt 組裝與回答狀態判斷（純函式）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from retrieval import Hit

# 模型「資料裡沒有答案」時要回的固定句（prompt 檔裡也要有這句）
NOT_FOUND = "目前資料裡沒有這題的答案。"
# 檢索分數低於門檻、不呼叫模型時回的固定句
NO_MATCH_MESSAGE = "這題目前資料裡找不到，可能不是遊戲相關的問題，或是還沒整理到。"
# 發生錯誤時給玩家看的訊息
ERROR_MESSAGE = "抱歉，系統出了點問題，請稍後再試一次。"


@dataclass(frozen=True)
class Prompt:
    version: str  # 檔名（不含副檔名），例如 answer_v1
    system: str


def load_prompt(path: str | Path) -> Prompt:
    path = Path(path)
    system = path.read_text(encoding="utf-8").strip()
    if NOT_FOUND not in system:
        raise ValueError(f"{path.name} 裡沒有固定句「{NOT_FOUND}」，狀態判斷會失準")
    return Prompt(version=path.stem, system=system)


def build_messages(system: str, question: str, hits: Sequence[Hit]) -> list[dict]:
    docs = "\n".join(f"[{i}] {h.display_text}" for i, h in enumerate(hits, 1))
    user = f"<資料>\n{docs}\n</資料>\n\n問題：{question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# 模型偶爾會回簡體版的固定句
_TO_TRAD = str.maketrans({"资": "資", "里": "裡", "裏": "裡", "这": "這", "题": "題", "没": "沒"})
_STRIP_RE = re.compile(r"[\s「」『』\"'“”。.!！~～]+")


def _core(text: str) -> str:
    return _STRIP_RE.sub("", text.translate(_TO_TRAD))


# 開頭就說資料沒有（模型沒照固定句、自己寫了一段），例如「結論：目前資料裡沒有提到…」
_LEADING_NOT_FOUND_RE = re.compile(r"^(結論[:：])?(目前|現有)?的?資料(裡|中|內)?(並|都)?(沒有|未|無)")


def classify_answer(answer: str) -> str:
    """回答一開頭就是「資料裡沒有」→ not_found；其他 → answered。

    比對前去掉空白、引號、句號，並把簡體字轉回來，
    這樣模型多一個標點、加了引號，或在固定句後面多講幾句都判得出來。
    「只回答一部分、最後補一句其他部分沒有」不算，因為開頭是答案。
    """
    core = _core(answer)
    if core.startswith(_core(NOT_FOUND)) or _LEADING_NOT_FOUND_RE.match(core):
        return "not_found"
    return "answered"
