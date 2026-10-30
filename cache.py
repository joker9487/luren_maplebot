"""相同問題快取的 key（純函式）。

快取本身存在 qa_logs（見 db.find_cached_answer）：同一個 key、最近 CACHE_TTL_HOURS 內
有模型正常回答過、而且沒有人按 👎，就直接回那次的答案，不再呼叫模型。

key 包含「問題正規化 + prompt 版本 + 模型 + 這次用到的每一塊資料的 hash」，所以
- 問法只差空白、全半形、結尾標點 → 同一個 key
- 知識庫內容改了（hash 變）、檢索到的塊不一樣、換 prompt 或模型 → 自動變成新 key
不需要另外寫「資料更新時清快取」的邏輯。
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from retrieval import Hit


def cache_key(question_norm: str, prompt_version: str, provider: str, model: str, hits: Sequence[Hit]) -> str:
    chunks = sorted((h.chunk_id, h.content_hash) for h in hits)
    raw = json.dumps([question_norm, prompt_version, provider, model, chunks], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
