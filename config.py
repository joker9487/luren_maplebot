"""設定全部從環境變數讀（啟動時會先載入 .env）。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_url: str
    embed_model: str
    embed_keep_alive: str
    score_threshold: float
    top_k: int
    data_dir: str
    chroma_dir: str
    chroma_collection: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    ollama_num_ctx: int
    openai_api_key: str
    openai_reasoning_effort: str
    db_path: str
    prompt_file: str
    env: str
    cooldown_seconds: float
    hourly_limit: int
    daily_limit: int
    max_concurrent: int
    queue_timeout: float
    cache_enabled: bool
    cache_ttl_hours: float
    suggest_cooldown_seconds: float
    suggest_hourly_limit: int
    suggest_daily_limit: int


def get_settings() -> Settings:
    return Settings(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        embed_model=os.getenv("EMBED_MODEL", "bge-m3"),
        embed_keep_alive=os.getenv("EMBED_KEEP_ALIVE", "24h"),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.55")),
        top_k=int(os.getenv("TOP_K", "4")),
        data_dir=os.getenv("DATA_DIR", "data"),
        chroma_dir=os.getenv("CHROMA_DIR", "storage/chroma"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "maplestar"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
        db_path=os.getenv("DB_PATH", "storage/qa.db"),
        prompt_file=os.getenv("PROMPT_FILE", "prompts/answer_v3.txt"),
        env=os.getenv("ENV", "development").lower(),
        cooldown_seconds=float(os.getenv("COOLDOWN_SECONDS", "5")),
        hourly_limit=int(os.getenv("HOURLY_LIMIT", "30")),
        daily_limit=int(os.getenv("DAILY_LIMIT", "1000")),
        max_concurrent=int(os.getenv("MAX_CONCURRENT", "3")),
        queue_timeout=float(os.getenv("QUEUE_TIMEOUT", "20")),
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
        cache_ttl_hours=float(os.getenv("CACHE_TTL_HOURS", "72")),
        suggest_cooldown_seconds=float(os.getenv("SUGGEST_COOLDOWN_SECONDS", "3")),
        suggest_hourly_limit=int(os.getenv("SUGGEST_HOURLY_LIMIT", "5")),
        suggest_daily_limit=int(os.getenv("SUGGEST_DAILY_LIMIT", "500")),
    )
