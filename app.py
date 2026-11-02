"""FastAPI：問答（SSE 串流）、回饋、健康檢查。

啟動：uvicorn app:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, AsyncIterator, Awaitable, Callable, Literal, Protocol, Sequence

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from cache import cache_key
from config import Settings, get_settings
from db import QADB, normalize_question
from llm import LLM, get_llm
from prompts import ERROR_MESSAGE, NO_MATCH_MESSAGE, Prompt, build_messages, classify_answer, load_prompt
from ratelimit import SUGGEST_MESSAGES, RateLimiter, client_ip, rate_key
from retrieval import ChromaStore, Hit, embed
from security import SecurityHeadersMiddleware, build_csp
from topics import build_topics, read_md_info

logger = logging.getLogger("maplestar")

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
BUSY_MESSAGE = "目前人有點多，稍後再試一次。"

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class SearchStore(Protocol):
    def search(self, query_vec: Sequence[float], k: int) -> list[Hit]: ...
    def count(self) -> int: ...
    def list_entries(self) -> list[tuple[str, str]]: ...


@dataclass
class Deps:
    settings: Settings
    store: SearchStore
    embed_fn: EmbedFn
    llm: LLM
    db: QADB
    prompt: Prompt
    check_ollama: Callable[[], Awaitable[bool]]


# ---- request bodies ----


class AskBody(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=200)]


class FeedbackBody(BaseModel):
    log_id: int
    token: Annotated[str, StringConstraints(min_length=8, max_length=64)]
    helpful: bool
    reason: Literal["答錯", "沒回答到問題", "資料沒有", "其他"] | None = None


class SuggestBody(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=200)]
    source: Literal["footer", "no_match", "not_found", "feedback"] = "footer"
    log_id: int | None = None


def _too_many(decision) -> JSONResponse:
    return JSONResponse(
        {"message": decision.message, "retry_after": decision.retry_after},
        status_code=429,
        headers={"Retry-After": str(decision.retry_after)},
    )


# ---- 問答串流 ----


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sources(hits: Sequence[Hit]) -> list[dict]:
    seen, out = set(), []
    for h in hits:
        key = (h.title, h.category)
        if key not in seen:
            seen.add(key)
            out.append({"title": h.title, "category": h.category})
    return out


async def _lookup_cache(deps: Deps, key: str) -> str | None:
    if not deps.settings.cache_enabled:
        return None
    try:
        return await deps.db.find_cached_answer(key, deps.settings.cache_ttl_hours)
    except Exception:
        logger.exception("查快取失敗（改成直接問模型）")
        return None


async def ask_stream(deps: Deps, question: str) -> AsyncIterator[str]:
    """產生 SSE：meta → token... → done。結束時（包含斷線、錯誤）一定 finish_log。"""
    s = deps.settings
    started = time.perf_counter()
    elapsed_ms = lambda: int((time.perf_counter() - started) * 1000)  # noqa: E731

    log_id = None
    token = secrets.token_urlsafe(12)  # 回饋時要帶回來，避免別人用連號的 log_id 亂按
    try:
        log_id = await deps.db.create_log(
            question, deps.llm.provider, deps.llm.model, deps.prompt.version, feedback_token=token
        )
    except Exception:
        logger.exception("建立紀錄失敗（不影響回答）")

    status = "error"
    parts: list[str] = []
    hits: list[Hit] = []
    first_token_ms = None
    meta_sent = False
    key = None
    cached = False
    try:
        vec = (await deps.embed_fn([question]))[0]
        hits = await asyncio.to_thread(deps.store.search, vec, s.top_k)
        used = [h for h in hits if h.score >= s.score_threshold]

        meta_sent = True
        yield sse("meta", {"log_id": log_id, "token": token, "sources": _sources(used)})

        if not used:
            # 最高分低於門檻：不呼叫模型
            status = "no_match"
            parts.append(NO_MATCH_MESSAGE)
            yield sse("token", {"t": NO_MATCH_MESSAGE})
        else:
            key = cache_key(
                normalize_question(question), deps.prompt.version, deps.llm.provider, deps.llm.model, used
            )
            hit = await _lookup_cache(deps, key)
            if hit is not None:
                # 相同問題、相同資料：直接回上次的答案，不呼叫模型
                cached = True
                status = "answered"
                first_token_ms = elapsed_ms()
                parts.append(hit)
                yield sse("token", {"t": hit})
            else:
                messages = build_messages(deps.prompt.system, question, used)
                async for t in deps.llm.stream(messages):
                    if first_token_ms is None:
                        first_token_ms = elapsed_ms()
                    parts.append(t)
                    yield sse("token", {"t": t})
                status = classify_answer("".join(parts))
        yield sse("done", {"status": status, "cached": cached})
    except (asyncio.CancelledError, GeneratorExit):
        status = "aborted"  # 使用者中途斷線
        raise
    except Exception:
        logger.exception("回答失敗")
        status = "error"
        if not meta_sent:
            yield sse("meta", {"log_id": log_id, "token": token, "sources": []})
        yield sse("done", {"status": "error", "message": ERROR_MESSAGE})
    finally:
        if log_id is not None:
            # 斷線時這個 task 已被取消，要 shield 才寫得進去
            with anyio.CancelScope(shield=True):
                try:
                    await deps.db.finish_log(
                        log_id,
                        status=status,
                        answer="".join(parts),
                        retrieved=[{"chunk": h.chunk_id, "score": round(h.score, 4)} for h in hits],
                        top_score=hits[0].score if hits else None,
                        category=hits[0].category if hits else None,
                        latency_ms=elapsed_ms(),
                        first_token_ms=first_token_ms,
                        cache_key=key,
                        cached=cached,
                    )
                except Exception:
                    logger.exception("寫入紀錄失敗（不影響回答）")


# ---- app ----


def build_deps(settings: Settings, client: httpx.AsyncClient) -> Deps:
    async def embed_fn(texts: list[str]) -> list[list[float]]:
        return await embed(
            texts, url=settings.ollama_url, model=settings.embed_model, client=client,
            keep_alive=settings.embed_keep_alive,
        )

    async def check_ollama() -> bool:
        try:
            resp = await client.get(f"{settings.ollama_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    store = ChromaStore(settings.chroma_dir, settings.chroma_collection)
    if store.embed_model not in (None, settings.embed_model):
        logger.warning("向量庫是用 %s 建的，現在 EMBED_MODEL=%s，請重跑 ingest.py", store.embed_model, settings.embed_model)
    return Deps(
        settings=settings,
        store=store,
        embed_fn=embed_fn,
        llm=get_llm(settings, client),
        db=QADB(settings.db_path),
        prompt=load_prompt(settings.prompt_file),
        check_ollama=check_ollama,
    )


class SlotStreamingResponse(StreamingResponse):
    """串流結束（包含斷線、還沒開始就失敗）一定把同時生成的名額還回去。"""

    def __init__(self, *args, slots: asyncio.Semaphore, **kwargs):
        super().__init__(*args, **kwargs)
        self._slots = slots

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._slots.release()


def create_app(deps: Deps | None = None) -> FastAPI:
    settings = deps.settings if deps is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if deps is not None:
            app.state.deps = deps
            await deps.db.init()
            yield
            return
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as client:
            app.state.deps = build_deps(settings, client)
            await app.state.deps.db.init()
            yield

    production = settings.env == "production"
    app = FastAPI(
        title="maplestar_lurenbot",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    # CSP 只放行首頁本身的 inline script / style（啟動時依檔案內容算雜湊）
    index_html = INDEX_HTML.read_text(encoding="utf-8") if INDEX_HTML.exists() else ""
    app.add_middleware(SecurityHeadersMiddleware, csp=build_csp(index_html))
    app.state.limiter = RateLimiter(settings.cooldown_seconds, settings.hourly_limit, settings.daily_limit)
    app.state.slots = asyncio.Semaphore(settings.max_concurrent)
    # 許願另外算，不會吃到問答的額度
    app.state.suggest_limiter = RateLimiter(
        settings.suggest_cooldown_seconds,
        settings.suggest_hourly_limit,
        settings.suggest_daily_limit,
        messages=SUGGEST_MESSAGES,
    )

    def ip_of(request: Request) -> str:
        return rate_key(client_ip(request.headers, request.client.host if request.client else None))

    @app.post("/api/ask")
    async def ask(body: AskBody, request: Request):
        decision = app.state.limiter.check(ip_of(request))
        if not decision.allowed:
            # 被擋的請求不寫入 qa_logs，只在記憶體計數
            return _too_many(decision)

        try:
            await asyncio.wait_for(app.state.slots.acquire(), timeout=settings.queue_timeout)
        except asyncio.TimeoutError:
            return JSONResponse({"message": BUSY_MESSAGE}, status_code=503)

        return SlotStreamingResponse(
            ask_stream(app.state.deps, body.question),
            slots=app.state.slots,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/feedback")
    async def feedback(body: FeedbackBody):
        result = await app.state.deps.db.set_feedback(body.log_id, body.helpful, body.reason, body.token)
        if result == "not_found":
            raise HTTPException(404, "找不到這筆問答")
        if result == "duplicate":
            raise HTTPException(409, "這題已經回饋過了")
        return {"ok": True}

    @app.post("/api/suggest")
    async def suggest(body: SuggestBody, request: Request):
        decision = app.state.suggest_limiter.check(ip_of(request))
        if not decision.allowed:
            return _too_many(decision)
        await app.state.deps.db.add_suggestion(body.text, body.source, body.log_id)
        return {"ok": True}

    @app.get("/api/topics")
    async def topics():
        d: Deps = app.state.deps
        entries = await asyncio.to_thread(d.store.list_entries)
        md_info = await asyncio.to_thread(read_md_info, d.settings.data_dir)
        return build_topics(entries, md_info)

    @app.get("/health")
    async def health():
        d: Deps = app.state.deps
        try:
            chunks = await asyncio.to_thread(d.store.count)
            chroma_ok = True
        except Exception:
            logger.exception("Chroma 健康檢查失敗")
            chunks, chroma_ok = None, False
        ollama_ok = await d.check_ollama()
        ok = chroma_ok and ollama_ok
        today = app.state.limiter.today
        body = {
            "status": "ok" if ok else "degraded",
            "chroma": chroma_ok,
            "ollama": ollama_ok,
            "chunks": chunks,
            "today": {"date": today.day, "asked": today.asked, "blocked": dict(today.blocked)},
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/")
    async def index():
        if INDEX_HTML.exists():
            return FileResponse(INDEX_HTML, headers={"Cache-Control": "no-cache"})
        return PlainTextResponse("楓星問答機器人— 開發中")

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
