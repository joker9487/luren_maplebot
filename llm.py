"""可切換的回答模型介面：OllamaLLM / OpenAILLM / FakeLLM。

每個實作都有 `provider`、`model` 屬性，以及
`stream(messages) -> AsyncIterator[str]`（一次吐一段文字）。
模型只回文字，不給任何工具。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Protocol

import httpx

from config import Settings


class LLM(Protocol):
    provider: str
    model: str

    def stream(self, messages: list[dict]) -> AsyncIterator[str]: ...


class OllamaLLM:
    provider = "ollama"

    def __init__(
        self, client: httpx.AsyncClient, url: str, model: str, temperature: float, max_tokens: int, num_ctx: int
    ):
        self._client = client
        self._url = url
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx  # Ollama 預設的 context 偏小，超過會默默截掉前面的資料

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens, "num_ctx": self._num_ctx},
        }
        async with self._client.stream("POST", f"{self._url}/api/chat", json=payload) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise RuntimeError(f"Ollama /api/chat 回 {resp.status_code}：{resp.text[:200]}")
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"Ollama 錯誤：{data['error']}")
                text = data.get("message", {}).get("content", "")
                if text:
                    yield text
                if data.get("done"):
                    break


class OpenAILLM:
    provider = "openai"

    def __init__(self, client: Any, model: str, temperature: float, max_tokens: int, reasoning_effort: str):
        self._client = client  # openai.AsyncOpenAI
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    def params(self) -> dict:
        """依模型決定參數：gpt-5 系列是推理模型，不收 temperature、改用 max_completion_tokens。"""
        if self.model.startswith("gpt-5"):
            return {"reasoning_effort": self._reasoning_effort, "max_completion_tokens": self._max_tokens}
        return {"temperature": self._temperature, "max_tokens": self._max_tokens}

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **self.params()
        )
        async for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


class FakeLLM:
    """測試用：依序吐出固定的片段；可指定吐到第幾段時丟例外。"""

    provider = "fake"

    def __init__(self, parts: list[str] | None = None, model: str = "fake", fail_after: int | None = None):
        self.parts = parts if parts is not None else ["這是", "假的", "回答。"]
        self.model = model
        self.fail_after = fail_after
        self.calls: list[list[dict]] = []

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        self.calls.append(messages)
        for i, part in enumerate(self.parts):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("FakeLLM 故意失敗")
            yield part


def get_llm(settings: Settings, http_client: httpx.AsyncClient) -> LLM:
    if settings.llm_provider == "ollama":
        return OllamaLLM(
            http_client,
            settings.ollama_url,
            settings.llm_model,
            settings.llm_temperature,
            settings.llm_max_tokens,
            settings.ollama_num_ctx,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai 但沒有設定 OPENAI_API_KEY")
        from openai import AsyncOpenAI

        return OpenAILLM(
            # 沒設 timeout 的話，OpenAI 卡住時這題會一直占著同時生成的名額
            AsyncOpenAI(api_key=settings.openai_api_key, timeout=httpx.Timeout(60, connect=10), max_retries=1),
            settings.llm_model,
            settings.llm_temperature,
            settings.llm_max_tokens,
            settings.openai_reasoning_effort,
        )
    raise RuntimeError(f"不認得的 LLM_PROVIDER：{settings.llm_provider}（只能是 ollama 或 openai）")
