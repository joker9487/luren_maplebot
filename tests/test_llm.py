import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from config import get_settings
from llm import FakeLLM, OllamaLLM, OpenAILLM, get_llm

MESSAGES = [{"role": "user", "content": "hi"}]


async def collect(agen):
    return [x async for x in agen]


def run_ollama(handler):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OllamaLLM(client, "http://ollama", "qwen2.5:7b", 0.2, 400, 8192)
            return await collect(llm.stream(MESSAGES))

    return asyncio.run(go())


def test_ollama_streams_and_sends_options():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        lines = [
            {"message": {"content": "你"}, "done": False},
            {"message": {"content": "好"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        return httpx.Response(200, content="\n".join(json.dumps(l) for l in lines) + "\n")

    assert run_ollama(handler) == ["你", "好"]
    assert seen["path"] == "/api/chat"
    assert seen["body"]["stream"] is True
    assert seen["body"]["options"] == {"temperature": 0.2, "num_predict": 400, "num_ctx": 8192}
    assert seen["body"]["messages"] == MESSAGES


def test_ollama_http_error():
    with pytest.raises(RuntimeError, match="404"):
        run_ollama(lambda r: httpx.Response(404, text="model not found"))


def test_ollama_error_line():
    with pytest.raises(RuntimeError, match="Ollama 錯誤"):
        run_ollama(lambda r: httpx.Response(200, content=json.dumps({"error": "boom"}) + "\n"))


# ---- OpenAI ----

class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs

        async def gen():
            for text in ["A", None, "B"]:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
            yield SimpleNamespace(choices=[])  # 最後的 usage 事件沒有 choices

        return gen()


def fake_openai():
    completions = FakeCompletions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_openai_regular_model_params():
    client, comp = fake_openai()
    llm = OpenAILLM(client, "gpt-4o-mini", 0.2, 400, "minimal")
    assert asyncio.run(collect(llm.stream(MESSAGES))) == ["A", "B"]
    assert comp.kwargs["stream"] is True
    assert comp.kwargs["temperature"] == 0.2 and comp.kwargs["max_tokens"] == 400
    assert "reasoning_effort" not in comp.kwargs


def test_openai_gpt5_params():
    llm = OpenAILLM(None, "gpt-5-nano", 0.2, 400, "minimal")
    assert llm.params() == {"reasoning_effort": "minimal", "max_completion_tokens": 400}


# ---- FakeLLM / get_llm ----

def test_fake_llm():
    assert asyncio.run(collect(FakeLLM(["a", "b"]).stream(MESSAGES))) == ["a", "b"]
    with pytest.raises(RuntimeError):
        asyncio.run(collect(FakeLLM(["a", "b"], fail_after=1).stream(MESSAGES)))


def test_get_llm_by_provider():
    base = get_settings()
    client = httpx.AsyncClient()
    assert isinstance(get_llm(replace(base, llm_provider="ollama"), client), OllamaLLM)
    assert isinstance(get_llm(replace(base, llm_provider="openai", openai_api_key="sk-test"), client), OpenAILLM)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm(replace(base, llm_provider="openai", openai_api_key=""), client)
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        get_llm(replace(base, llm_provider="claude"), client)
