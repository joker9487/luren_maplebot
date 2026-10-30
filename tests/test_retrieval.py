import asyncio
import json

import httpx
import pytest

from retrieval import embed


def run_embed(handler, texts):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await embed(texts, url="http://ollama", model="bge-m3", client=client)

    return asyncio.run(go())


def test_embed_sends_whole_list_in_one_request():
    seen = []

    def handler(request):
        seen.append(request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    vecs = run_embed(handler, ["a", "b", "c"])
    assert len(seen) == 1
    assert seen[0].url.path == "/api/embed"
    assert json.loads(seen[0].content) == {"model": "bge-m3", "input": ["a", "b", "c"]}
    assert vecs == [[1.0, 0.0]] * 3


def test_embed_empty_list_makes_no_request():
    def handler(request):
        raise AssertionError("不該送出請求")

    assert run_embed(handler, []) == []


def test_embed_http_error_raises():
    with pytest.raises(httpx.HTTPStatusError):
        run_embed(lambda r: httpx.Response(500), ["a"])


def test_embed_count_mismatch_raises():
    with pytest.raises(RuntimeError, match="數量不符"):
        run_embed(lambda r: httpx.Response(200, json={"embeddings": [[1.0]]}), ["a", "b"])


def test_embed_sends_keep_alive_when_set():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await embed(["a"], url="http://o", model="bge-m3", client=client, keep_alive="24h")

    asyncio.run(go())
    assert seen[0]["keep_alive"] == "24h"
