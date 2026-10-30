import asyncio
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import Deps, ask_stream, create_app
from config import get_settings
from db import QADB
from llm import FakeLLM
from prompts import ERROR_MESSAGE, NO_MATCH_MESSAGE, NOT_FOUND, load_prompt
from retrieval import Hit


class FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query_vec, k):
        return self.hits[:k]

    def count(self):
        return len(self.hits)

    def list_entries(self):
        return list(dict.fromkeys((h.category, h.title) for h in self.hits))


HIT_A = Hit("誤鎖/物品被誤鎖#0", "誤鎖", "物品被誤鎖", "物品被誤鎖\n到 Discord 開單", 0.8)
HIT_B = Hit("誤鎖/角色被鎖#0", "誤鎖", "角色被鎖", "角色被鎖\n等 30 分鐘", 0.6)
HIT_LOW = Hit("練等/去哪練#0", "練等", "去哪練", "去哪練\n新手村", 0.3)


async def fake_embed(texts):
    return [[0.1, 0.2] for _ in texts]


async def ollama_ok():
    return True


@pytest.fixture
def make_deps(tmp_path):
    def make(hits=(HIT_A, HIT_B, HIT_LOW), llm=None, embed_fn=fake_embed):
        settings = replace(get_settings(), score_threshold=0.5, top_k=4)
        return Deps(
            settings=settings,
            store=FakeStore(list(hits)),
            embed_fn=embed_fn,
            llm=llm or FakeLLM(),
            db=QADB(tmp_path / "qa.db"),
            prompt=load_prompt("prompts/answer_v2.txt"),
            check_ollama=ollama_ok,
        )

    return make


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def ask(deps, question="東西不能交易怎麼辦"):
    with TestClient(create_app(deps)) as client:
        resp = client.post("/api/ask", json={"question": question})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return parse_sse(resp.text)


def log_of(deps, events):
    return deps.db.get_log_sync(events[0][1]["log_id"])


def test_answer_event_order_and_log(make_deps):
    deps = make_deps()
    events = ask(deps)
    assert [e for e, _ in events] == ["meta", "token", "token", "token", "done"]
    meta = events[0][1]
    assert meta["sources"] == [{"title": "物品被誤鎖", "category": "誤鎖"}, {"title": "角色被鎖", "category": "誤鎖"}]
    assert "".join(d["t"] for e, d in events if e == "token") == "這是假的回答。"
    assert events[-1][1] == {"status": "answered", "cached": False}

    log = log_of(deps, events)
    assert log["status"] == "answered"
    assert log["answer"] == "這是假的回答。"
    assert log["question_norm"] == "東西不能交易怎麼辦"
    assert log["category"] == "誤鎖"
    assert log["top_score"] == pytest.approx(0.8)
    assert json.loads(log["retrieved"])[0] == {"chunk": "誤鎖/物品被誤鎖#0", "score": 0.8}
    assert log["provider"] == "fake" and log["prompt_version"] == "answer_v2"
    assert log["latency_ms"] >= 0 and log["first_token_ms"] is not None


def test_only_hits_above_threshold_go_to_model(make_deps):
    llm = FakeLLM()
    ask(make_deps(llm=llm))
    user_msg = llm.calls[0][1]["content"]
    assert "[1] 物品被誤鎖" in user_msg and "[2] 角色被鎖" in user_msg
    assert "去哪練" not in user_msg
    assert user_msg.endswith("問題：東西不能交易怎麼辦")


def test_no_match_does_not_call_model(make_deps):
    llm = FakeLLM()
    deps = make_deps(hits=[HIT_LOW], llm=llm)
    events = ask(deps)
    assert llm.calls == []
    assert events[0][0] == "meta" and events[0][1]["sources"] == []
    assert events[1] == ("token", {"t": NO_MATCH_MESSAGE})
    assert events[-1][1] == {"status": "no_match", "cached": False}
    log = log_of(deps, events)
    assert log["status"] == "no_match" and log["category"] == "練等"


def test_empty_store_is_no_match(make_deps):
    events = ask(make_deps(hits=[]))
    assert events[-1][1] == {"status": "no_match", "cached": False}


def test_not_found_answer(make_deps):
    deps = make_deps(llm=FakeLLM(["目前資料裡", "沒有這題的答案。"]))
    events = ask(deps)
    assert events[-1][1] == {"status": "not_found", "cached": False}
    assert log_of(deps, events)["status"] == "not_found"


def test_llm_error_gives_friendly_message(make_deps):
    deps = make_deps(llm=FakeLLM(fail_after=1))
    events = ask(deps)
    assert events[-1] == ("done", {"status": "error", "message": ERROR_MESSAGE})
    log = log_of(deps, events)
    assert log["status"] == "error" and log["answer"] == "這是"


def test_embed_error_still_sends_meta_and_logs(make_deps):
    async def broken_embed(texts):
        raise RuntimeError("Ollama 掛了")

    deps = make_deps(embed_fn=broken_embed)
    events = ask(deps)
    assert [e for e, _ in events] == ["meta", "done"]
    assert events[-1][1]["status"] == "error"
    assert log_of(deps, events)["status"] == "error"


def test_client_disconnect_finishes_log_as_aborted(make_deps):
    deps = make_deps()

    async def go():
        await deps.db.init()
        gen = ask_stream(deps, "東西不能交易怎麼辦")
        meta = await gen.__anext__()
        await gen.__anext__()  # 第一個 token
        await gen.aclose()  # 模擬斷線
        return json.loads(meta.split("data: ", 1)[1])["log_id"]

    log = deps.db.get_log_sync(asyncio.run(go()))
    assert log["status"] == "aborted"
    assert log["answer"] == "這是"


def test_db_failure_does_not_break_answer(make_deps):
    deps = make_deps()

    async def broken(*a, **kw):
        raise RuntimeError("disk full")

    deps.db.create_log = broken
    events = ask(deps)
    assert events[0][1]["log_id"] is None
    assert events[-1][1] == {"status": "answered", "cached": False}


@pytest.mark.parametrize("question", ["", " ", "a", " a ", "字" * 201])
def test_ask_validation_422(make_deps, question):
    with TestClient(create_app(make_deps())) as client:
        assert client.post("/api/ask", json={"question": question}).status_code == 422


def test_ask_strips_question(make_deps):
    deps = make_deps()
    events = ask(deps, "  東西不能交易  ")
    assert log_of(deps, events)["question"] == "東西不能交易"


def test_feedback_ok_duplicate_and_missing(make_deps):
    deps = make_deps()
    with TestClient(create_app(deps)) as client:
        events = parse_sse(client.post("/api/ask", json={"question": "東西不能交易"}).text)
        log_id, token = events[0][1]["log_id"], events[0][1]["token"]

        # token 不對、沒帶 token：猜到 log_id 也不能亂按
        wrong = {"log_id": log_id, "token": "x" * 16, "helpful": False, "reason": "答錯"}
        assert client.post("/api/feedback", json=wrong).status_code == 404
        assert client.post("/api/feedback", json={"log_id": log_id, "helpful": False}).status_code == 422

        body = {"log_id": log_id, "token": token, "helpful": False, "reason": "答錯"}
        assert client.post("/api/feedback", json=body).status_code == 200
        assert client.post("/api/feedback", json=body).status_code == 409
        missing = {"log_id": 9999, "token": token, "helpful": True}
        assert client.post("/api/feedback", json=missing).status_code == 404
        bad_reason = {"log_id": log_id, "token": token, "helpful": False, "reason": "亂寫"}
        assert client.post("/api/feedback", json=bad_reason).status_code == 422

    log = deps.db.get_log_sync(log_id)
    assert log["helpful"] == 0 and log["feedback_reason"] == "答錯" and log["feedback_at"]
    assert len(token) >= 16


def test_health(make_deps):
    deps = make_deps()
    with TestClient(create_app(deps)) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert {k: body[k] for k in ("status", "chroma", "ollama", "chunks")} == {"status": "ok", "chroma": True, "ollama": True, "chunks": 3}

        async def down():
            return False

        deps.check_ollama = down
        resp = client.get("/health")
        assert resp.status_code == 503 and resp.json()["ollama"] is False


def test_index(make_deps):
    with TestClient(create_app(make_deps())) as client:
        assert client.get("/").status_code == 200


# ---- 限流、同時生成上限、正式環境 ----

def with_settings(deps, **kw):
    deps.settings = replace(deps.settings, **kw)
    return deps


def test_rate_limited_returns_429_and_is_not_logged(make_deps):
    deps = with_settings(make_deps(), cooldown_seconds=60)
    with TestClient(create_app(deps)) as client:
        headers = {"CF-Connecting-IP": "1.2.3.4"}
        assert client.post("/api/ask", json={"question": "東西不能交易"}, headers=headers).status_code == 200
        resp = client.post("/api/ask", json={"question": "東西不能交易"}, headers=headers)
        assert resp.status_code == 429
        body = resp.json()
        assert body["retry_after"] == int(resp.headers["Retry-After"]) and 0 < body["retry_after"] <= 60
        assert body["message"].startswith("問太快了")
        # 不同 IP 不受影響
        other = client.post("/api/ask", json={"question": "東西不能交易"}, headers={"CF-Connecting-IP": "5.6.7.8"})
        assert other.status_code == 200

        health = client.get("/health").json()
        assert health["today"]["asked"] == 2
        assert health["today"]["blocked"]["cooldown"] == 1

    # 被擋的那次沒有寫入紀錄
    assert deps.db.get_log_sync(3) is None
    assert deps.db.get_log_sync(2) is not None


def test_daily_limit_429(make_deps):
    deps = with_settings(make_deps(), daily_limit=1)
    with TestClient(create_app(deps)) as client:
        client.post("/api/ask", json={"question": "東西不能交易"}, headers={"CF-Connecting-IP": "1.1.1.1"})
        resp = client.post("/api/ask", json={"question": "東西不能交易"}, headers={"CF-Connecting-IP": "2.2.2.2"})
        assert resp.status_code == 429
        assert resp.json()["message"] == "今天的問答次數已經用完了，明天再來。"


def test_queue_timeout_returns_503(make_deps):
    deps = with_settings(make_deps(), max_concurrent=0, queue_timeout=0.05)  # 0 個名額 = 一定排不到
    with TestClient(create_app(deps)) as client:
        resp = client.post("/api/ask", json={"question": "東西不能交易"})
        assert resp.status_code == 503
        assert resp.json() == {"message": "目前人有點多，稍後再試一次。"}


def test_slots_are_released_after_each_answer(make_deps):
    deps = with_settings(make_deps(), max_concurrent=1, queue_timeout=0.5, cooldown_seconds=0)
    with TestClient(create_app(deps)) as client:
        for i in range(3):  # 名額沒還的話第 2 題就會 503
            resp = client.post("/api/ask", json={"question": f"問題{i}號"})
            assert resp.status_code == 200
        # 模型出錯也要還
        deps.llm = FakeLLM(fail_after=0)
        assert client.post("/api/ask", json={"question": "會出錯的問題"}).status_code == 200
        deps.llm = FakeLLM()
        assert client.post("/api/ask", json={"question": "出錯之後"}).status_code == 200


def test_docs_disabled_in_production(make_deps):
    with TestClient(create_app(make_deps())) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200

    deps = with_settings(make_deps(), env="production")
    with TestClient(create_app(deps)) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
        assert client.get("/health").status_code == 200


def test_slot_released_when_client_disconnects():
    from app import SlotStreamingResponse

    async def go():
        slots = asyncio.Semaphore(1)
        await slots.acquire()

        async def body():
            yield "event: token\ndata: {}\n\n"

        async def receive():
            await asyncio.sleep(10)
            return {"type": "http.disconnect"}

        async def send(message):
            raise OSError("client gone")  # 送 header 時就斷線

        resp = SlotStreamingResponse(body(), slots=slots, media_type="text/event-stream")
        with pytest.raises(Exception):  # starlette 會轉成 ClientDisconnect
            await resp({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        return slots.locked()

    assert asyncio.run(go()) is False


# ---- 相同問題快取 ----

def ask_in(client, question, ip):
    resp = client.post("/api/ask", json={"question": question}, headers={"CF-Connecting-IP": ip})
    assert resp.status_code == 200
    return parse_sse(resp.text)


def answer_of(events):
    return "".join(d["t"] for e, d in events if e == "token")


def test_same_question_served_from_cache(make_deps):
    llm = FakeLLM(["第一次", "的答案"])
    deps = make_deps(llm=llm)
    with TestClient(create_app(deps)) as client:
        first = ask_in(client, "影武者幾等轉職？", "1.1.1.1")
        second = ask_in(client, "  影武者幾等轉職  ", "2.2.2.2")  # 正規化後一樣
    assert len(llm.calls) == 1
    assert first[-1][1] == {"status": "answered", "cached": False}
    assert second[-1][1] == {"status": "answered", "cached": True}
    assert answer_of(second) == "第一次的答案"
    assert second[0][1]["sources"] == first[0][1]["sources"]

    log2 = log_of(deps, second)
    assert log2["cached"] == 1 and log2["status"] == "answered"
    assert log2["cache_key"] == log_of(deps, first)["cache_key"]


def test_cache_misses_when_data_changes(make_deps):
    llm = FakeLLM()
    deps = make_deps(llm=llm)
    with TestClient(create_app(deps)) as client:
        ask_in(client, "影武者幾等轉職", "1.1.1.1")
        deps.store.hits[0] = replace(HIT_A, content_hash="改過的內容")
        third = ask_in(client, "影武者幾等轉職", "2.2.2.2")
    assert len(llm.calls) == 2
    assert third[-1][1]["cached"] is False


def test_cache_misses_when_prompt_or_model_changes(make_deps):
    llm = FakeLLM()
    deps = make_deps(llm=llm)
    with TestClient(create_app(deps)) as client:
        ask_in(client, "影武者幾等轉職", "1.1.1.1")
        llm.model = "另一個模型"
        ask_in(client, "影武者幾等轉職", "2.2.2.2")
        deps.prompt = replace(deps.prompt, version="answer_v9")
        ask_in(client, "影武者幾等轉職", "3.3.3.3")
    assert len(llm.calls) == 3


def test_thumbs_down_stops_cache(make_deps):
    llm = FakeLLM()
    deps = make_deps(llm=llm)
    with TestClient(create_app(deps)) as client:
        ask_in(client, "影武者幾等轉職", "1.1.1.1")
        cached = ask_in(client, "影武者幾等轉職", "2.2.2.2")
        assert cached[-1][1]["cached"] is True
        # 對快取出來的答案按 👎，之後就不再用這個答案
        meta = cached[0][1]
        client.post("/api/feedback", json={"log_id": meta["log_id"], "token": meta["token"], "helpful": False, "reason": "答錯"})
        again = ask_in(client, "影武者幾等轉職", "3.3.3.3")
    assert again[-1][1]["cached"] is False
    assert len(llm.calls) == 2


def test_not_found_and_errors_are_not_cached(make_deps):
    llm = FakeLLM(["目前資料裡沒有這題的答案。"])
    deps = make_deps(llm=llm)
    with TestClient(create_app(deps)) as client:
        ask_in(client, "影武者幾等轉職", "1.1.1.1")
        ask_in(client, "影武者幾等轉職", "2.2.2.2")
    assert len(llm.calls) == 2


def test_cache_disabled(make_deps):
    llm = FakeLLM()
    deps = with_settings(make_deps(llm=llm), cache_enabled=False)
    with TestClient(create_app(deps)) as client:
        ask_in(client, "影武者幾等轉職", "1.1.1.1")
        second = ask_in(client, "影武者幾等轉職", "2.2.2.2")
    assert len(llm.calls) == 2 and second[-1][1]["cached"] is False


def test_cache_respects_ttl(make_deps):
    llm = FakeLLM()
    deps = make_deps(llm=llm)
    with TestClient(create_app(deps)) as client:
        first = ask_in(client, "影武者幾等轉職", "1.1.1.1")
        # 把第一筆改成 4 天前
        with deps.db._connect() as conn:
            conn.execute("UPDATE qa_logs SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (first[0][1]["log_id"],))
        second = ask_in(client, "影武者幾等轉職", "2.2.2.2")
    assert second[-1][1]["cached"] is False


def test_cache_lookup_failure_falls_back_to_model(make_deps):
    llm = FakeLLM()
    deps = make_deps(llm=llm)

    async def broken(*a, **kw):
        raise RuntimeError("db locked")

    deps.db.find_cached_answer = broken
    events = ask(deps)
    assert events[-1][1] == {"status": "answered", "cached": False}
    assert len(llm.calls) == 1


# ---- 許願、收錄清單 ----

def test_suggest_saved_without_ip(make_deps):
    deps = make_deps()
    with TestClient(create_app(deps)) as client:
        body = {"text": "  夜使者技能怎麼點？ ", "source": "no_match", "log_id": 7}
        assert client.post("/api/suggest", json=body, headers={"CF-Connecting-IP": "1.2.3.4"}).json() == {"ok": True}
    with deps.db._connect() as conn:
        row = dict(conn.execute("SELECT * FROM suggestions").fetchone())
    assert row["text"] == "夜使者技能怎麼點？" and row["text_norm"] == "夜使者技能怎麼點"
    assert row["source"] == "no_match" and row["log_id"] == 7
    assert "1.2.3.4" not in str(row)


@pytest.mark.parametrize("body", [{"text": "a"}, {"text": "字" * 201}, {"text": "正常的", "source": "亂寫"}])
def test_suggest_validation(make_deps, body):
    with TestClient(create_app(make_deps())) as client:
        assert client.post("/api/suggest", json=body).status_code == 422


def test_suggest_has_its_own_rate_limit(make_deps):
    deps = with_settings(make_deps(), suggest_cooldown_seconds=0, suggest_hourly_limit=2, cooldown_seconds=60)
    with TestClient(create_app(deps)) as client:
        h = {"CF-Connecting-IP": "1.1.1.1"}
        assert client.post("/api/suggest", json={"text": "第一個"}, headers=h).status_code == 200
        assert client.post("/api/suggest", json={"text": "第二個"}, headers=h).status_code == 200
        resp = client.post("/api/suggest", json={"text": "第三個"}, headers=h)
        assert resp.status_code == 429
        assert resp.json()["message"].startswith("這個小時已經送了不少了")
        # 許願用完不影響問答
        assert client.post("/api/ask", json={"question": "東西不能交易"}, headers=h).status_code == 200


def test_topics_endpoint(make_deps, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    md = "# 誤鎖\n這是誤鎖的說明。\n\n## 角色被鎖\n內容\n## 物品被誤鎖\n內容\n"
    (data / "誤鎖.md").write_text(md, encoding="utf-8")
    deps = with_settings(make_deps(), data_dir=str(data))
    with TestClient(create_app(deps)) as client:
        body = client.get("/api/topics").json()
    assert body["total"] == 3
    assert body["categories"][0] == {"name": "誤鎖", "intro": "這是誤鎖的說明。", "titles": ["角色被鎖", "物品被誤鎖"]}
    assert body["categories"][1] == {"name": "練等", "intro": "", "titles": ["去哪練"]}


# ---- IPv6 限流、安全 header ----

def test_ipv6_same_64_shares_cooldown(make_deps):
    deps = with_settings(make_deps(), cooldown_seconds=60)
    with TestClient(create_app(deps)) as client:
        ask_one = lambda ip: client.post("/api/ask", json={"question": "東西不能交易"}, headers={"CF-Connecting-IP": ip})
        assert ask_one("2001:db8:1:2::1").status_code == 200
        assert ask_one("2001:db8:1:2:dead:beef::9").status_code == 429  # 同一段 /64 換位址
        assert ask_one("2001:db8:1:3::1").status_code == 200  # 別段


def test_security_headers(make_deps):
    import base64
    import hashlib
    import re

    from app import INDEX_HTML

    with TestClient(create_app(make_deps())) as client:
        index = client.get("/")
        api = client.get("/health")

    for resp in (index, api):
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"
        assert "strict-transport-security" in resp.headers
    assert "content-security-policy" not in api.headers  # CSP 只給首頁

    csp = index.headers["content-security-policy"]
    assert "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    script = re.search(r"<script>(.*?)</script>", INDEX_HTML.read_text(encoding="utf-8"), re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode()
    assert f"'sha256-{digest}'" in csp


def test_security_headers_do_not_break_streaming(make_deps):
    with TestClient(create_app(make_deps())) as client:
        resp = client.post("/api/ask", json={"question": "東西不能交易"})
    assert resp.headers["x-frame-options"] == "DENY"
    assert parse_sse(resp.text)[-1][1]["status"] == "answered"
