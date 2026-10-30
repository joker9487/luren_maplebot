import asyncio
import hashlib

import pytest

from chunking import Entry, split_entry
from ingest import plan_sync, run_ingest
from retrieval import ChromaStore


def chunk(title, content="內容", category="c"):
    return split_entry(Entry(category, title, content))[0]


def fake_vec(text: str) -> list[float]:
    d = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255 + 0.01 for b in d[:8]]


class FakeEmbed:
    def __init__(self):
        self.calls: list[list[str]] = []

    async def __call__(self, texts):
        self.calls.append(texts)
        return [fake_vec(t) for t in texts]

    @property
    def count(self):
        return sum(len(c) for c in self.calls)


# ---- plan_sync（純函式） ----

def test_plan_sync_add_update_skip_delete():
    a, b, c = chunk("a"), chunk("b"), chunk("c")
    existing = {a.id: a.content_hash, b.id: "舊的hash", "c/gone#0": "x"}
    plan = plan_sync([a, b, c], existing)
    assert [x.id for x in plan.add] == [c.id]
    assert [x.id for x in plan.update] == [b.id]
    assert plan.skip == [a.id]
    assert plan.delete == ["c/gone#0"]
    assert plan.summary() == "新增 1、更新 1、刪除 1、跳過 1"


# ---- run_ingest + 暫存資料夾的 Chroma ----

@pytest.fixture
def store(tmp_path):
    return ChromaStore(str(tmp_path / "chroma"), "test")


def ingest(chunks, store, embed, model="m1", **kw):
    return asyncio.run(run_ingest(chunks, store, embed, model, **kw))


def test_first_ingest_adds_all_and_records_model(store):
    embed = FakeEmbed()
    plan = ingest([chunk("a"), chunk("b")], store, embed)
    assert len(plan.add) == 2
    assert store.count() == 2
    assert store.embed_model == "m1"


def test_second_ingest_skips_unchanged_and_updates_changed(store):
    ingest([chunk("a"), chunk("b")], store, FakeEmbed())
    embed = FakeEmbed()
    plan = ingest([chunk("a"), chunk("b", "改過的內容")], store, embed)
    assert plan.skip == ["c/a#0"]
    assert [x.id for x in plan.update] == ["c/b#0"]
    assert embed.count == 1  # 只重算有改的
    assert store.existing_hashes()["c/b#0"] == chunk("b", "改過的內容").content_hash


def test_removed_entry_is_deleted(store):
    ingest([chunk("a"), chunk("b")], store, FakeEmbed())
    plan = ingest([chunk("a")], store, FakeEmbed())
    assert plan.delete == ["c/b#0"]
    assert set(store.existing_hashes()) == {"c/a#0"}


def test_model_change_rebuilds_everything(store):
    ingest([chunk("a"), chunk("b")], store, FakeEmbed(), model="m1")
    embed = FakeEmbed()
    plan = ingest([chunk("a")], store, embed, model="m2")
    assert plan.rebuild
    assert embed.count == 1  # 沒改內容也重算
    assert plan.delete == ["c/b#0"]
    assert store.embed_model == "m2"
    assert set(store.existing_hashes()) == {"c/a#0"}


def test_rebuild_flag_recomputes_all(store):
    ingest([chunk("a")], store, FakeEmbed())
    embed = FakeEmbed()
    plan = ingest([chunk("a")], store, embed, rebuild=True)
    assert plan.rebuild and len(plan.add) == 1 and embed.count == 1


def test_dry_run_changes_nothing(store):
    ingest([chunk("a")], store, FakeEmbed())
    embed = FakeEmbed()
    plan = ingest([chunk("a", "新內容"), chunk("b")], store, embed, dry_run=True)
    assert len(plan.add) == 1 and len(plan.update) == 1
    assert embed.calls == []
    assert store.existing_hashes()["c/a#0"] == chunk("a").content_hash


def test_dry_run_on_empty_store(store):
    plan = ingest([chunk("a")], store, FakeEmbed(), dry_run=True)
    assert len(plan.add) == 1
    assert store.count() == 0


def test_search_scores_and_metadata(store):
    a, b = chunk("a", category="誤鎖"), chunk("b")
    ingest([a, b], store, FakeEmbed())
    hits = store.search(fake_vec(a.embed_text), k=2)
    assert hits[0].chunk_id == a.id
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].category == "誤鎖"
    assert hits[0].display_text == a.display_text
    assert hits[0].score >= hits[1].score


def test_search_empty_store(store):
    assert store.search([0.1] * 8, k=4) == []


# ---- app 開著時，另一個行程跑 ingest ----

INGEST_SCRIPT = """
import asyncio, hashlib, sys
sys.path.insert(0, {root!r})
from chunking import Entry, split_entry
from ingest import run_ingest
from retrieval import ChromaStore
def vec(t):
    d = hashlib.sha256(t.encode("utf-8")).digest()
    return [b / 255 + 0.01 for b in d[:8]]
async def emb(ts):
    return [vec(t) for t in ts]
titles, model, rebuild = sys.argv[1].split(","), sys.argv[2], sys.argv[3] == "1"
chunks = [split_entry(Entry("c", t, "內容"))[0] for t in titles]
asyncio.run(run_ingest(chunks, ChromaStore({path!r}, "test"), emb, model, rebuild=rebuild))
"""


def ingest_in_other_process(path, titles, model="m1", rebuild=False):
    import subprocess
    import sys
    from pathlib import Path

    script = INGEST_SCRIPT.format(root=str(Path(__file__).resolve().parent.parent), path=str(path))
    subprocess.run(
        [sys.executable, "-c", script, ",".join(titles), model, "1" if rebuild else "0"], check=True, timeout=120
    )


def test_running_app_sees_ingest_from_other_process(tmp_path):
    path = tmp_path / "chroma"
    app_store = ChromaStore(str(path), "test")  # app 先啟動，庫還不存在
    assert app_store.search([0.1] * 8, 1) == []

    ingest_in_other_process(path, ["a", "b"])
    assert app_store.count() == 2
    assert app_store.search(fake_vec(chunk("a").embed_text), 1)[0].title == "a"

    ingest_in_other_process(path, ["a", "b", "new"])  # 新增
    assert app_store.search(fake_vec(chunk("new").embed_text), 1)[0].title == "new"

    ingest_in_other_process(path, ["a", "x"], model="m2")  # 換模型 → 整批重建
    assert app_store.embed_model == "m2"
    assert {t for _, t in app_store.list_entries()} == {"a", "x"}
    assert app_store.search(fake_vec(chunk("x").embed_text), 1)[0].title == "x"
