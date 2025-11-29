from pathlib import Path

from models.mirrormind.memory import (
    EpisodicMemoryStore,
    Episode,
    SemanticMemoryStore,
    SemanticSummary,
    build_semantic_summaries,
)


def test_dense_sparse_indexes():
    store = EpisodicMemoryStore()
    ep1 = Episode(id="1", entity_id="r1", time=None, type="doc", text="alpha beta", graph_context=[], dense=[1.0], sparse={})
    ep2 = Episode(id="2", entity_id="r1", time=None, type="doc", text="beta gamma", graph_context=[], dense=[2.0], sparse={})
    store.bulk_add([ep1, ep2])
    store.build_indexes(use_faiss=False)
    hits = store.query(entity_id="r1", text="beta", top_k=1)
    assert hits
    assert hits[0].id in {"1", "2"}


def test_persistence(tmp_path: Path):
    store = EpisodicMemoryStore()
    ep = Episode(id="1", entity_id="r1", time=None, type="doc", text="alpha", graph_context=[], dense=[1.0], sparse={})
    store.add(ep)
    out_path = tmp_path / "epis.jsonl"
    store.save(out_path)
    loaded = EpisodicMemoryStore.load(out_path)
    assert loaded.query(entity_id="r1", text="alpha", top_k=1)[0].text == "alpha"


def test_semantic_memory_embeddings_and_metrics():
    store = SemanticMemoryStore()
    s1 = SemanticSummary(id="s1", entity_id="e1", time_window="", scope="foo", summary_text="foo bar baz", key_concepts=["foo"], dense=[])
    s2 = SemanticSummary(id="s2", entity_id="e1", time_window="", scope="foo", summary_text="unrelated text", key_concepts=[], dense=[])
    store.bulk_add([s1, s2])
    results = store.query(entity_id="e1", text="foo baz", top_k=1)
    assert results and results[0].id == "s1"
    assert store.last_query_metrics.get("candidates") == 2
    assert store.last_query_metrics.get("returned") == 1
    assert store.last_query_metrics.get("used_embedding") in (True, False)


def test_build_semantic_summaries_basic_span():
    eps = [
        Episode(
            id="e1",
            entity_id="r1",
            time="100",
            type="doc",
            text="alpha",
            graph_context=[],
            dense=[],
            sparse={},
        ),
        Episode(
            id="e2",
            entity_id="r1",
            time="200",
            type="doc",
            text="beta",
            graph_context=[],
            dense=[],
            sparse={},
        ),
    ]
    summaries = build_semantic_summaries("r1", eps, scope_label="test")
    assert len(summaries) == 1
    s = summaries[0]
    assert s.entity_id == "r1"
    assert s.scope == "test"
    assert s.time_window in ("100-200", "200-100", "100", "200")
    assert s.summary_text
    assert s.dense



def test_entities_helpers():
    store = EpisodicMemoryStore()
    store.add(Episode(id="1", entity_id="a", time=None, type="doc", text="hello", graph_context=[], dense=[], sparse={}))
    store.add(Episode(id="2", entity_id="b", time=None, type="doc", text="world", graph_context=[], dense=[], sparse={}))
    assert set(store.entities()) == {"a", "b"}
    assert {ep.id for ep in store.episodes_for("a")} == {"1"}
