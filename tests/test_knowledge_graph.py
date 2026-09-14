from pathlib import Path

from kgr_memory.extractor import Triple
from kgr_memory.knowledge_graph import KnowledgeGraph
from kgr_memory.vector_store import VectorStore


class FakeEmbedder:
    model = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            vec = [
                1.0 if "brisbane" in lower or "live" in lower else 0.0,
                1.0 if "queensland" in lower or "capital" in lower else 0.0,
                1.0 if "sydney" in lower else 0.0,
            ]
            if sum(vec) == 0:
                vec = [0.1, 0.1, 0.1]
            out.append(vec)
        return out


class FakeExtractor:
    def extract(self, text: str, speaker: str = "user") -> list[Triple]:
        known = {
            "I live in Brisbane": [Triple(speaker, "lives_in", "Brisbane")],
            "I don't live in Sydney": [Triple(speaker, "not_lives_in", "Sydney")],
            "Brisbane is the capital of Queensland": [
                Triple("Brisbane", "capital_of", "Queensland")
            ],
            "I love him so much": [],
        }
        return known.get(text, [])


def _stores(tmp_path: Path) -> tuple[VectorStore, KnowledgeGraph]:
    db = tmp_path / "memory.sqlite3"
    store = VectorStore(db, embedder=FakeEmbedder())
    graph = KnowledgeGraph(db, extractor=FakeExtractor())
    return store, graph


def test_ingest_writes_nodes_and_negated_edge(tmp_path: Path) -> None:
    store, graph = _stores(tmp_path)
    uid = store.add("I don't live in Sydney", role="user")
    triples = graph.ingest(uid)
    assert triples == [
        graph.triples_for([uid])[0],
    ]
    assert triples[0].predicate == "not_lives_in"
    assert triples[0].object == "Sydney"
    store.close()
    graph.close()


def test_ingest_is_idempotent_and_merges_case(tmp_path: Path) -> None:
    store, graph = _stores(tmp_path)
    uid = store.add("I live in Brisbane", role="user")
    graph.ingest(uid)
    graph.ingest(uid)
    assert graph.triples_for([uid]) == [
        graph.query_from_utterances([uid])[0],
    ]
    count = graph._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert count == 2  # user, Brisbane
    edge_count = graph._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert edge_count == 1
    store.close()
    graph.close()


def test_pronoun_utterance_stores_no_edges(tmp_path: Path) -> None:
    store, graph = _stores(tmp_path)
    uid = store.add("I love him so much", role="user")
    assert graph.ingest(uid) == []
    assert graph.triples_for([uid]) == []
    store.close()
    graph.close()


def test_query_from_utterances_seeds_then_one_hop(tmp_path: Path) -> None:
    store, graph = _stores(tmp_path)
    live_id = store.add("I live in Brisbane", role="user")
    capital_id = store.add("Brisbane is the capital of Queensland", role="user")
    graph.ingest(live_id)
    graph.ingest(capital_id)

    hits = store.query("Where do I live?", k=1)
    assert hits[0].id == live_id

    triples = graph.query_from_utterances([hit.id for hit in hits])
    facts = {(t.subject, t.predicate, t.object) for t in triples}
    assert ("user", "lives_in", "Brisbane") in facts
    assert ("Brisbane", "capital_of", "Queensland") in facts
    store.close()
    graph.close()


def test_query_from_utterances_empty_seed(tmp_path: Path) -> None:
    store, graph = _stores(tmp_path)
    assert graph.query_from_utterances([]) == []
    store.close()
    graph.close()
