from pathlib import Path

from kgr_memory.vector_store import VectorStore


class FakeEmbedder:
    """Tiny keyword vectors so tests do not call OpenAI."""

    model = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            vec = [
                1.0 if "brisbane" in lower or "live" in lower else 0.0,
                1.0 if "hospital" in lower or "doctor" in lower else 0.0,
                1.0 if "cat" in lower else 0.0,
            ]
            if sum(vec) == 0:
                vec = [0.1, 0.1, 0.1]
            out.append(vec)
        return out


def test_add_and_query_ranks_related_utterance(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "memory.sqlite3", embedder=FakeEmbedder())
    store.add("I live in Brisbane", role="user")
    store.add("The cat sat on the mat", role="user")
    store.add("She works at a hospital", role="user")

    hits = store.query("Where do I live?", k=3)

    assert store.count() == 3
    assert hits[0].text == "I live in Brisbane"
    assert hits[0].score > hits[1].score
    store.close()


def test_query_empty_store(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "memory.sqlite3", embedder=FakeEmbedder())
    assert store.query("anything") == []
    store.close()
