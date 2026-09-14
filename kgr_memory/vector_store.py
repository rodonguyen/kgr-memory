"""SQLite vector store: utterances in, nearest utterances out.

Query path is brute-force cosine similarity over every stored vector.
That is intentional for v1 (exact, easy to inspect). Add an ANN index later
if N gets large.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kgr_memory.embeddings import Embedder, OpenAIEmbedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    role TEXT,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _ensure_utterance_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(utterances)")}
    for leftover in ("updated_at", "extracted_at"):
        if leftover in cols:
            conn.execute(f"ALTER TABLE utterances DROP COLUMN {leftover}")


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


@dataclass(frozen=True)
class Hit:
    id: int
    text: str
    role: str | None
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(
        self,
        db_path: str | Path,
        embedder: Embedder | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or OpenAIEmbedder()
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(SCHEMA)
        _ensure_utterance_columns(self._conn)
        self._conn.commit()

    def add(self, text: str, role: str | None = None) -> int:
        vector = np.asarray(self.embedder.embed([text])[0], dtype=np.float32)
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO utterances (text, role, embedding, created_at) VALUES (?, ?, ?, ?)",
            (text, role, _to_blob(vector), now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def query(self, text: str, k: int = 5) -> list[Hit]:
        query_vec = np.asarray(self.embedder.embed([text])[0], dtype=np.float32)
        rows = self._conn.execute(
            "SELECT id, text, role, embedding FROM utterances"
        ).fetchall()
        if not rows:
            return []

        ids, texts, roles, blobs = zip(*rows)
        matrix = np.stack([_from_blob(blob) for blob in blobs])
        query_norm = float(np.linalg.norm(query_vec)) or 1.0
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms[row_norms == 0] = 1.0
        scores = (matrix @ query_vec) / (row_norms * query_norm)

        k = min(k, len(scores))
        top = np.argsort(scores)[::-1][:k]
        return [
            Hit(
                id=int(ids[i]),
                text=str(texts[i]),
                role=roles[i],
                score=float(scores[i]),
            )
            for i in top
        ]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
