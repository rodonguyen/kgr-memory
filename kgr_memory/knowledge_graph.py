"""SQLite knowledge graph: triples linked back to utterances.

Write: extract triples from an utterance, upsert nodes, insert edges.
Query: start from utterance ids (usually vector hits), take those nodes,
then return the 1-hop neighbourhood. No cosine. No merge with vector scores.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from kgr_memory.extractor import OpenAIExtractor, Triple
from kgr_memory.vector_store import _ensure_utterance_columns

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_norm TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES nodes(id),
    predicate TEXT NOT NULL,
    predicate_norm TEXT NOT NULL,
    object_id INTEGER NOT NULL REFERENCES nodes(id),
    utterance_id INTEGER NOT NULL REFERENCES utterances(id),
    UNIQUE (subject_id, predicate_norm, object_id, utterance_id)
);
"""


class Extractor(Protocol):
    def extract(self, text: str, speaker: str = "user") -> list[Triple]: ...


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


@dataclass(frozen=True)
class GraphTriple:
    subject: str
    predicate: str
    object: str
    utterance_id: int


class KnowledgeGraph:
    def __init__(
        self,
        db_path: str | Path,
        extractor: Extractor | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.extractor = extractor or OpenAIExtractor()
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        has_utterances = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='utterances'"
        ).fetchone()
        if has_utterances:
            _ensure_utterance_columns(self._conn)
        self._conn.commit()

    def get_or_create_node(self, name: str) -> int:
        label = name.strip()
        norm = _norm(label)
        row = self._conn.execute(
            "SELECT id FROM nodes WHERE name_norm = ?", (norm,)
        ).fetchone()
        if row:
            return int(row[0])
        cur = self._conn.execute(
            "INSERT INTO nodes (name, name_norm) VALUES (?, ?)",
            (label, norm),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def ingest(self, utterance_id: int) -> list[GraphTriple]:
        row = self._conn.execute(
            "SELECT text, role FROM utterances WHERE id = ?",
            (utterance_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"utterance {utterance_id} does not exist")
        text, role = row
        speaker = role or "user"
        triples = self.extractor.extract(text, speaker=speaker)
        stored: list[GraphTriple] = []
        for triple in triples:
            subject_id = self.get_or_create_node(triple.subject)
            object_id = self.get_or_create_node(triple.object)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO edges
                    (subject_id, predicate, predicate_norm, object_id, utterance_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    subject_id,
                    triple.predicate.strip(),
                    _norm(triple.predicate),
                    object_id,
                    utterance_id,
                ),
            )
            stored.append(
                GraphTriple(
                    subject=triple.subject.strip(),
                    predicate=triple.predicate.strip(),
                    object=triple.object.strip(),
                    utterance_id=utterance_id,
                )
            )
        self._conn.commit()
        return stored

    def triples_for(self, utterance_ids: Iterable[int]) -> list[GraphTriple]:
        ids = list(utterance_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"""
            SELECT s.name, e.predicate, o.name, e.utterance_id
            FROM edges e
            JOIN nodes s ON s.id = e.subject_id
            JOIN nodes o ON o.id = e.object_id
            WHERE e.utterance_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        return [
            GraphTriple(subject=s, predicate=p, object=o, utterance_id=int(uid))
            for s, p, o, uid in rows
        ]

    def neighbors(self, node_ids: Iterable[int]) -> list[GraphTriple]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"""
            SELECT s.name, e.predicate, o.name, e.utterance_id
            FROM edges e
            JOIN nodes s ON s.id = e.subject_id
            JOIN nodes o ON o.id = e.object_id
            WHERE e.subject_id IN ({placeholders})
               OR e.object_id IN ({placeholders})
            """,
            ids + ids,
        ).fetchall()
        return [
            GraphTriple(subject=s, predicate=p, object=o, utterance_id=int(uid))
            for s, p, o, uid in rows
        ]

    def query_from_utterances(self, utterance_ids: Iterable[int]) -> list[GraphTriple]:
        """Seed nodes from those utterances, then 1-hop. Empty seed → empty list."""
        ids = list(utterance_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        node_rows = self._conn.execute(
            f"""
            SELECT subject_id FROM edges WHERE utterance_id IN ({placeholders})
            UNION
            SELECT object_id FROM edges WHERE utterance_id IN ({placeholders})
            """,
            ids + ids,
        ).fetchall()
        node_ids = [int(row[0]) for row in node_rows]
        return self.neighbors(node_ids)

    def close(self) -> None:
        self._conn.close()
