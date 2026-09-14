"""Hybrid memory CLI: vector store + knowledge graph.

Typical use:
  python -m kgr_memory add "I live in Brisbane" --role user
  python -m kgr_memory query "Where do I live?"

ingest is optional. It does not add a memory; it re-extracts triples
for an utterance that already exists:
  python -m kgr_memory ingest 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kgr_memory.knowledge_graph import KnowledgeGraph
from kgr_memory.vector_store import VectorStore

DEFAULT_DB = Path("data/memory.sqlite3")

_EPILOG = """
Typical use:
  python -m kgr_memory add "I live in Brisbane" --role user
  python -m kgr_memory query "Where do I live?"

ingest does not add a new memory. It re-runs extraction on an existing
utterance id (the number printed by add):
  python -m kgr_memory ingest 1
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid memory: vector store + knowledge graph.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite file (default: {DEFAULT_DB})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="command")

    add_p = sub.add_parser(
        "add",
        help="save a new utterance: embed it and extract graph triples",
        description=(
            "Normal write path. Inserts the text, embeds it for vector search, "
            "and extracts fact triples into the graph. This is how you remember something."
        ),
    )
    add_p.add_argument("text", help='utterance to store, e.g. "I live in Brisbane"')
    add_p.add_argument(
        "--role",
        default="user",
        help="speaker stored on the row and used as 'I' in triples (default: user)",
    )

    ingest_p = sub.add_parser(
        "ingest",
        help="re-extract triples for an existing utterance (does not add a memory)",
        description=(
            "Does not add a new utterance.\n"
            "Reads an id that already exists (printed by add), runs the extractor "
            "again, and inserts any new triples. Duplicates are ignored.\n\n"
            "Use this if the extract prompt changed, or the first extract was empty/wrong.\n\n"
            "Example:\n"
            "  python -m kgr_memory ingest 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ingest_p.add_argument(
        "utterance_id",
        type=int,
        help="existing utterances.id from add, e.g. 1",
    )

    query_p = sub.add_parser(
        "query",
        help="search: print a vector list and a graph list (not merged)",
        description=(
            "Fan-out search. Prints two separate lists:\n"
            "  vector — nearest utterances by cosine similarity\n"
            "  graph  — 1-hop triples seeded from those utterance ids\n\n"
            "Example:\n"
            '  python -m kgr_memory query "Where do I live?"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    query_p.add_argument("text", help="question or search text")
    query_p.add_argument(
        "-k",
        type=int,
        default=5,
        help="how many vector hits to return (default: 5)",
    )

    args = parser.parse_args()
    if args.cmd == "add":
        store = VectorStore(args.db)
        graph = KnowledgeGraph(args.db)
        try:
            utterance_id = store.add(args.text, role=args.role)
            print(f"stored utterance {utterance_id}")
            triples = graph.ingest(utterance_id)
            if not triples:
                print("graph: (no triples)")
            for triple in triples:
                print(f"  {triple.subject} -- {triple.predicate} --> {triple.object}")
        finally:
            store.close()
            graph.close()
        return

    if args.cmd == "ingest":
        graph = KnowledgeGraph(args.db)
        try:
            triples = graph.ingest(args.utterance_id)
            print(f"ingested utterance {args.utterance_id} ({len(triples)} triple(s))")
            for triple in triples:
                print(f"  {triple.subject} -- {triple.predicate} --> {triple.object}")
        finally:
            graph.close()
        return

    store = VectorStore(args.db)
    graph = KnowledgeGraph(args.db)
    try:
        hits = store.query(args.text, k=args.k)
        print("=== vector ===")
        if not hits:
            print("(none)")
        for hit in hits:
            role = hit.role or "-"
            print(f"{hit.score:.3f}  [{hit.id} {role}] {hit.text}")

        triples = graph.query_from_utterances(hit.id for hit in hits)
        print("=== graph ===")
        if not triples:
            print("(none)")
        for triple in triples:
            print(
                f"{triple.subject} -- {triple.predicate} --> {triple.object}"
                f"    (utterance {triple.utterance_id})"
            )
    finally:
        store.close()
        graph.close()


if __name__ == "__main__":
    main()
