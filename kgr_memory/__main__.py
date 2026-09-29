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

from kgr_memory.knowledge_graph import EdgeRow, KnowledgeGraph, NodeRow
from kgr_memory.vector_store import VectorStore

DEFAULT_DB = Path("data/memory.sqlite3")


def print_nodes_and_edges(nodes: list[NodeRow], edges: list[EdgeRow]) -> None:
    print("=== nodes ===")
    if not nodes:
        print("(none)")
    for node in nodes:
        print(f"[{node.id}] {node.name}")
    print("=== edges ===")
    if not edges:
        print("(none)")
    for edge in edges:
        print(
            f"[{edge.id}] {edge.subject_id}:{edge.subject}"
            f" -- {edge.predicate} -->"
            f" {edge.object_id}:{edge.object}"
            f"  (utterance {edge.utterance_id})"
        )

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

    store_p = sub.add_parser(
        "store",
        help="write one utterance into the Graphiti knowledge graph",
    )
    store_p.add_argument("text")
    store_p.add_argument("--group", default="kgr", help="conversation partition")

    search_p = sub.add_parser(
        "search",
        help="one Graphiti search; prints fact edges",
    )
    search_p.add_argument("text")
    search_p.add_argument("--group", default="kgr")

    ask_p = sub.add_parser(
        "ask",
        help="answer from the graph; oneshot or a ReAct loop",
    )
    ask_p.add_argument("text")
    ask_p.add_argument("--group", default="kgr")
    ask_p.add_argument(
        "--mode",
        choices=("oneshot", "react"),
        default="react",
    )
    ask_p.add_argument(
        "--prompt",
        choices=("strict", "balanced", "loose"),
        default="balanced",
        help="which definition of enough the ReAct model uses",
    )
    ask_p.add_argument("--max-rounds", type=int, default=10)
    ask_p.add_argument(
        "--question-type",
        default=None,
        help="LongMemEval type, stored on the log row",
    )
    ask_p.add_argument(
        "--log",
        type=Path,
        default=Path("data/runs/asks.jsonl"),
        help="JSONL trace (gitignored under data/)",
    )

    args = parser.parse_args()
    if args.cmd in {"store", "search", "ask"}:
        _graphiti_command(args)
        return
    if args.cmd == "add":
        store = VectorStore(args.db)
        graph = KnowledgeGraph(args.db)
        try:
            utterance_id = store.add(args.text, role=args.role)
            print(f"stored utterance {utterance_id}")
            graph.ingest(utterance_id)
            nodes, edges = graph.nodes_and_edges_for(utterance_id)
            print_nodes_and_edges(nodes, edges)
        finally:
            store.close()
            graph.close()
        return

    if args.cmd == "ingest":
        graph = KnowledgeGraph(args.db)
        try:
            graph.ingest(args.utterance_id)
            nodes, edges = graph.nodes_and_edges_for(args.utterance_id)
            print(f"ingested utterance {args.utterance_id}")
            print_nodes_and_edges(nodes, edges)
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


def _graphiti_command(args) -> None:
    import asyncio

    from kgr_memory.graphiti_memory import add_text, build_graphiti, ensure_indices, search_facts
    from kgr_memory.live import answer_live, append_log, result_record
    from kgr_memory.react import AskResult, RoundLog

    async def run():
        graphiti = build_graphiti()
        try:
            await ensure_indices(graphiti)

            if args.cmd == "store":
                await add_text(graphiti, args.text, args.group)
                print(f"stored in group {args.group}")
                return
            if args.cmd == "search":
                facts, seconds = await search_facts(graphiti, args.text, args.group)
                print(f"{seconds:.3f}s  {len(facts)} facts")
                for fact in facts:
                    print(f"[{fact.edge_id}] {fact.text}")
                return

            if args.mode == "oneshot":
                facts, seconds = await search_facts(graphiti, args.text, args.group)
                result = AskResult(
                    question=args.text,
                    prompt_name=args.prompt,
                    mode="oneshot",
                    answer=answer_live(args.text, facts),
                    rounds=[
                        RoundLog(
                            round=1,
                            query=args.text,
                            facts=facts,
                            search_seconds=seconds,
                        )
                    ],
                    stopped_because="oneshot",
                )
            else:
                from kgr_memory.live import decide_live
                from kgr_memory.react import ask_react_async

                async def search(query: str):
                    return await search_facts(graphiti, query, args.group)

                result = await ask_react_async(
                    args.text,
                    search,
                    decide_live,
                    answer_live,
                    prompt_name=args.prompt,
                    max_rounds=args.max_rounds,
                )
            print(result.answer)
            print(
                f"mode={result.mode} prompt={result.prompt_name} "
                f"rounds={len(result.rounds)} stop={result.stopped_because}"
            )
            append_log(
                args.log,
                result_record(
                    result,
                    group_id=args.group,
                    question_type=args.question_type,
                ),
            )
            print(f"log {args.log}")
        finally:
            await graphiti.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
