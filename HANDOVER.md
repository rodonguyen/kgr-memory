# Handover

**Date:** 14.09.2026  
**Repo:** [kgr-memory](https://github.com/rodonguyen/kgr-memory)  
**Course:** Griffith University — knowledge graphs, enriching KGs with AI (Python). Assessment: project 60% / presentation 10% / report 30%.

Read this, then [AGENTS.md](AGENTS.md) (how to code), [log.md](log.md) (decisions), [README.md](README.md) (full spec).

## What this is

Hybrid **LLM agent memory**. Goal: answers that are correct, complete, and **grounded** in stored memory — especially high-stakes cases. Expected cost: extra latency.

Three modules, in order. Start tiny, then extend. Do not jump to production scale.

| # | Module | Status |
| --- | --- | --- |
| 1 | Vector store | **Done (v1)** |
| 2 | Knowledge graph | **Done (v1)** |
| 3 | Multi-turn memory query | **Next** |

## What works today

- **`add`**: save an utterance to SQLite, embed with OpenAI `text-embedding-3-small`, extract triples with `gpt-4o-mini`.
- **`query`**: two **separate** lists (do not merge scores):
  1. Vector — brute-force cosine over **all** stored embeddings, top-k utterances.
  2. Graph — take those utterance ids → nodes on their edges → **1-hop** neighbourhood.
- **`ingest <id>`**: optional redo. Does **not** add a memory; re-extracts triples for an existing row.

Store: one SQLite file `data/memory.sqlite3` (gitignored). Not CSV, not RAM. PostgreSQL / ANN / BM25 / Wikidata are later.

### Run

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows bash; .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env             # set OPENAI_API_KEY

python -m kgr_memory -h
python -m kgr_memory add "I live in Brisbane" --role user
python -m kgr_memory query "Where do I live?"
python -m pytest -q
```

### Layout

| Path | Role |
| --- | --- |
| `kgr_memory/vector_store.py` | utterances + embeddings + cosine |
| `kgr_memory/embeddings.py` | OpenAI embedder |
| `kgr_memory/extractor.py` | utterance → triples |
| `kgr_memory/knowledge_graph.py` | nodes/edges, ingest, 1-hop from utterance ids |
| `kgr_memory/__main__.py` | CLI |
| `tests/` | no live API; `FakeEmbedder` / `FakeExtractor` |

Schema (same DB): `utterances(id, text, role, embedding, created_at)`, `nodes`, `edges` (edge has `utterance_id`).

## Decisions you should not silently reverse

See [log.md](log.md). Short version:

- Ignore unresolved pronouns and intensifiers; empty extract is OK.
- Negation is a `not_*` predicate (`not_lives_in`).
- Time/status and conflicts are **not** handled (stale and new edges both stay).
- Entity merge is `lower()` only.
- Graph entry is **vector-seeded**, not a free-text graph search.
- No `Co-authored-by` on commits ([AGENTS.md](AGENTS.md)).

## Known behaviour (not bugs to “fix” unless you mean to)

- **`query` with empty graph** means those hits have no edges (utterances added before `add` also ingested). Re-run `ingest <id>` or add new rows.
- **1-hop is noisy.** Top-k vector hits (default k=5) all seed the graph, so weak hits (China, Elon, …) still contribute triples. Lists are unmerged on purpose.
- Extractor quality is messy (duplicate people, awkward predicates). Fine for v1.

## Next step: multi-turn memory query

Bare minimum only. Do not build a full agent framework.

**Behaviour:** the model does not stop after one `query`. It looks at the two lists, may issue another memory query (narrower question, different k, follow a node), repeats until it thinks it has enough, then answers the user.

**Stop rule (already decided):**

- LLM scores “enough evidence” (e.g. 1–10) and decides to stop.
- Always a **hard max-round cap** (small, e.g. 3) so it cannot loop forever.

**Keep for v1 of this module:**

- Still print / return **vector list and graph list separately** each round (or log them). Do not fuse into one rank yet.
- Reuse `VectorStore.query` and `KnowledgeGraph.query_from_utterances`.
- Cap must be visible in the CLI (e.g. `--max-rounds`).

**Suggested first slice:**

1. CLI `ask "…"` (or `query --turns N`) that loops: retrieve → show both lists → LLM “score + next query or STOP” → repeat.
2. After STOP or cap, one grounded answer that cites utterance ids / triples.
3. Tests with a fake LLM (no API): force 2 hops then stop; force cap.

**Do not do yet:** ANN index, BM25, Neo4j, pgvector, Wikidata, merging ranks, coref, conflict resolution.

**After that**: download longmemeval-s to test a conversation and see how it works, test early - fix earlier

## If you get stuck

- Principle: smallest end-to-end loop, then extend.
- Append new decisions under a new date in [log.md](log.md).
- Course hook to keep: **enriching and traversing a KG with AI**, not a hosted vector DB demo.
