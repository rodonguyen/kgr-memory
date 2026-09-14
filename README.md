# kgr-memory

Knowledge graph for an LLM agent's memory, with reasoning.

Griffith University course project. Python. Assessment: project 60%, final presentation 10%, final report 30%.

## Course context

Knowledge graphs are used in expert systems, web search, question answering, and recommendation. This course covers how machine-processible knowledge is organised, synthesised, and managed in large graphs such as DBpedia and Wikidata, and how those graphs can be enriched with AI techniques.

This project applies that idea to **agent memory**: store, relate, and retrieve facts so the model answers from evidence instead of guessing.

## Goal

Build a hybrid memory system so an LLM agent can return **correct, complete, and grounded** answers — especially in high-stakes settings where missing or invented information is costly.

The hoped-for gain is better recall, precision, and grounding. The expected cost is higher latency, because retrieval may take several turns.

## Architecture

Three modules, built in this order. Each starts at the bare minimum, then grows.

```
user query
    │
    ▼
┌─────────────────────┐
│  3. Multi-turn      │  keep querying until the result is good enough
│     memory query    │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐  ┌──────────────┐
│ 1. Vector│  │ 2. Knowledge │
│    store │  │    graph     │
└─────────┘  └──────────────┘
    │               │
    └───────┬───────┘
            ▼
      grounded answer
```

### 1. Vector store

Whole **utterances** (a user or agent turn) are embedded and stored. On a query, retrieve the most similar utterances. This is the fast first pass: *what was said that looks relevant?*

Needs an **embedding model** to turn text into vectors. v1 uses OpenAI `text-embedding-3-small` (Mem0’s default). Vectors live in SQLite; a query embeds the question and ranks **all** stored vectors by cosine similarity.

### 2. Knowledge graph

An LLM (`gpt-4o-mini`) extracts **fact triples** from those utterances. Subject and object become **nodes**; the predicate becomes an **edge**. Negated facts use a `not_` prefix (`not_lives_in`). Pronoun-only / intensifier-only turns may extract nothing.

Query **fans out**. It does not merge scores:

1. **Vector list** — cosine over utterances.
2. **Graph list** — take those utterance ids, load their nodes, return the **1-hop** neighbourhood.

```bash
python -m kgr_memory add "I live in Brisbane" --role user
python -m kgr_memory query "Where do I live?"
```

### 3. Multi-turn memory query

The agent does not stop after one retrieval. It queries both stores, inspects what came back, and queries again. It decides when to stop using its own reasoning (for example a 1–10 “enough evidence” score). Then it answers the user.

Still cap the number of rounds so a loop cannot run forever. That cap can be small and dumb at first.

## Build principle

Start simple. Implement the smallest working version of each module, understand it, then extend. Do not jump to a production-scale design.

See [AGENTS.md](AGENTS.md) for how work on this repo should proceed.

## Current design decisions

Working answers for this stage. Change them if a later module shows they are wrong.

| Question | Current answer |
| --- | --- |
| What is stored? | **Utterance** in the vector store. **Fact triples** in the graph (subject / object as nodes, predicate as edge). Unsure this split is final. |
| Who writes the graph? | **`gpt-4o-mini` extracts** triples. `I` maps to the utterance `role`. Negation is `not_*`. Skip unresolved pronouns and intensifiers. |
| Graph query entry? | **Vector-seeded.** Question → cosine utterances → edges on those ids → start nodes → 1-hop. Two printed lists; no fused rank. |
| When does multi-turn stop? | The **LLM decides**, by reasoning and a score (e.g. 1–10). Always keep a hard max-round cap. |
| Where does memory live? | **SQLite** (one local file). Not a CSV, not process RAM as the source of truth. PostgreSQL later if we outgrow SQLite. Wikidata / DBpedia can wait. |
| Embeddings? | **Yes.** OpenAI [`text-embedding-3-small`](https://platform.openai.com/docs/models/text-embedding-3-small) (1536-d). Same default as [Mem0](https://docs.mem0.ai/components/embedders/models/openai). Store vectors in SQLite; query is brute-force cosine over all rows. |

### Advice that can change

- Splitting utterance vs triple is a good first cut: similarity over “what was said”, graph over “what is true”. If personal chat is too noisy for clean triples, we may store coarser nodes later.
- LLM-extracted graphs will be messy (duplicate entities, bad relations). Fine at first; entity merging comes after a working extractor.
- `add` writes the utterance **and** extracts triples. `ingest <id>` re-runs extraction for that row.
- An LLM stop-score is easy to demo and easy to fool. Keep the max-round cap, and later we can log scores against actual recall/precision.
- Local memory means we control the data and can measure grounding. A public KG is an enrichment step, not a starting point.
- SQLite is the baseline store (utterances, vectors, later triples). Skip CSV. Skip a hosted vector DB. PostgreSQL / pgvector is an upgrade path, not v1.
- `text-embedding-3-small` is an API cost/latency choice made to stay comparable with Mem0. A local embedder can replace it later without changing the SQLite schema.

## Vector store (v1)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env            # set OPENAI_API_KEY

python -m kgr_memory add "I live in Brisbane" --role user
python -m kgr_memory query "Where do I live?"
```

SQLite file defaults to `data/memory.sqlite3`.
