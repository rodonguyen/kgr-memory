# kgr-memory

Knowledge graph for an LLM agent's memory, with reasoning.

The assignment path calls Graphiti (unchanged) against local Neo4j. The model is `openai/gpt-4o-mini`. Embeddings are `qwen/qwen3-embedding-8b` (4096-d). The agent may search the graph again until it says the facts are enough, or until 10 rounds. `--prompt` is `strict`, `balanced`, or `loose`: that is the “enough” experiment. The SQLite `add` / `query` commands are the earlier prototype.

```bash
python -m kgr_memory store "I live in Brisbane" --group demo
python -m kgr_memory search "Where do I live?" --group demo
python -m kgr_memory ask "Where do I live?" --group demo --mode react --prompt balanced
python -m kgr_memory ask "Where do I live?" --group demo --mode oneshot
```

Live checks use a few short sentences in their own `--group`. Do not point a test at a full benchmark file until that small case is already working.

Benchmark files live under `data/` and are gitignored:

| Set | Path | What it is |
| --- | --- | --- |
| LoCoMo | `data/locomo/locomo10.json` | The only published size: 10 conversations. No small/full split. |
| LoCoMo slice | `kgr_memory/fixtures/locomo_session1_10.json` | First 10 turns of session 1, used by the chat button. |
| LongMemEval-S | `data/longmemeval/longmemeval_s_cleaned.json` | Small-history setting, 500 questions. The medium file is not downloaded. |

```bash
python -m kgr_memory chat
```

That opens http://127.0.0.1:8765. You talk to the model. On each message it decides whether to search the graph and whether the message is a fact worth storing. The right-hand column shows those calls as they happen. **LoCoMo 10** replays the first 10 turns of one LoCoMo session into a fresh group. Search stops after 3 rounds in the chat; the `ask` command still caps at 10.

Griffith University course project. Python. Assessment: project 60%, final presentation 10%, final report 30%.

## Course context

Knowledge graphs are used in expert systems, web search, question answering, and recommendation. This course covers how machine-processible knowledge is organised, synthesised, and managed in large graphs such as DBpedia and Wikidata, and how those graphs can be enriched with AI techniques.

This project applies that idea to **agent memory**: store, relate, and retrieve facts so the model answers from evidence instead of guessing.

## Goal

Build a hybrid memory system so an LLM agent can return **correct, complete, and grounded** answers — especially in high-stakes settings where missing or invented information is costly.

The hoped-for gain is better recall, precision, and grounding. The expected cost is higher latency, because retrieval may take several turns.

## Architecture

The assignment path is Graphiti on local Neo4j. The model searches that graph, decides whether the facts are enough, and may search again. The SQLite vector store and triple extractor below are the earlier prototype (`add` / `query`). Do not extend that prototype for the course comparison.

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

The assignment path is the first table. The SQLite rows are the earlier prototype only.

| Question | Current answer |
| --- | --- |
| Where does the assignment memory live? | **Neo4j** via Graphiti. Bolt `localhost:7687`, user `neo4j`. One `group_id` per conversation. |
| Who writes and answers? | **`openai/gpt-4o-mini`** on OpenRouter. Embeddings are **`qwen/qwen3-embedding-8b`**, 4096 dimensions. |
| What does a chat turn do? | The model decides. It searches when the answer depends on something already said. It stores only a durable personal fact, as one sentence, not every line. |
| When does search stop? | JSON `{enough, next_query, answer}`. `ask` caps at **10** rounds. The chat page caps a search at **3**. `--prompt` is `strict`, `balanced`, or `loose` (the enough rule only). |
| Benchmarks on disk | **LoCoMo** `data/locomo/locomo10.json` (10 conversations, the full published set). **LongMemEval-S** `data/longmemeval/longmemeval_s_cleaned.json` (500 questions). Neither file is committed. |

Prototype (`add` / `query`), not the submitted graph:

| Question | Prototype answer |
| --- | --- |
| What is stored? | **Utterance** in SQLite. **Fact triples** in the same file. |
| Who writes the graph? | **`gpt-4o-mini` extracts** triples. Negation is `not_*`. |
| Graph query entry? | **Vector-seeded** cosine, then 1-hop. Two lists, not merged. |
| Embeddings? | OpenAI `text-embedding-3-small` (1536-d), Mem0’s old default. The assignment path does not use this. |

### Advice that can change

These notes are about the SQLite prototype, not the Graphiti path.

- Splitting utterance vs triple is a good first cut: similarity over “what was said”, graph over “what is true”.
- `add` writes the utterance and extracts triples. `ingest <id>` re-runs extraction for that row.
- The assignment path does not use SQLite or `text-embedding-3-small`. Its embedder is `qwen/qwen3-embedding-8b`.

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
