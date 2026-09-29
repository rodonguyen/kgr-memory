# Design log

### 14.09.2026
Build order: vector store → knowledge graph → multi-turn query. So far we have simple vector store and knowledge graph. Each query is giving separate results from graph query and vector query. Multi-turn query will have a max turn limit.
I started simple with SQLite. PostgreSQL later
Embeddings: OpenAI `text-embedding-3-small` (same default as Mem0)
Query vectors with brute-force cosine over all rows (no ANN yet)
Vector store holds utterances; graph holds fact triples
Extractor: `gpt-4o-mini`; `I`/`me` maps to utterance role
Some design decisions in triple storing:
  - Negation supported as `not_*` predicates (`not_lives_in`)
  - ignore unresolved pronouns (him, her, them, it)
  - ignore intensifiers (so much, really, very)
  - ignore questions, jokes, hypotheticals
  - empty extract is allowed
  - time / status not supported (used to, anymore)
  - conflict not supported (old and new edges both stay)
  - entity merge is `lower()` only (Alex vs Alexander stay two nodes)
Graph query is vector-seeded then 1-hop
BM25 not in v1; later it belongs on the lexical/graph side, not inside cosine
`add` writes utterance and extracts triples; `ingest <id>` re-extracts

### 18.09.2026
downloaded LongMemEval_S cleaned JSON to `data/longmemeval/` (gitignored, ~264 MB)
source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
not wired into memory ingest yet

### 29.09.2026
Course loop stays in this repo. Do not fork mem0 or Graphiti.
Call Graphiti as the knowledge-graph tool. mem0 graph mode only if Graphiti's database blocks the first run.
Pi (https://pi.dev/) may sketch tools. It is not the measured agent.
Submitted contrast: one Graphiti search vs a capped ReAct loop in this repo that chooses the next graph action.
The local utterance store and triple extractor remain the from-scratch prototype. They are not the submitted graph.
Vault notes: RodoVault `Honours_work/6005ICT/`

### 30.09.2026 implementation
`store`, `search`, and `ask` call Graphiti. `ask --mode react` loops up to 10 rounds. `--prompt` selects strict, balanced, or loose.
`ask --mode oneshot` is one search. Traces go to `data/runs/asks.jsonl`.
SQLite `add` / `query` remain the old prototype.
Philosophy: the model is in control; the graph is a tool it reasons over.

### 30.09.2026 model change
LLM is `openai/gpt-4o-mini` for graph writing, the ReAct loop, and the answer. Embeddings stay `qwen/qwen3-embedding-8b` (4096-d).
ReAct cap is 10 rounds.
Logs should record the Zep and Mem0 measures: judge hit, search latency, total latency, p50/p95, context tokens, rounds, and LongMemEval question type. Construction time is recorded at ingest.