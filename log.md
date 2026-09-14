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