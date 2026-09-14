# Working on kgr-memory

Griffith University project. Full spec is in [README.md](README.md). If you are picking this up, start with [HANDOVER.md](HANDOVER.md). Follow this file when writing or changing code.

## Principle

Start from the most basic thing. Get a tiny version working. Understand it. Then extend.

Do not introduce production, scalable, or framework-heavy design until the current module is understood as a small program.

For each module:

1. Bare minimum that can run end to end
2. Check that the behaviour is clear
3. Only then add the next piece

If a change is not needed for the current step, do not add it.

## Build order

Do not start a later module until the earlier one has a working baseline.

1. **Vector store** — embed whole utterances with OpenAI `text-embedding-3-small` (Mem0 default); store in SQLite; retrieve by brute-force cosine similarity
2. **Knowledge graph** — `gpt-4o-mini` extracts triples into SQLite nodes/edges (negation = `not_*`); query is vector-seeded then 1-hop; print two lists, do not merge
3. **Multi-turn memory query** — agent queries both stores in a loop; LLM scores whether it has enough (e.g. 1–10) and stops; hard max-round cap

Memory is local. Populate from conversation or a benchmark later. Design decisions live in [log.md](log.md) and [README.md](README.md); do not freeze abstractions around them.

Each module may be extended after its baseline exists. Do not skip ahead to make the later module "ready".

## While coding

- Prefer a small Python script or package over a large layout
- Prefer **SQLite** as the source of truth (utterances, embeddings, later graph triples)
- Use OpenAI **`text-embedding-3-small`** for embeddings (Mem0’s default). Do not swap this without updating the README.
- Prefer brute-force cosine over all SQLite vectors before Chroma, Pinecone, pgvector, or ANN indexes
- Do not use CSV or process RAM as the durable store; RAM is only a cache if needed
- Prefer one clear data path over abstractions "for later"
- Keep the three modules separable so they can be understood alone
- Latency vs accuracy is an explicit tradeoff; do not hide extra retrieval rounds

## Git

- Do not add `Co-authored-by` (or any Cursor/agent trailer) to commits
- Commit as the repo author only
- After a commit, if a trailer was injected, strip it before pushing

## Target behaviour

The memory stack should help the agent answer with **correct, complete, and grounded** information. That matters most in high-stakes use, where the model must find the right memories instead of filling gaps.
