# Working on kgr-memory

Griffith University 6005ICT project. Read [README.md](README.md) first. The assignment system is Graphiti plus a ReAct loop in this repo. The SQLite `add` / `query` path is the earlier prototype. Do not extend it.

## Principle

Start from the most basic thing. Get a tiny version working. Understand it. Then extend.

Do not introduce production, scalable, or framework-heavy design until the current module is understood as a small program.

If a change is not needed for the current step, do not add it.

## What to work on

1. **Graphiti on local Neo4j.** Call the installed `graphiti-core`. Do not edit the Graphiti checkout. `store` writes an episode. Pass `--at` with the session date when the line is from a benchmark; chat turns have no session date and use the clock. `search` returns dated fact edges, entity summaries, and the raw episode when that sentence is not already listed.
2. **Benchmark ingest** (`bench-ingest`) writes every raw `speaker: text` line of one LoCoMo conversation or one LongMemEval question into one `group_id`. It does not use the chat gate. A new `--group` keeps the previous ingest. `bench-ask` then asks that dataset question and appends `{"question_id", "hypothesis"}` for `data/longmemeval/evaluate_qa.py`. `scripts/bench_utterances.py` is only the timing slice through chat.
3. **ReAct loop** in `kgr_memory/react.py`. The model returns whether the facts are enough and, if not, a next query. It does not write the scored answer. `ask` stops at 10 rounds. One-shot and ReAct then call the same answer function, so a score gap is the extra search.
4. **Chat** (`python -m kgr_memory chat`). The user talks to the model. The model decides whether to search and whether to store a durable fact. Chat search stops at 3 rounds. The page shows those calls.

Live checks use a few short sentences in their own `--group`. Benchmark files are local only:

- `data/locomo/locomo10.json` — the full published LoCoMo set (10 conversations). There is no smaller official split. The chat button uses `kgr_memory/fixtures/locomo_session1_10.json` (10 turns).
- `data/longmemeval/longmemeval_s_cleaned.json` — LongMemEval-S, 500 questions. Do not download the medium file for a first run.

A timing run of 100 lines is not an accuracy score. LongMemEval’s published judge is `gpt-4o-2024-08-06` with the yes/no prompts in `data/longmemeval/evaluate_qa.py`, on all 500 questions. A `gpt-4o-mini` judge must be labelled as ours.

## While coding

- Models stay `openai/gpt-4o-mini` and `qwen/qwen3-embedding-8b` (4096-d) unless the README changes first
- Do not commit `.env`, `data/*.sqlite3`, `data/locomo/*.json`, `data/longmemeval/*.json`, or `data/runs/`
- Prefer one clear data path over abstractions for later
- Latency and accuracy are both reported. Do not hide extra retrieval rounds

## Git

- Do not add `Co-authored-by` (or any Cursor/agent trailer) to commits
- Commit as the repo author only
- After a commit, if a trailer was injected, strip it before pushing

## Target behaviour

The memory stack should help the agent answer with **correct, complete, and grounded** information. The graph is a tool the model uses. The comparison is one search against a loop that may search again, on the same graph.
