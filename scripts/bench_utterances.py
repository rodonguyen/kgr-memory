"""Run 100 LoCoMo turns and 100 LongMemEval-S turns through the chat path.

Writes one JSON line per turn under data/runs/. This is a timing slice, not the full benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from kgr_memory.chat_turn import run_turn
from kgr_memory.graphiti_memory import add_text, build_graphiti, ensure_indices, search_facts
from kgr_memory.live import context_tokens, decide_live, plan_live, reply_live

LOCOMO = Path(r"D:\Github\AgenticMemory\data\locomo10.json")
LME = Path(r"D:\Github\kgr-memory\data\longmemeval\longmemeval_s_cleaned.json")


def load_locomo(n: int) -> list[dict]:
    data = json.loads(LOCOMO.read_text(encoding="utf-8"))
    conv = data[0]["conversation"]
    turns = []
    i = 1
    while f"session_{i}" in conv and len(turns) < n:
        for turn in conv[f"session_{i}"]:
            text = f"{turn['speaker']}: {turn['text']}"
            turns.append(
                {
                    "dataset": "locomo",
                    "sample_id": data[0].get("sample_id"),
                    "dia_id": turn.get("dia_id"),
                    "role": turn.get("speaker"),
                    "text": text,
                    "chars": len(text),
                }
            )
            if len(turns) == n:
                break
        i += 1
    return turns


def load_longmemeval(n: int) -> list[dict]:
    data = json.loads(LME.read_text(encoding="utf-8"))
    item = data[0]
    turns = []
    for session_id, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
        for index, turn in enumerate(session):
            text = f"{turn['role']}: {turn['content']}"
            turns.append(
                {
                    "dataset": "longmemeval_s",
                    "sample_id": item["question_id"],
                    "question_type": item["question_type"],
                    "dia_id": f"{session_id}:{index}",
                    "role": turn["role"],
                    "text": text,
                    "chars": len(text),
                }
            )
            if len(turns) == n:
                return turns
    return turns


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return round(ordered[index], 3)


def summarize(rows: list[dict]) -> dict:
    def col(name: str, subset: list[dict] | None = None) -> list[float]:
        source = rows if subset is None else subset
        return [float(row[name]) for row in source if row.get(name) is not None]

    stored = [row for row in rows if row["stored"]]
    searched = [row for row in rows if row["searched"]]
    return {
        "turns": len(rows),
        "stored": len(stored),
        "searched": len(searched),
        "errors": sum(1 for row in rows if row.get("error")),
        "chars_mean": round(sum(row["chars"] for row in rows) / len(rows), 1) if rows else None,
        "total_s_mean": round(sum(col("total_s")) / len(rows), 3) if rows else None,
        "total_s_p50": _pct(col("total_s"), 0.5),
        "total_s_p95": _pct(col("total_s"), 0.95),
        "plan_s_p50": _pct(col("plan_s"), 0.5),
        "reply_s_p50": _pct(col("reply_s"), 0.5),
        "store_s_p50": _pct(col("store_s"), 0.5),
        "store_s_p95": _pct(col("store_s"), 0.95),
        "store_s_mean": round(sum(col("store_s")) / len(stored), 3) if stored else None,
        "search_s_p50": _pct(col("search_s", searched), 0.5),
        "search_s_p95": _pct(col("search_s", searched), 0.95),
        "rounds_mean": round(sum(row["rounds"] for row in searched) / len(searched), 2) if searched else None,
        "context_tokens_mean": (
            round(sum(row["context_tokens"] or 0 for row in searched) / len(searched), 1) if searched else None
        ),
    }


async def run_dataset(graphiti, turns: list[dict], group_id: str, log_path: Path) -> list[dict]:
    rows = []
    history: list[dict] = []
    for number, turn in enumerate(turns, start=1):
        clock = {"plan_s": 0.0, "reply_s": 0.0, "store_s": 0.0}

        def plan(text, prior, clock=clock):
            started = time.perf_counter()
            chosen = plan_live(text, prior)
            clock["plan_s"] += time.perf_counter() - started
            return chosen

        async def search(query: str):
            return await search_facts(graphiti, query, group_id)

        async def store(memory: str, clock=clock):
            started = time.perf_counter()
            await add_text(graphiti, memory, group_id)
            clock["store_s"] += time.perf_counter() - started

        def answer(text, facts, prior, just, clock=clock):
            started = time.perf_counter()
            reply = reply_live(text, facts, prior, just)
            clock["reply_s"] += time.perf_counter() - started
            return reply

        started = time.perf_counter()
        events = []
        error = None
        try:
            async for event in run_turn(
                turn["text"],
                plan=plan,
                search=search,
                store=store,
                decide=decide_live,
                answer=answer,
                history=history[-8:],
                prompt_name="balanced",
            ):
                events.append(event)
        except Exception as exc:
            error = str(exc)
        total = time.perf_counter() - started
        searches = [event for event in events if event["type"] == "search"]
        plan_event = next((event for event in events if event["type"] == "plan"), {})
        reply = next((event.get("text") for event in events if event["type"] == "assistant"), "")
        fact_texts = [fact["text"] for event in searches for fact in event.get("facts") or []]
        row = {
            "dataset": turn["dataset"],
            "group_id": group_id,
            "i": number,
            "dia_id": turn.get("dia_id"),
            "role": turn.get("role"),
            "chars": turn["chars"],
            "question_type": turn.get("question_type"),
            "searched": bool(plan_event.get("search")),
            "stored": bool(plan_event.get("store")),
            "rounds": len(searches),
            "enough": searches[-1]["enough"] if searches else None,
            "search_s": round(sum(event["seconds"] for event in searches), 3),
            "plan_s": round(clock["plan_s"], 3),
            "store_s": round(clock["store_s"], 3),
            "reply_s": round(clock["reply_s"], 3),
            "total_s": round(total, 3),
            "context_tokens": context_tokens(fact_texts),
            "error": error,
            "reply_chars": len(reply or ""),
        }
        rows.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"{turn['dataset']} {number}/{len(turns)} total={row['total_s']:.1f}s "
            f"store={row['store_s']:.1f}s search={row['search_s']:.1f}s "
            f"stored={row['stored']} searched={row['searched']}",
            flush=True,
        )
        if reply:
            history.append({"role": "user", "content": turn["text"][:500]})
            history.append({"role": "assistant", "content": reply})
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--log", type=Path, default=Path("data/runs/bench-100.jsonl"))
    args = parser.parse_args()
    locomo = load_locomo(args.n)
    longmem = load_longmemeval(args.n)
    if len(locomo) < args.n or len(longmem) < args.n:
        raise SystemExit(f"short slice locomo={len(locomo)} lme={len(longmem)}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    graphiti = build_graphiti()
    try:
        await ensure_indices(graphiti)
        all_rows = []
        for turns, group in (
            (locomo, "bench-locomo"),
            (longmem, "bench-lme"),
        ):
            all_rows.extend(await run_dataset(graphiti, turns, group, args.log))
        summary = {
            "locomo": summarize([row for row in all_rows if row["dataset"] == "locomo"]),
            "longmemeval_s": summarize([row for row in all_rows if row["dataset"] == "longmemeval_s"]),
        }
        summary_path = args.log.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
