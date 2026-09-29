"""One chat turn. The model chooses whether to search memory and whether to store a fact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from kgr_memory.prompts import DEFAULT_PROMPT
from kgr_memory.react import Decision, Fact, _merge, _stop_or_continue

CHAT_SEARCH_ROUNDS = 3

PLAN_SYSTEM = (
    "You decide how a conversation uses a knowledge-graph memory. "
    "The user talks to you, not to the memory. You choose the tool. "
    "For the latest message only, return JSON. "
    "Set search to true when a good answer depends on something already said "
    "(a preference, plan, person, place, or past event). "
    "Set search to false for a greeting, thanks, or a new fact that does not ask anything. "
    "When search is true, query is a short search string; otherwise query is null. "
    "Set store to true only when the message states a durable fact about a person. "
    "Set store to false for greetings, questions, and small talk that states no fact. "
    "When store is true, memory is one plain sentence of that fact; otherwise memory is null. "
    "Do not store the question itself. "
    'Example: "Where should I go for food?" means search true, query "food preferences", store false. '
    '"I like Thai and Vietnamese food." means search false, store true, '
    'memory "The user likes Thai and Vietnamese food."'
)

REPLY_SYSTEM = (
    "You are the person the user is talking to. "
    "Use the remembered facts when they answer the message. "
    "If a new fact is listed as just stated, acknowledge it. "
    "If the facts are empty and the user asked for something personal, "
    "say you do not know that yet and give a short general suggestion. "
    "Keep the reply to a few sentences."
)


@dataclass
class TurnPlan:
    search: bool
    query: str | None
    store: bool
    memory: str | None


def parse_plan(raw: str) -> TurnPlan:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    query = data.get("query")
    memory = data.get("memory")
    return TurnPlan(
        search=bool(data.get("search")),
        query=query if isinstance(query, str) and query.strip() else None,
        store=bool(data.get("store")),
        memory=memory if isinstance(memory, str) and memory.strip() else None,
    )


def load_locomo_sample() -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / "locomo_session1_10.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def run_turn(
    text: str,
    *,
    plan: Callable[[str, list[dict]], TurnPlan],
    search: Callable[[str], Awaitable[tuple[list[Fact], float]]],
    store: Callable[[str], Awaitable[None]],
    decide: Callable[[str, list[Fact], str, list[str]], Decision],
    answer: Callable[[str, list[Fact], list[dict], str | None], str],
    history: list[dict] | None = None,
    prompt_name: str = DEFAULT_PROMPT,
    max_rounds: int = CHAT_SEARCH_ROUNDS,
) -> AsyncIterator[dict]:
    """Yield debug events, then the assistant reply.

    search, store, decide, and answer are injected so a test can skip the network.
    """
    history = history or []
    yield {"type": "status", "text": "deciding whether to search or store"}
    chosen = plan(text, history)
    query = chosen.query or text
    yield {
        "type": "plan",
        "search": chosen.search,
        "query": query if chosen.search else None,
        "store": chosen.store,
        "memory": chosen.memory,
    }

    facts: list[Fact] = []
    if chosen.search:
        sent: list[str] = []
        current = query.strip() or text
        for n in range(1, max_rounds + 1):
            yield {"type": "status", "text": f"search {n}: {current}"}
            found, seconds = await search(current)
            facts = _merge(facts, found)
            sent.append(current)
            decision = decide(text, facts, prompt_name, sent)
            yield {
                "type": "search",
                "round": n,
                "query": current,
                "seconds": round(seconds, 3),
                "enough": decision.enough,
                "next_query": decision.next_query,
                "facts": [{"edge_id": fact.edge_id, "text": fact.text} for fact in found],
            }
            _final, reason, nxt = _stop_or_continue(
                decision,
                round_n=n,
                cap=max_rounds,
                sent=sent,
                question=text,
                facts=facts,
                answer=lambda _q, _f: "",
            )
            if reason or not nxt:
                break
            current = nxt

    memory = chosen.memory or text
    if chosen.store:
        yield {"type": "status", "text": "writing that fact into the graph"}
        await store(memory)
        yield {"type": "store", "text": memory}

    reply = answer(text, facts, history, memory if chosen.store else None)
    yield {"type": "assistant", "text": reply}
