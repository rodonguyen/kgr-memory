"""One chat turn. The model chooses whether to search memory and whether to store a fact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from kgr_memory.prompts import DEFAULT_PROMPT
from kgr_memory.react import Decision, Fact, _json_object, _search_rounds

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
    "If a remembered fact answers the message, say that fact. "
    "Do not claim you do not know when a fact is listed. "
    "If a new fact is listed as just stated, acknowledge it. "
    "Only if the facts are empty and the user asked for something personal, "
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
    data = _json_object(raw)
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
        async for step in _search_rounds(
            text,
            search,
            decide,
            lambda _question, _facts: "",
            prompt_name=prompt_name,
            max_rounds=max_rounds,
            query=query.strip() or text,
        ):
            if step.decision is None:
                yield {"type": "status", "text": f"search {step.round}: {step.query}"}
                continue
            facts = step.facts or []
            yield {
                "type": "search",
                "round": step.round,
                "query": step.query,
                "seconds": round(step.seconds, 3),
                "enough": step.decision.enough,
                "next_query": step.decision.next_query,
                "facts": [
                    {"edge_id": fact.edge_id, "text": fact.text} for fact in (step.found or [])
                ],
            }

    memory = chosen.memory or text
    if chosen.store:
        yield {"type": "status", "text": "writing that fact into the graph"}
        await store(memory)
        yield {"type": "store", "text": memory}

    reply = answer(text, facts, history, memory if chosen.store else None)
    yield {"type": "assistant", "text": reply}
