"""ReAct loop over a memory search tool.

The model does not receive a fixed retrieval. Each round it sees the facts so far
and returns whether they are enough and, if not, a next query. The scored answer
is written afterwards by the same answer function one-shot uses.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from kgr_memory.prompts import DEFAULT_PROMPT, system_prompt

MAX_ROUNDS = 10


@dataclass
class Fact:
    edge_id: str
    text: str


@dataclass
class Decision:
    enough: bool
    next_query: str | None = None
    answer: str | None = None


@dataclass
class RoundLog:
    round: int
    query: str
    facts: list[Fact]
    search_seconds: float
    enough: bool | None = None
    next_query: str | None = None


@dataclass
class AskResult:
    question: str
    prompt_name: str
    mode: str
    answer: str
    rounds: list[RoundLog] = field(default_factory=list)
    stopped_because: str = ""

    def fact_texts(self) -> list[str]:
        seen: list[str] = []
        for round_log in self.rounds:
            for fact in round_log.facts:
                if fact.text not in seen:
                    seen.append(fact.text)
        return seen


SearchFn = Callable[[str], tuple[list[Fact], float]]
DecideFn = Callable[[str, list[Fact], str, list[str]], Decision]
AnswerFn = Callable[[str, list[Fact]], str]


def _json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def parse_decision(raw: str) -> Decision:
    data = _json_object(raw)
    nxt = data.get("next_query")
    answer = data.get("answer")
    return Decision(
        enough=bool(data.get("enough")),
        next_query=nxt if isinstance(nxt, str) and nxt.strip() else None,
        answer=answer if isinstance(answer, str) and answer.strip() else None,
    )


def _merge(existing: list[Fact], new: list[Fact]) -> list[Fact]:
    seen = {fact.edge_id for fact in existing}
    out = list(existing)
    for fact in new:
        if fact.edge_id in seen:
            continue
        seen.add(fact.edge_id)
        out.append(fact)
    return out


def _stop_or_continue(
    decision: Decision,
    *,
    round_n: int,
    cap: int,
    sent: list[str],
    question: str,
    facts: list[Fact],
    answer: AnswerFn,
) -> tuple[str | None, str | None, str | None]:
    """Return (final_answer, stop_reason, next_query). The answer always comes from answer()."""
    if decision.enough or round_n == cap:
        reason = "enough" if decision.enough else "cap"
        return answer(question, facts), reason, None
    nxt = (decision.next_query or "").strip()
    if not nxt or nxt in sent:
        return answer(question, facts), "no_new_query", None
    return None, None, nxt


@dataclass
class _Step:
    """One beat of the search loop. decision is None until that search has returned."""

    round: int
    query: str
    found: list[Fact] | None = None
    seconds: float = 0.0
    decision: Decision | None = None
    facts: list[Fact] | None = None
    final: str | None = None
    reason: str | None = None


async def _search_rounds(
    question: str,
    search,
    decide: DecideFn,
    answer: AnswerFn,
    *,
    prompt_name: str,
    max_rounds: int,
    query: str,
) -> AsyncIterator[_Step]:
    """Yield before each search, then the round. Stop rules live in _stop_or_continue."""
    facts: list[Fact] = []
    sent: list[str] = []
    current = query
    for n in range(1, max_rounds + 1):
        yield _Step(round=n, query=current)
        found, seconds = await search(current)
        facts = _merge(facts, found)
        sent.append(current)
        decision = decide(question, facts, prompt_name, sent)
        final, reason, nxt = _stop_or_continue(
            decision,
            round_n=n,
            cap=max_rounds,
            sent=sent,
            question=question,
            facts=facts,
            answer=answer,
        )
        yield _Step(
            round=n,
            query=current,
            found=found,
            seconds=seconds,
            decision=decision,
            facts=facts,
            final=final,
            reason=reason,
        )
        if reason:
            return
        current = nxt or current


def ask_react(
    question: str,
    search: SearchFn,
    decide: DecideFn,
    answer: AnswerFn,
    *,
    prompt_name: str = DEFAULT_PROMPT,
    max_rounds: int = MAX_ROUNDS,
) -> AskResult:
    """Search, then let the model stop or query again. Cap is max_rounds."""

    async def search_async(query: str):
        return search(query)

    return asyncio.run(
        ask_react_async(
            question,
            search_async,
            decide,
            answer,
            prompt_name=prompt_name,
            max_rounds=max_rounds,
        )
    )


async def ask_react_async(
    question: str,
    search,
    decide: DecideFn,
    answer: AnswerFn,
    *,
    prompt_name: str = DEFAULT_PROMPT,
    max_rounds: int = MAX_ROUNDS,
) -> AskResult:
    """Same stop rules as ask_react. search is async and returns (facts, seconds)."""
    system_prompt(prompt_name)
    result = AskResult(question=question, prompt_name=prompt_name, mode="react", answer="")
    facts: list[Fact] = []
    async for step in _search_rounds(
        question,
        search,
        decide,
        answer,
        prompt_name=prompt_name,
        max_rounds=max_rounds,
        query=question,
    ):
        decision = step.decision
        if decision is None:
            continue
        facts = step.facts or []
        result.rounds.append(
            RoundLog(
                round=step.round,
                query=step.query,
                facts=list(step.found or []),
                search_seconds=step.seconds,
                enough=decision.enough,
                next_query=decision.next_query,
            )
        )
        if step.reason:
            result.answer = step.final or ""
            result.stopped_because = step.reason
            return result
    result.answer = answer(question, facts)
    result.stopped_because = "cap"
    return result


def ask_once(
    question: str,
    search: SearchFn,
    answer: AnswerFn,
    *,
    prompt_name: str = DEFAULT_PROMPT,
) -> AskResult:
    """One search, then an answer. No second call to the graph."""

    async def search_async(query: str):
        return search(query)

    return asyncio.run(
        ask_once_async(question, search_async, answer, prompt_name=prompt_name)
    )


async def ask_once_async(
    question: str,
    search,
    answer: AnswerFn,
    *,
    prompt_name: str = DEFAULT_PROMPT,
) -> AskResult:
    """Same one-search path as ask_once. search is async and returns (facts, seconds)."""
    found, seconds = await search(question)
    return AskResult(
        question=question,
        prompt_name=prompt_name,
        mode="oneshot",
        answer=answer(question, found),
        rounds=[
            RoundLog(round=1, query=question, facts=list(found), search_seconds=seconds)
        ],
        stopped_because="oneshot",
    )
