"""ReAct loop over a memory search tool.

The model does not receive a fixed retrieval. Each round it sees the facts so far
and returns whether they are enough, a next query, or an answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

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


def parse_decision(raw: str) -> Decision:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
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
    """Return (final_answer, stop_reason, next_query). Continue when all three are unset except next_query."""
    if decision.enough or round_n == cap:
        text = decision.answer or answer(question, facts)
        return text, ("enough" if decision.enough else "cap"), None
    nxt = (decision.next_query or "").strip()
    if not nxt or nxt in sent:
        text = decision.answer or answer(question, facts)
        return text, "no_new_query", None
    return None, None, nxt


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
    system_prompt(prompt_name)
    result = AskResult(question=question, prompt_name=prompt_name, mode="react", answer="")
    facts: list[Fact] = []
    query = question
    sent: list[str] = []

    for n in range(1, max_rounds + 1):
        found, seconds = search(query)
        facts = _merge(facts, found)
        sent.append(query)
        decision = decide(question, facts, prompt_name, sent)
        result.rounds.append(
            RoundLog(
                round=n,
                query=query,
                facts=list(found),
                search_seconds=seconds,
                enough=decision.enough,
                next_query=decision.next_query,
            )
        )
        final, reason, nxt = _stop_or_continue(
            decision,
            round_n=n,
            cap=max_rounds,
            sent=sent,
            question=question,
            facts=facts,
            answer=answer,
        )
        if reason:
            result.answer = final or ""
            result.stopped_because = reason
            return result
        query = nxt or query

    result.answer = answer(question, facts)
    result.stopped_because = "cap"
    return result


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
    query = question
    sent: list[str] = []
    for n in range(1, max_rounds + 1):
        found, seconds = await search(query)
        facts = _merge(facts, found)
        sent.append(query)
        decision = decide(question, facts, prompt_name, sent)
        result.rounds.append(
            RoundLog(
                round=n,
                query=query,
                facts=list(found),
                search_seconds=seconds,
                enough=decision.enough,
                next_query=decision.next_query,
            )
        )
        final, reason, nxt = _stop_or_continue(
            decision,
            round_n=n,
            cap=max_rounds,
            sent=sent,
            question=question,
            facts=facts,
            answer=answer,
        )
        if reason:
            result.answer = final or ""
            result.stopped_because = reason
            return result
        query = nxt or query
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
    found, seconds = search(question)
    result = AskResult(
        question=question,
        prompt_name=prompt_name,
        mode="oneshot",
        answer=answer(question, found),
        rounds=[
            RoundLog(round=1, query=question, facts=list(found), search_seconds=seconds)
        ],
        stopped_because = "oneshot",
    )
    return result
