"""Live OpenRouter calls for the ReAct decision and the final answer."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from kgr_memory.chat_turn import PLAN_SYSTEM, REPLY_SYSTEM, parse_plan
from kgr_memory.graphiti_memory import LLM_MODEL, OPENROUTER_BASE_URL, require_key
from kgr_memory.prompts import system_prompt
from kgr_memory.react import Decision, Fact, parse_decision

load_dotenv()

_openai: OpenAI | None = None

_DECISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "enough": {"type": "boolean"},
                "next_query": {"type": ["string", "null"]},
                "answer": {"type": ["string", "null"]},
            },
            "required": ["enough", "next_query", "answer"],
            "additionalProperties": False,
        },
    },
}


def _client() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=require_key(), base_url=OPENROUTER_BASE_URL)
    return _openai


def _fact_block(facts: list[Fact]) -> str:
    if not facts:
        return "(no facts)"
    return "\n".join(f"- {fact.text}" for fact in facts)


def decide_live(question: str, facts: list[Fact], prompt_name: str, sent: list[str]) -> Decision:
    prior = ", ".join(sent) if sent else "(none)"
    user = (
        f"Question: {question}\n"
        f"Searches already sent: {prior}\n"
        f"Facts:\n{_fact_block(facts)}"
    )
    response = _client().chat.completions.create(
        model=os.environ.get("LLM_MODEL", LLM_MODEL),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt(prompt_name)},
            {"role": "user", "content": user},
        ],
        response_format=_DECISION_SCHEMA,
    )
    content = response.choices[0].message.content or ""
    return parse_decision(content)


_PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "search": {"type": "boolean"},
                "query": {"type": ["string", "null"]},
                "store": {"type": "boolean"},
                "memory": {"type": ["string", "null"]},
            },
            "required": ["search", "query", "store", "memory"],
            "additionalProperties": False,
        },
    },
}


def _history_block(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for turn in history[-8:]:
        role = turn.get("role") or "user"
        content = turn.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def plan_live(text: str, history: list[dict]):
    response = _client().chat.completions.create(
        model=os.environ.get("LLM_MODEL", LLM_MODEL),
        temperature=0,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM},
            {
                "role": "user",
                "content": f"Recent talk:\n{_history_block(history)}\n\nLatest message:\n{text}",
            },
        ],
        response_format=_PLAN_SCHEMA,
    )
    return parse_plan(response.choices[0].message.content or "")


def reply_live(
    text: str,
    facts: list[Fact],
    history: list[dict],
    just_stored: str | None,
) -> str:
    stored = just_stored or "(nothing new)"
    response = _client().chat.completions.create(
        model=os.environ.get("LLM_MODEL", LLM_MODEL),
        temperature=0,
        messages=[
            {"role": "system", "content": REPLY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Recent talk:\n{_history_block(history)}\n"
                    f"Latest message:\n{text}\n"
                    f"Facts from memory:\n{_fact_block(facts)}\n"
                    f"Fact just stored:\n{stored}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def answer_live(question: str, facts: list[Fact]) -> str:
    response = _client().chat.completions.create(
        model=os.environ.get("LLM_MODEL", LLM_MODEL),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the facts. "
                    "If they do not contain the answer, say what is missing."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nFacts:\n{_fact_block(facts)}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def context_tokens(texts: list[str]) -> int | None:
    """Mem0 counted retrieved context with tiktoken cl100k_base. None if tiktoken is absent."""
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(text)) for text in texts)


def result_record(result, *, group_id: str, question_type: str | None) -> dict:
    facts = result.fact_texts()
    search_s = sum(r.search_seconds for r in result.rounds)
    return {
        "question": result.question,
        "question_type": question_type,
        "group_id": group_id,
        "mode": result.mode,
        "prompt_name": result.prompt_name,
        "stopped_because": result.stopped_because,
        "rounds_used": len(result.rounds),
        "answer": result.answer,
        "search_seconds": search_s,
        "context_tokens": context_tokens(facts),
        "rounds": [
            {
                "round": r.round,
                "query": r.query,
                "enough": r.enough,
                "next_query": r.next_query,
                "search_seconds": r.search_seconds,
                "facts": [{"edge_id": f.edge_id, "text": f.text} for f in r.facts],
            }
            for r in result.rounds
        ],
    }


def append_log(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
