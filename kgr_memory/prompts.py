"""ReAct system prompts. The experiment swaps these and compares accuracy.

"enough" is the only instruction that changes. The graph, model, and round cap stay fixed.
"""

from __future__ import annotations

PROMPTS: dict[str, str] = {
    "strict": (
        "You control a knowledge-graph memory tool. "
        "Set enough to true only when a returned fact states the answer explicitly. "
        "A hint, a related entity, or a partial overlap is not enough. "
        "If it is not enough, set next_query to a new search string that is not one you already sent, "
        "and leave answer empty. "
        "If it is enough, set next_query to null and put the answer in answer, using only the facts."
    ),
    "balanced": (
        "You control a knowledge-graph memory tool. "
        "Set enough to true when the facts can support an answer to the question. "
        "If they cannot, set next_query to a different search string and leave answer empty. "
        "If they can, set next_query to null and answer from those facts only. "
        "Do not invent facts that were not returned."
    ),
    "loose": (
        "You control a knowledge-graph memory tool. "
        "Set enough to true when any returned fact is partly relevant, even if the answer is incomplete. "
        "Prefer stopping over another search. "
        "If you stop, answer from the facts you have and say what is missing. "
        "If you continue, next_query must be a different search string."
    ),
}

DEFAULT_PROMPT = "balanced"


def system_prompt(name: str) -> str:
    try:
        return PROMPTS[name]
    except KeyError as exc:
        known = ", ".join(PROMPTS)
        raise ValueError(f"unknown prompt {name!r}; choose one of: {known}") from exc
