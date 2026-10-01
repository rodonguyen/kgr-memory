"""ReAct system prompts. The experiment swaps these and compares accuracy.

"enough" is the only instruction that changes. The graph, model, and round cap stay fixed.
"""

from __future__ import annotations

_CONTRACT = (
    "You control a knowledge-graph memory tool. "
    "Do not answer the question in this step. Set answer to null. "
    "If enough is true, set next_query to null. "
    "If enough is false, set next_query to a search string you have not already sent."
)

# Only the enough rule changes. The stop/continue contract above is shared.
_ENOUGH: dict[str, str] = {
    "strict": (
        "Set enough to true only when a returned fact states the answer explicitly. "
        "A hint or a partial overlap is not enough."
    ),
    "balanced": (
        "Set enough to true when the returned facts can support an answer to the question."
    ),
    "loose": (
        "Set enough to true when any returned fact is partly relevant, even if the answer is incomplete."
    ),
}

DEFAULT_PROMPT = "balanced"


def system_prompt(name: str) -> str:
    try:
        enough = _ENOUGH[name]
    except KeyError as exc:
        known = ", ".join(_ENOUGH)
        raise ValueError(f"unknown prompt {name!r}; choose one of: {known}") from exc
    return f"{_CONTRACT} {enough}"
