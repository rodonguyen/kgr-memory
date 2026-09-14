"""LLM triple extractor (knowledge graph write path, function 1).

Maps I/me to the speaker. Negation uses a not_ predicate prefix.
Pronoun-only objects/subjects and intensifiers are dropped by the model
(empty list is allowed).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

EXTRACTOR_MODEL = "gpt-4o-mini"

load_dotenv()

_PROMPT = """Extract factual triples from the utterance.

Rules:
- Speaker is "{speaker}". Map first-person I/me/my to that speaker.
- Negated facts: prefix the predicate with not_ (example: not_lives_in).
- Skip a triple if subject or object is only an unresolved pronoun (him, her, them, it).
- Do not turn intensifiers into nodes (so much, really, very).
- Skip questions, jokes, and hypotheticals.
- Empty triples is allowed.

Return JSON only: {{"triples": [{{"subject": "", "predicate": "", "object": ""}}]}}
"""


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


class OpenAIExtractor:
    def __init__(self, model: str = EXTRACTOR_MODEL, api_key: str | None = None) -> None:
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
        self._client = OpenAI(api_key=key)

    def extract(self, text: str, speaker: str = "user") -> list[Triple]:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT.format(speaker=speaker)},
                {"role": "user", "content": text},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        raw = json.loads(content).get("triples", [])
        triples: list[Triple] = []
        for item in raw:
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if subject and predicate and obj:
                triples.append(Triple(subject=subject, predicate=predicate, object=obj))
        return triples
