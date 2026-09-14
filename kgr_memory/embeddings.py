"""Embedding models for the vector store.
v1 uses OpenAI ``text-embedding-3-small`` (1536-d). That is Mem0's default
embedder, so results are easier to compare. See:
https://docs.mem0.ai/components/embedders/models/openai
"""

from __future__ import annotations
import os
from typing import Protocol
from dotenv import load_dotenv
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv()


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:

    def __init__(
        self, model: str = EMBEDDING_MODEL, api_key: str | None = None
    ) -> None:
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
        self._client = OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned_input = [t.replace("\n", " ") for t in texts]
        resp = self._client.embeddings.create(model=self.model, input=cleaned_input)
        ordered = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
