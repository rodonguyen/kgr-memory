"""Graphiti calls. The library owns the graph. This module only adds and searches."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)

from kgr_memory.react import Fact

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "4096"))
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


def require_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from the environment or .env")
    return key


def build_graphiti():
    key = require_key()
    llm_config = LLMConfig(
        api_key=key,
        model=LLM_MODEL,
        small_model=LLM_MODEL,
        base_url=OPENROUTER_BASE_URL,
    )
    llm_client = OpenAIGenericClient(config=llm_config)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=key,
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=EMBEDDING_DIM,
            base_url=OPENROUTER_BASE_URL,
        )
    )
    return Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )


async def ensure_indices(graphiti) -> None:
    """One index build. Neo4jDriver already schedules it when the loop is running."""
    task = getattr(graphiti.driver, "_init_task", None)
    if task is not None:
        await task
        return
    await graphiti.build_indices_and_constraints()


_LONGMEMEVAL_DATE = "%Y/%m/%d (%a) %H:%M"
_LOCOMO_DATE = "%I:%M %p on %d %B, %Y"


def parse_reference_time(value: datetime | str) -> datetime:
    """Session dates from LongMemEval and LoCoMo, or an ISO timestamp.

    Graphiti dates a fact from this value. A missing zone is treated as UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = value.strip()
    if text.endswith(" UTC"):
        text = text[: -len(" UTC")].strip()
    for fmt in (_LONGMEMEVAL_DATE, _LOCOMO_DATE):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"unrecognised session date: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


async def add_text(
    graphiti,
    text: str,
    group_id: str,
    name: str | None = None,
    reference_time: datetime | str | None = None,
) -> None:
    """Write one episode. reference_time is the session date, not the clock time of the run."""
    when = (
        datetime.now(timezone.utc)
        if reference_time is None
        else parse_reference_time(reference_time)
    )
    await graphiti.add_episode(
        name=name or text[:80],
        episode_body=text,
        source_description="kgr-memory",
        reference_time=when,
        source=EpisodeType.message,
        group_id=group_id,
    )


def _date_stamp(valid_at, invalid_at) -> str:
    start = valid_at.isoformat() if valid_at else "date unknown"
    end = invalid_at.isoformat() if invalid_at else "present"
    return f"(valid {start}; invalid {end})"


def memory_search_config(limit: int) -> SearchConfig:
    """Edges, entity summaries, and episodes. Same RRF hybrid the one-shot path already used."""
    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.rrf,
        ),
        episode_config=EpisodeSearchConfig(
            search_methods=[EpisodeSearchMethod.bm25],
            reranker=EpisodeReranker.rrf,
        ),
        limit=limit,
    )


def facts_from_search(edges, nodes, episodes) -> list[Fact]:
    """Dated fact edges, then entity summaries, then the raw episode when it is not already listed."""
    facts: list[Fact] = []
    seen: set[str] = set()

    def add(item_id: str, text: str) -> None:
        if not text or text in seen:
            return
        seen.add(text)
        facts.append(Fact(edge_id=item_id, text=text))

    for edge in edges:
        fact = getattr(edge, "fact", None)
        if not fact:
            continue
        stamp = _date_stamp(getattr(edge, "valid_at", None), getattr(edge, "invalid_at", None))
        add(str(edge.uuid), f"{fact} {stamp}")
    for node in nodes:
        name = (getattr(node, "name", None) or "").strip()
        summary = (getattr(node, "summary", None) or "").strip()
        if not name or not summary:
            continue
        add(str(node.uuid), f"{name}: {summary}")
    for episode in episodes:
        content = getattr(episode, "content", None)
        if not content:
            continue
        valid_at = getattr(episode, "valid_at", None)
        text = content if valid_at is None else f"{content} (valid {valid_at.isoformat()})"
        add(str(episode.uuid), text)
    return facts


async def search_facts(
    graphiti, query: str, group_id: str, limit: int = 10
) -> tuple[list[Fact], float]:
    started = time.perf_counter()
    results = await graphiti.search_(
        query,
        config=memory_search_config(limit),
        group_ids=[group_id],
    )
    elapsed = time.perf_counter() - started
    return facts_from_search(results.edges, results.nodes, results.episodes), elapsed
