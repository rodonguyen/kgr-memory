import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kgr_memory.graphiti_memory import (
    add_text,
    facts_from_search,
    parse_reference_time,
    search_facts,
)
from kgr_memory.react import Fact


def test_longmemeval_session_date_is_utc():
    when = parse_reference_time("2023/05/20 (Sat) 02:21")
    assert when == datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)


def test_locomo_session_date_is_utc():
    when = parse_reference_time("1:56 pm on 8 May, 2023")
    assert when == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)


def test_naive_datetime_is_treated_as_utc():
    when = parse_reference_time(datetime(2023, 5, 8, 13, 56))
    assert when.tzinfo == timezone.utc
    assert when.hour == 13


def test_unknown_date_is_rejected():
    with pytest.raises(ValueError):
        parse_reference_time("next Tuesday")


def test_add_text_passes_the_session_date():
    seen = {}

    class Graph:
        async def add_episode(self, **kwargs):
            seen.update(kwargs)

    asyncio.run(
        add_text(
            Graph(),
            "user: I graduated in May",
            "e47becba",
            name="answer_280352e9:0",
            reference_time="2023/05/20 (Sat) 02:21",
        )
    )
    assert seen["group_id"] == "e47becba"
    assert seen["name"] == "answer_280352e9:0"
    assert seen["episode_body"] == "user: I graduated in May"
    assert seen["reference_time"] == datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)


def test_fact_text_includes_valid_and_invalid_dates():
    edge = SimpleNamespace(
        uuid="e1",
        fact="Rodo lives in Brisbane",
        valid_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
        invalid_at=None,
    )
    facts = facts_from_search([edge], [], [])
    assert facts == [
        Fact(
            "e1",
            "Rodo lives in Brisbane (valid 2023-05-08T00:00:00+00:00; invalid present)",
        )
    ]


def test_invalid_date_is_kept_when_a_fact_was_replaced():
    edge = SimpleNamespace(
        uuid="e2",
        fact="Rodo lives in Sydney",
        valid_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
        invalid_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
    )
    text = facts_from_search([edge], [], [])[0].text
    assert "invalid 2023-05-08T00:00:00+00:00" in text


def test_entity_summary_is_returned_with_the_facts():
    node = SimpleNamespace(uuid="n1", name="Caroline", summary="Attended a support group.")
    facts = facts_from_search([], [node], [])
    assert facts == [Fact("n1", "Caroline: Attended a support group.")]


def test_empty_entity_summary_is_skipped():
    node = SimpleNamespace(uuid="n1", name="Caroline", summary="  ")
    assert facts_from_search([], [node], []) == []


def test_search_returns_dated_facts_summaries_and_episodes():
    class Results:
        edges = [
            SimpleNamespace(
                uuid="e1",
                fact="Caroline attended a support group",
                valid_at=datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc),
                invalid_at=None,
            )
        ]
        nodes = [SimpleNamespace(uuid="n1", name="Caroline", summary="Went to a support group.")]
        episodes = [SimpleNamespace(uuid="ep1", content="Caroline: I went to a support group.")]

    seen = {}

    class Graph:
        async def search_(self, query, config, group_ids):
            seen["query"] = query
            seen["groups"] = group_ids
            seen["nodes"] = config.node_config is not None
            seen["edges"] = config.edge_config is not None
            seen["episodes"] = config.episode_config is not None
            return Results()

    facts, _seconds = asyncio.run(search_facts(Graph(), "support group", "conv-26"))
    assert seen == {
        "query": "support group",
        "groups": ["conv-26"],
        "nodes": True,
        "edges": True,
        "episodes": True,
    }
    assert "valid 2023-05-08T13:56:00+00:00" in facts[0].text
    assert "invalid present" in facts[0].text
    assert facts[1].text == "Caroline: Went to a support group."
    assert facts[2].text == "Caroline: I went to a support group."


def test_add_text_uses_now_when_no_session_date_is_given():
    seen = {}

    class Graph:
        async def add_episode(self, **kwargs):
            seen.update(kwargs)

    before = datetime.now(timezone.utc)
    asyncio.run(add_text(Graph(), "hello", "chat-1"))
    after = datetime.now(timezone.utc)
    assert before <= seen["reference_time"] <= after
