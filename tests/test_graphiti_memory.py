import asyncio
from datetime import datetime, timezone

import pytest

from kgr_memory.graphiti_memory import add_text, parse_reference_time


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


def test_add_text_uses_now_when_no_session_date_is_given():
    seen = {}

    class Graph:
        async def add_episode(self, **kwargs):
            seen.update(kwargs)

    before = datetime.now(timezone.utc)
    asyncio.run(add_text(Graph(), "hello", "chat-1"))
    after = datetime.now(timezone.utc)
    assert before <= seen["reference_time"] <= after
