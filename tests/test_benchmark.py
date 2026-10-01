import asyncio
from datetime import datetime, timezone

from kgr_memory.benchmark import (
    ingest_messages,
    locomo_messages,
    longmemeval_messages,
    select_locomo,
    select_longmemeval,
)


LOCOMO = {
    "sample_id": "conv-26",
    "conversation": {
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to a support group."},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "That is good."},
        ],
        "session_2_date_time": "2:00 pm on 9 May, 2023",
        "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "I painted yesterday."},
        ],
    },
}

LONGMEMEVAL = {
    "question_id": "e47becba",
    "haystack_session_ids": ["s1", "answer_280352e9"],
    "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/29 (Mon) 18:04"],
    "haystack_sessions": [
        [{"role": "user", "content": "A river puzzle."}, {"role": "assistant", "content": "Try the chicken first."}],
        [{"role": "user", "content": "I graduated with a degree in Business Administration."}],
    ],
}


def test_locomo_keeps_every_turn_in_one_group():
    messages = locomo_messages(LOCOMO)
    assert [m.group_id for m in messages] == ["conv-26", "conv-26", "conv-26"]
    assert messages[0].body == "Caroline: I went to a support group."
    assert messages[0].name == "D1:1"
    assert messages[0].reference_time == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    assert messages[2].reference_time == datetime(2023, 5, 9, 14, 0, tzinfo=timezone.utc)


def test_longmemeval_keeps_the_haystack_in_the_question_group():
    messages = longmemeval_messages(LONGMEMEVAL)
    assert len(messages) == 3
    assert {m.group_id for m in messages} == {"e47becba"}
    assert messages[-1].body == "user: I graduated with a degree in Business Administration."
    assert messages[-1].name == "answer_280352e9:0"
    assert messages[-1].reference_time == datetime(2023, 5, 29, 18, 4, tzinfo=timezone.utc)


def test_group_override_does_not_mix_histories():
    messages = longmemeval_messages(LONGMEMEVAL, group_id="e47becba-v2")
    assert {m.group_id for m in messages} == {"e47becba-v2"}


def test_select_rejects_an_unknown_id():
    import pytest

    with pytest.raises(ValueError):
        select_locomo([LOCOMO], "conv-99")
    with pytest.raises(ValueError):
        select_longmemeval([LONGMEMEVAL], "missing")


def test_ingest_writes_the_raw_line_and_the_session_date():
    seen = []

    class Graph:
        async def add_episode(self, **kwargs):
            seen.append(kwargs)

    count = asyncio.run(ingest_messages(Graph(), locomo_messages(LOCOMO)))
    assert count == 3
    assert seen[0]["episode_body"] == "Caroline: I went to a support group."
    assert seen[0]["reference_time"] == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    assert seen[0]["group_id"] == "conv-26"
    assert "The user" not in seen[0]["episode_body"]
