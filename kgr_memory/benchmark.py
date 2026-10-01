"""Benchmark histories. Each line is stored raw, in its own group, with its session date.

The chat turn decides whether to store a paraphrase. A score cannot use that gate:
LongMemEval and LoCoMo grade the full history, and the other lines are the distractors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kgr_memory.graphiti_memory import add_text, parse_reference_time

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOCOMO_PATH = DATA_DIR / "locomo" / "locomo10.json"
LONGMEMEVAL_PATH = DATA_DIR / "longmemeval" / "longmemeval_s_cleaned.json"


@dataclass(frozen=True)
class BenchmarkMessage:
    group_id: str
    name: str
    body: str
    reference_time: datetime


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_locomo(data: list[dict], sample_id: str) -> dict:
    for item in data:
        if item.get("sample_id") == sample_id:
            return item
    raise ValueError(f"no LoCoMo conversation {sample_id!r}")


def select_longmemeval(data: list[dict], question_id: str) -> dict:
    for item in data:
        if item.get("question_id") == question_id:
            return item
    raise ValueError(f"no LongMemEval question {question_id!r}")


def locomo_messages(item: dict, group_id: str | None = None) -> list[BenchmarkMessage]:
    """Every turn of one conversation. One group, shared by later questions on that conversation."""
    group = group_id or item["sample_id"]
    conversation = item["conversation"]
    messages: list[BenchmarkMessage] = []
    session = 1
    while f"session_{session}" in conversation:
        when = parse_reference_time(conversation[f"session_{session}_date_time"])
        for turn in conversation[f"session_{session}"]:
            speaker = turn["speaker"]
            messages.append(
                BenchmarkMessage(
                    group_id=group,
                    name=turn["dia_id"],
                    body=f"{speaker}: {turn['text']}",
                    reference_time=when,
                )
            )
        session += 1
    if not messages:
        raise ValueError(f"LoCoMo conversation {item.get('sample_id')!r} has no turns")
    return messages


def longmemeval_messages(item: dict, group_id: str | None = None) -> list[BenchmarkMessage]:
    """Every line of one question's haystack. The next question id is a different group."""
    group = group_id or item["question_id"]
    messages: list[BenchmarkMessage] = []
    sessions = zip(
        item["haystack_session_ids"],
        item["haystack_sessions"],
        item["haystack_dates"],
    )
    for session_id, session, date in sessions:
        when = parse_reference_time(date)
        for index, turn in enumerate(session):
            messages.append(
                BenchmarkMessage(
                    group_id=group,
                    name=f"{session_id}:{index}",
                    body=f"{turn['role']}: {turn['content']}",
                    reference_time=when,
                )
            )
    if not messages:
        raise ValueError(f"LongMemEval question {item.get('question_id')!r} has no turns")
    return messages


async def ingest_messages(graphiti, messages: list[BenchmarkMessage]) -> int:
    """Write each line as its own episode. No plan, search, or reply."""
    for message in messages:
        await add_text(
            graphiti,
            message.body,
            message.group_id,
            name=message.name,
            reference_time=message.reference_time,
        )
    return len(messages)


def messages_for(dataset: str, item: dict, group_id: str | None) -> list[BenchmarkMessage]:
    if dataset == "locomo":
        return locomo_messages(item, group_id)
    if dataset == "longmemeval":
        return longmemeval_messages(item, group_id)
    raise ValueError(f"unknown dataset {dataset!r}")


def dataset_path(dataset: str) -> Path:
    if dataset == "locomo":
        return LOCOMO_PATH
    if dataset == "longmemeval":
        return LONGMEMEVAL_PATH
    raise ValueError(f"unknown dataset {dataset!r}")


def select_item(dataset: str, data: list[dict], item_id: str) -> dict:
    if dataset == "locomo":
        return select_locomo(data, item_id)
    if dataset == "longmemeval":
        return select_longmemeval(data, item_id)
    raise ValueError(f"unknown dataset {dataset!r}")


async def ingest_dataset(
    graphiti,
    dataset: str,
    item_id: str,
    group_id: str | None = None,
) -> tuple[str, int]:
    path = dataset_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    item = select_item(dataset, load_json(path), item_id)
    messages = messages_for(dataset, item, group_id)
    count = await ingest_messages(graphiti, messages)
    return messages[0].group_id, count
