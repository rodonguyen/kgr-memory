import asyncio
from types import SimpleNamespace

from kgr_memory.chat_turn import TurnPlan, load_locomo_sample, parse_plan, run_turn
from kgr_memory.graphiti_memory import ensure_indices, facts_from_search
from kgr_memory.react import Decision, Fact


def test_episode_text_is_kept_when_there_is_no_fact_edge():
    facts = facts_from_search(
        [],
        [SimpleNamespace(uuid="ep1", content="The user likes Thai food.")],
    )
    assert facts == [Fact("ep1", "The user likes Thai food.")]


def test_parse_plan_strips_fence():
    raw = '```json\n{"search": true, "query": "food preferences", "store": false, "memory": null}\n```'
    plan = parse_plan(raw)
    assert plan == TurnPlan(True, "food preferences", False, None)


def test_locomo_sample_is_ten_turns():
    sample = load_locomo_sample()
    assert sample["sample_id"] == "conv-26"
    assert len(sample["turns"]) == 10
    assert sample["turns"][0]["dia_id"] == "D1:1"
    assert sample["turns"][-1]["dia_id"] == "D1:10"


def test_fact_is_stored_without_a_search():
    stored = []

    async def store(memory):
        stored.append(memory)

    async def search(_query):
        raise AssertionError("search should not run")

    events = asyncio.run(
        _events(
            "I like Thai and Vietnamese food.",
            plan=lambda _text, _history: TurnPlan(
                False, None, True, "The user likes Thai and Vietnamese food."
            ),
            search=search,
            store=store,
            decide=lambda *_args: Decision(True),
            answer=lambda text, facts, history, just: f"noted: {just}",
        )
    )
    kinds = [event["type"] for event in events]
    assert "search" not in kinds
    assert stored == ["The user likes Thai and Vietnamese food."]
    assert events[-1] == {
        "type": "assistant",
        "text": "noted: The user likes Thai and Vietnamese food.",
    }


def test_question_searches_until_enough():
    queries = []

    async def search(query):
        queries.append(query)
        if query == "food":
            return [], 0.2
        return [Fact("e1", "The user likes Thai food.")], 0.3

    def decide(_question, facts, _prompt, _sent):
        if not facts:
            return Decision(enough=False, next_query="Thai food")
        return Decision(enough=True, answer="Thai")

    events = asyncio.run(
        _events(
            "Where should I go for food?",
            plan=lambda _text, _history: TurnPlan(True, "food", False, None),
            search=search,
            store=lambda _memory: _fail(),
            decide=decide,
            answer=lambda text, facts, history, just: facts[0].text,
        )
    )
    searches = [event for event in events if event["type"] == "search"]
    assert queries == ["food", "Thai food"]
    assert searches[0]["enough"] is False
    assert searches[1]["facts"][0]["text"] == "The user likes Thai food."
    assert events[-1]["text"] == "The user likes Thai food."


def test_ensure_indices_waits_for_the_driver_task():
    async def scenario():
        calls = []

        async def driver_build():
            calls.append("driver")

        class Graph:
            def __init__(self):
                self.driver = SimpleNamespace(_init_task=asyncio.create_task(driver_build()))

            async def build_indices_and_constraints(self):
                calls.append("again")

        graph = Graph()
        await ensure_indices(graph)
        await ensure_indices(graph)
        return calls

    assert asyncio.run(scenario()) == ["driver"]


def test_ensure_indices_builds_when_the_driver_did_not():
    async def scenario():
        calls = []

        class Graph:
            driver = SimpleNamespace()

            async def build_indices_and_constraints(self):
                calls.append("build")

        await ensure_indices(Graph())
        return calls

    assert asyncio.run(scenario()) == ["build"]


async def _fail():
    raise AssertionError("store should not run")


async def _events(text, **kwargs):
    out = []
    async for event in run_turn(text, **kwargs):
        out.append(event)
    return out
