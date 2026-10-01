from kgr_memory.react import Decision, Fact, ask_once, ask_react, parse_decision


def test_parse_decision_strips_fence():
    raw = '```json\n{"enough": false, "next_query": "city", "answer": null}\n```'
    decision = parse_decision(raw)
    assert decision.enough is False
    assert decision.next_query == "city"
    assert decision.answer is None


def test_react_stops_when_enough():
    calls = []

    def search(query):
        calls.append(query)
        return [Fact("e1", "Rodo lives in Brisbane")], 0.1

    def decide(question, facts, prompt_name, sent):
        return Decision(enough=True, answer="from the decision prompt")

    result = ask_react(
        "Where does Rodo live?",
        search,
        decide,
        lambda q, f: "Brisbane",
    )
    assert result.stopped_because == "enough"
    assert result.answer == "Brisbane"
    assert calls == ["Where does Rodo live?"]
    assert len(result.rounds) == 1


def test_oneshot_and_react_share_the_answer_step():
    def search(query):
        return [Fact("e1", "Brisbane")], 0.0

    def decide(question, facts, prompt_name, sent):
        return Decision(enough=True, answer="from the decision prompt")

    def answer(question, facts):
        return f"{question}:{facts[0].text}"

    once = ask_once("Where?", search, answer)
    react = ask_react("Where?", search, decide, answer, max_rounds=2)
    assert once.answer == react.answer == "Where?:Brisbane"


def test_react_stops_when_the_next_query_repeats():
    def search(query):
        return [Fact(query, f"fact for {query}")], 0.2

    def decide(question, facts, prompt_name, sent):
        if len(sent) == 1:
            return Decision(enough=False, next_query="follow up")
        return Decision(enough=False, next_query="follow up")

    result = ask_react(
        "q",
        search,
        decide,
        lambda q, f: "forced",
        max_rounds=3,
    )
    assert result.stopped_because == "no_new_query"
    assert [r.query for r in result.rounds] == ["q", "follow up"]
    assert result.answer == "forced"


def test_react_hits_cap():
    n = {"i": 0}

    def search(query):
        n["i"] += 1
        return [Fact(str(n["i"]), query)], 0.0

    def decide(question, facts, prompt_name, sent):
        return Decision(enough=False, next_query=f"q{len(sent)}")

    result = ask_react("start", search, decide, lambda q, f: "capped", max_rounds=3)
    assert result.stopped_because == "cap"
    assert len(result.rounds) == 3
    assert result.answer == "capped"


def test_oneshot_searches_once():
    calls = []

    def search(query):
        calls.append(query)
        return [Fact("1", "a fact")], 0.05

    result = ask_once("q", search, lambda q, facts: f"{len(facts)} facts")
    assert result.mode == "oneshot"
    assert result.answer == "1 facts"
    assert calls == ["q"]


def test_prompts_share_the_stop_contract():
    from kgr_memory.prompts import system_prompt

    texts = [system_prompt(name) for name in ("strict", "balanced", "loose")]
    contract = "If enough is true, set next_query to null"
    assert all(contract in text for text in texts)
    assert len(set(texts)) == 3


def test_unknown_prompt_rejected():
    import pytest

    from kgr_memory.prompts import system_prompt

    with pytest.raises(ValueError):
        system_prompt("nope")
