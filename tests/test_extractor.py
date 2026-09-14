import json
from types import SimpleNamespace
from unittest.mock import patch

from kgr_memory.extractor import OpenAIExtractor, Triple


def _completion(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload))
            )
        ]
    )


@patch("kgr_memory.extractor.OpenAI")
def test_extract_keeps_grounded_and_negated_triples(mock_openai) -> None:
    mock_openai.return_value.chat.completions.create.return_value = _completion(
        {
            "triples": [
                {"subject": "user", "predicate": "lives_in", "object": "Brisbane"},
                {"subject": "user", "predicate": "not_lives_in", "object": "Sydney"},
                {"subject": "", "predicate": "loves", "object": "him"},
            ]
        }
    )
    extractor = OpenAIExtractor(api_key="test")
    triples = extractor.extract("I live in Brisbane, not Sydney.", speaker="user")
    assert triples == [
        Triple("user", "lives_in", "Brisbane"),
        Triple("user", "not_lives_in", "Sydney"),
    ]
