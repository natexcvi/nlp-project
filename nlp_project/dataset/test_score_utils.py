from unittest.mock import MagicMock

import pytest

from nlp_project.dataset.score_utils import ScoreUtils, SemanticContainment


@pytest.fixture
def score_utils():
    openai_api_key = "test_api_key"
    score_utils_instance = ScoreUtils(openai_api_key=openai_api_key)
    score_utils_instance.openai_client = MagicMock()
    return score_utils_instance


@pytest.mark.parametrize(
    "target, text, found, extracted_match, embeddings, expected",
    [
        ("test", "This is a test text.", True, "test", [[1, 0, 0], [1, 0, 0]], 1.0),
        ("test", "This is a sample text.", False, "", [], 0),
        (
            "test",
            "This is a testing text.",
            True,
            "testing",
            [[0.9, 0.1, 0], [1, 0, 0]],
            lambda x: 0 < x < 1,
        ),
    ],
)
def test_contains_semantically(
    score_utils, target, text, found, extracted_match, embeddings, expected
):
    response_mock = MagicMock()
    response_mock.choices[0].message.parsed = SemanticContainment(
        found=found, extracted_match=extracted_match
    )
    score_utils.openai_client.beta.chat.completions.parse.return_value = response_mock
    if embeddings:
        embedding_response_mock = MagicMock()
        embedding_response_mock.data = []
        for embedding in embeddings:
            embedding_response_mock.data.append(MagicMock(embedding=embedding))
        score_utils.openai_client.embeddings.create.return_value = (
            embedding_response_mock
        )
    result = score_utils.contains_semantically(target, text)
    if callable(expected):
        assert expected(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 2", "4"),
        ("2 * 2", "4"),
        ("2 - 2", "0"),
        ("2 / 2", "1"),
        ("2 + 2 * 2", "6"),
        ("(2 + 2) * 2", "8"),
        ("(2 + 2) * (2 + 2)", "16"),
    ],
)
def test_simplify_math(score_utils, expression, expected):
    result = score_utils.simplify_math(expression)
    assert result == expected
