"""tests/test_main.py
Unit tests for the Context‑Aware Multilingual Phrasebook Generator API.

The tests cover:
- Basic health check endpoint.
- Successful phrase generation via the ``/generate`` endpoint.
- Validation error when required fields are missing.

The heavy external dependencies (Transformer models, JinaAI embeddings,
and SQLite) are mocked so the test suite runs quickly and deterministically.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# The FastAPI application instance is expected to be defined in ``app/main.py``.
# Adjust the import path if the project structure differs.
from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


def _assert_json_response(response: Any, expected_status: int, expected_keys: Dict[str, Any]) -> None:
    """
    Helper to assert a JSON response.

    Args:
        response: The ``requests.Response`` object returned by the TestClient.
        expected_status: Expected HTTP status code.
        expected_keys: Mapping of expected JSON keys to either a value or a callable
            that validates the value.
    """
    assert response.status_code == expected_status, f"Unexpected status code: {response.status_code}"
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise AssertionError("Response is not valid JSON") from exc

    for key, validator in expected_keys.items():
        assert key in data, f"Missing key '{key}' in response JSON"
        if callable(validator):
            assert validator(data[key]), f"Validation failed for key '{key}'"
        else:
            assert data[key] == validator, f"Expected {key} == {validator!r}, got {data[key]!r}"


def test_health_endpoint(client: TestClient) -> None:
    """The ``/health`` endpoint must return a simple JSON status payload."""
    response = client.get("/health")
    _assert_json_response(
        response,
        expected_status=200,
        expected_keys={"status": "ok"},
    )


def test_generate_phrase_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """
    ``POST /generate`` should return a translation when all required fields are present.

    The heavy generation logic is mocked to return a deterministic string.
    """
    # Mock the core generation function. The real implementation lives in ``app.generator``.
    def mock_generate_phrasebook(context: str, language: str, phrase: str) -> str:
        # Simple deterministic mock – echo the phrase with language appended.
        return f"{phrase} [{language}] (context: {context})"

    monkeypatch.setattr("app.generator.generate_phrasebook", mock_generate_phrasebook)

    payload = {
        "context": "travel",
        "language": "es",
        "phrase": "Where is the nearest train station?",
    }

    response = client.post("/generate", json=payload)

    _assert_json_response(
        response,
        expected_status=200,
        expected_keys={
            "translation": lambda v: v == "Where is the nearest train station? [es] (context: travel)"
        },
    )


def test_generate_phrase_validation_error(client: TestClient) -> None:
    """
    ``POST /generate`` must reject requests missing required fields with a 422 status.
    """
    # ``phrase`` field is omitted intentionally to trigger validation.
    incomplete_payload = {"context": "travel", "language": "es"}

    response = client.post("/generate", json=incomplete_payload)

    # FastAPI returns a validation error payload; we only assert the status code here.
    assert response.status_code == 422
    # Ensure the response contains the standard FastAPI validation error structure.
    error_body = response.json()
    assert "detail" in error_body
    assert any(err["loc"] == ["body", "phrase"] for err in error_body["detail"])
