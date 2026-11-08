"""POST /extract, with the model dependency replaced so nothing leaves the machine."""

import json

import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import app, get_llm

EATING = "My cow has not been eating since yesterday and she is breathing very fast."


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, system: str, user: str) -> str:
        return self.content


class BrokenLLM:
    def complete(self, system: str, user: str) -> str:
        raise LLMError("The model request failed (503)")


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _use(llm) -> None:
    app.dependency_overrides[get_llm] = lambda: llm


def test_extract_returns_findings_and_does_not_escalate(client):
    _use(
        FakeLLM(
            json.dumps(
                {
                    "findings": [
                        {
                            "sign_id": "reduced_appetite",
                            "value": "yes",
                            "quote": "not been eating",
                        },
                        {
                            "sign_id": "rapid_breathing",
                            "value": "yes",
                            "quote": "breathing very fast",
                        },
                    ]
                }
            )
        )
    )
    response = client.post("/extract", json={"message": EATING})
    assert response.status_code == 200
    body = response.json()
    assert body["escalate"] is False
    assert body["reason"] == "No emergency signs reported."
    assert [item["sign_id"] for item in body["findings"]] == [
        "reduced_appetite",
        "rapid_breathing",
    ]


def test_extract_escalates_when_the_model_reports_foam(client):
    message = "There's foam coming from her nose and she can't get up."
    _use(
        FakeLLM(
            json.dumps(
                {
                    "findings": [
                        {
                            "sign_id": "froth_nose_mouth",
                            "value": "yes",
                            "quote": "foam coming from her nose",
                        },
                        {"sign_id": "unable_to_stand", "value": "yes", "quote": "can't get up"},
                    ]
                }
            )
        )
    )
    response = client.post("/extract", json={"message": message})
    assert response.status_code == 200
    body = response.json()
    assert body["escalate"] is True
    assert body["reason"] == (
        "Escalating because foam or froth from the nose or mouth was reported, "
        "and inability to stand was reported."
    )


def test_empty_and_huge_messages_are_rejected(client):
    assert client.post("/extract", json={"message": "   "}).status_code == 422
    assert client.post("/extract", json={"message": "x" * 4001}).status_code == 422
    assert client.post("/extract", json={}).status_code == 422


def test_unusable_model_json_is_a_bad_gateway(client):
    _use(FakeLLM("the cow is sick"))
    response = client.post("/extract", json={"message": EATING})
    assert response.status_code == 502
    assert response.json()["detail"] == "The model did not return usable JSON"


def test_model_transport_error_is_a_bad_gateway(client):
    _use(BrokenLLM())
    response = client.post("/extract", json={"message": EATING})
    assert response.status_code == 502
    assert "503" in response.json()["detail"]


def test_missing_api_key_fails_before_any_network_call(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    app.dependency_overrides.clear()
    response = client.post("/extract", json={"message": EATING})
    assert response.status_code == 502
    assert response.json()["detail"] == "LLM_API_KEY is not set"
