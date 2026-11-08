"""The HTTP client, against an httpx mock transport. No socket is opened."""

import json

import httpx
import pytest

from app.llm import LLMError, OpenAICompatibleClient


def _client(handler, **kwargs) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://llm.test/v1",
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_complete_posts_chat_completions_and_returns_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"findings": []}'}}]},
        )

    text = _client(handler).complete("system text", "user text")
    assert text == '{"findings": []}'
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]


def test_json_mode_can_be_turned_off():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "response_format" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    assert _client(handler, json_mode=False).complete("s", "u") == "{}"


def test_content_parts_are_joined():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": '{"findings":'},
                                {"type": "text", "text": " []}"},
                            ]
                        }
                    }
                ]
            },
        )

    assert _client(handler).complete("s", "u") == '{"findings": []}'


def test_missing_api_key_does_not_call_the_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be called")

    client = OpenAICompatibleClient(
        base_url="https://llm.test/v1",
        api_key="  ",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        client.complete("s", "u")


def test_http_error_and_odd_payloads_raise():
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "nope"})

    with pytest.raises(LLMError, match="500"):
        _client(down).complete("s", "u")

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    with pytest.raises(LLMError, match="empty"):
        _client(empty).complete("s", "u")

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(LLMError, match="did not include text"):
        _client(broken).complete("s", "u")
