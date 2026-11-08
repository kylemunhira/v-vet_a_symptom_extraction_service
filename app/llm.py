"""OpenAI-compatible chat client. Tests pass a fake or an httpx mock transport."""

from __future__ import annotations

import json
from typing import Protocol

import httpx


class LLMError(Exception):
    """The model could not be called, or its response had no text."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text for one system/user turn."""
        ...


class OpenAICompatibleClient:
    """POST {base_url}/chat/completions. Works with OpenAI, Groq, OpenRouter, or Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        json_mode: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.json_mode = json_mode
        self.transport = transport

    def complete(self, system: str, user: str) -> str:
        if not self.api_key.strip():
            raise LLMError("LLM_API_KEY is not set")

        payload: dict = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
                trust_env=False,
            ) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            raise LLMError("The model request failed before a response arrived") from exc

        if response.status_code >= 400:
            reason = _provider_error(response)
            suffix = f": {reason}" if reason else ""
            raise LLMError(f"The model request failed ({response.status_code}){suffix}")

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("The model response did not include text") from exc

        text = _content_to_text(content)
        if not text.strip():
            raise LLMError("The model response was empty")
        return text


def _provider_error(response: httpx.Response) -> str:
    try:
        error = response.json().get("error")
    except (json.JSONDecodeError, AttributeError):
        return ""
    if isinstance(error, dict):
        error = error.get("message")
    return str(error)[:300] if error else ""


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise LLMError("The model response did not include text")
