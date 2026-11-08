"""FastAPI app. POST /extract is the only route."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from app.extract import ParseError, extract_symptoms
from app.llm import LLMClient, LLMError, OpenAICompatibleClient
from app.schemas import ExtractRequest, ExtractResponse

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env without overriding variables already set."""
    env_path = path or ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

app = FastAPI(
    title="V-Vet symptom extraction",
    summary="Turn a farmer's WhatsApp message into a fixed list of clinical signs.",
    version="0.1.0",
)


def get_llm() -> LLMClient:
    json_mode = os.environ.get("LLM_JSON_MODE", "true").strip().lower() not in {"0", "false", "no"}
    timeout_raw = os.environ.get("LLM_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 30.0
    return OpenAICompatibleClient(
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        timeout=timeout,
        json_mode=json_mode,
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(body: ExtractRequest, llm: LLMClient = Depends(get_llm)) -> ExtractResponse:
    try:
        return extract_symptoms(body.message, llm)
    except (LLMError, ParseError) as exc:
        # 502, not an empty all-clear. A caller that cannot read the message should retry
        # or escalate, rather than tell the farmer nothing is wrong.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
