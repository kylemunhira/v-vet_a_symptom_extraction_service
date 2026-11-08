"""Request and response models for POST /extract."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_CHARS = 4000


class ExtractRequest(BaseModel):
    message: str = Field(examples=["There's foam coming from her nose and she can't get up."])

    @field_validator("message")
    @classmethod
    def message_must_have_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message is empty")
        if len(value) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message is longer than {MAX_MESSAGE_CHARS} characters")
        return value


class Finding(BaseModel):
    sign_id: str
    value: Literal["yes", "no", "not_sure"]
    quote: str


class ExtractResponse(BaseModel):
    findings: list[Finding]
    escalate: bool
    reason: str
