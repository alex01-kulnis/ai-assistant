from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str | None = Field(default=None, min_length=1)
    question: str | None = Field(default=None, min_length=1)
    user_id: str | None = None
    customer_id: str | None = None
    ticket_id: str | None = None
    document_id: str | None = None
    action: str | None = None
    selected_text: str | None = None
    debug: bool = False
    input_mode: str | None = None
    input_audio_path: str | None = None
    input_transcript: str | None = None
    stt_provider: str | None = None
    stt_model: str | None = None
    stt_latency_ms: int | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_message_text_field(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "message_text" not in data:
            return data
        if data.get("message") or data.get("question"):
            return data
        normalized_data = dict(data)
        normalized_data["question"] = normalized_data["message_text"]
        return normalized_data

    @model_validator(mode="after")
    def validate_message_or_question(self) -> Self:
        if self.message is not None:
            self.message = self.message.strip()
        if self.question is not None:
            self.question = self.question.strip()
        if not self.message and not self.question:
            raise ValueError("message or question must not be empty")
        return self

    @property
    def message_text(self) -> str:
        return self.question or self.message or ""


class ChatSource(BaseModel):
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    sources: list[ChatSource]
    agent_run_id: str | None = None
    intent: str | None = None
    intent_confidence: float | None = None
    router_source: str | None = None
    selected_agent: str | None = None
    validation_errors: list[str] | None = None
    needs_human_review: bool | None = None
    trace: list[dict[str, Any]] | None = None
