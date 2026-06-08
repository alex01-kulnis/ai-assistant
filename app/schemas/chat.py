from __future__ import annotations

from typing import Self

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
    intent: str | None = None
    intent_confidence: float | None = None
    router_source: str | None = None
