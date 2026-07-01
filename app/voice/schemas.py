from __future__ import annotations

from pydantic import BaseModel


class VoiceTranscriptionResult(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    latency_ms: int
    stt_provider: str
    stt_model: str


class VoiceChatResponse(BaseModel):
    transcript: str
    status: str
    answer: str | None = None
    agent_run_id: str | None = None
    review_reason: str | None = None
    stt_provider: str
    stt_model: str
    stt_latency_ms: int

