from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    request_id: str
    conversation_id: str | None = None
    message_id: str | None = None
    user_id: str | None = None
    message_text: str
    intent: str
    intent_confidence: float
    router_source: str
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
    current_agent: str | None = None
    selected_agent: str | None = None
    agent_run_id: str | None = None
    answer: str | None = None
    sources: list[Any] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    trace: list[dict[str, Any]] = Field(default_factory=list)

    def add_trace_step(
        self,
        agent: str,
        action: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            {
                "agent": agent,
                "action": action,
                "data": data or {},
            }
        )
