from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal[
    "supervisor",
    "rag_agent",
    "summarization_agent",
    "customer_analysis_agent",
    "guardrails_agent",
    "fallback",
]


class AgentResult(BaseModel):
    answer: str
    sources: list[Any] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    selected_agent: str | None = None

