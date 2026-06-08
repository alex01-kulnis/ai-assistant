from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "rag_question",
    "summarization",
    "customer_analysis",
    "unsupported",
]
RouterSource = Literal["context", "rules", "llm", "embeddings", "fallback"]


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    source: RouterSource

