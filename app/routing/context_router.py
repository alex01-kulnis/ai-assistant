from __future__ import annotations

from app.routing.schemas import IntentResult
from app.schemas.chat import ChatRequest


def route_by_context(payload: ChatRequest) -> IntentResult | None:
    action = payload.action.strip().casefold() if payload.action else None

    if action == "summarize" and payload.selected_text:
        return IntentResult(intent="summarization", confidence=0.99, source="context")

    if payload.customer_id:
        return IntentResult(intent="customer_analysis", confidence=0.95, source="context")

    if payload.document_id:
        return IntentResult(intent="rag_question", confidence=0.95, source="context")

    return None

