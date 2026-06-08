from __future__ import annotations

from app.routing.context_router import route_by_context
from app.routing.llm_router import route_by_llm
from app.routing.rule_router import route_by_rules
from app.routing.schemas import IntentResult
from app.schemas.chat import ChatRequest
from app.services.llm_service import OllamaLLMService


async def classify_intent(
    payload: ChatRequest,
    llm_service: OllamaLLMService | None = None,
) -> IntentResult:
    context_result = route_by_context(payload)
    if context_result and context_result.confidence >= 0.95:
        return context_result

    rule_result = route_by_rules(payload.message_text)
    if rule_result and _is_confident_rule_result(rule_result):
        return rule_result

    llm_result = await route_by_llm(payload.message_text, llm_service=llm_service)
    if llm_result.confidence >= 0.65:
        return llm_result

    return IntentResult(intent="unsupported", confidence=0.0, source="fallback")


def _is_confident_rule_result(result: IntentResult) -> bool:
    if result.confidence >= 0.85:
        return True
    return result.intent == "rag_question" and result.confidence >= 0.80

