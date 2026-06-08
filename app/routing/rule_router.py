from __future__ import annotations

from app.routing.schemas import IntentResult

UNSUPPORTED_KEYWORDS = (
    "удали все",
    "удалить все",
    "drop table",
    "delete all",
    "truncate",
    "очисти базу",
    "удали документы",
)

SUMMARIZATION_KEYWORDS = (
    "суммаризируй",
    "саммаризируй",
    "сделай summary",
    "сделай саммари",
    "кратко перескажи",
    "краткое содержание",
    "резюмируй",
)

GENERIC_CUSTOMER_KEYWORDS = (
    "клиент",
    "клиента",
    "customer",
)

STRONG_CUSTOMER_ANALYSIS_KEYWORDS = (
    "churn",
    "отток",
    "риск ухода",
    "риск оттока",
    "next-best-action",
    "следующее действие",
    "что ему предложить",
    "реактивация",
    "reactivation",
)

CUSTOMER_ANALYSIS_KEYWORDS = GENERIC_CUSTOMER_KEYWORDS + STRONG_CUSTOMER_ANALYSIS_KEYWORDS

RAG_KEYWORDS = (
    # "как",
    "что делать",
    "инструкция",
    "политика",
    "документ",
    "регламент",
    "возврат",
    "sla",
    "правила",
    "процедура",
)


def route_by_rules(question: str) -> IntentResult | None:
    normalized_question = question.strip().casefold()
    if not normalized_question:
        return None

    if _contains_any(normalized_question, UNSUPPORTED_KEYWORDS):
        return IntentResult(intent="unsupported", confidence=0.95, source="rules")

    if _contains_any(normalized_question, SUMMARIZATION_KEYWORDS):
        return IntentResult(intent="summarization", confidence=0.90, source="rules")

    if _contains_any(normalized_question, STRONG_CUSTOMER_ANALYSIS_KEYWORDS):
        return IntentResult(intent="customer_analysis", confidence=0.85, source="rules")

    if _contains_any(normalized_question, RAG_KEYWORDS):
        return IntentResult(intent="rag_question", confidence=0.80, source="rules")

    if _contains_any(normalized_question, GENERIC_CUSTOMER_KEYWORDS):
        return IntentResult(intent="customer_analysis", confidence=0.85, source="rules")

    return None


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)
