from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.routing.schemas import IntentResult
from app.services.llm_service import OllamaLLMService

logger = logging.getLogger(__name__)

_ALLOWED_INTENTS = {
    "rag_question",
    "summarization",
    "customer_analysis",
    "unsupported",
}

_SYSTEM_PROMPT = (
    "You are an intent router for a support AI application. "
    "Return only valid JSON and no markdown."
)

_USER_PROMPT_TEMPLATE = """Classify the user request into exactly one intent.

Allowed intents:
- rag_question: user asks a question that should be answered from the knowledge base
- summarization: user asks to summarize selected or provided text
- customer_analysis: user asks to analyze a customer profile, churn risk, or next best action
- unsupported: destructive, unsafe, unclear, or unsupported requests

Return strict JSON:
{{
  "intent": "rag_question|summarization|customer_analysis|unsupported",
  "confidence": 0.0
}}

User request:
{question}
"""


async def route_by_llm(
    question: str,
    llm_service: OllamaLLMService | None = None,
) -> IntentResult:
    service = llm_service or OllamaLLMService()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_PROMPT_TEMPLATE.format(question=question.strip()),
        },
    ]

    try:
        response = await service.generate_chat_response(messages, temperature=0.0)
        payload = _parse_json_response(response)
        intent = payload.get("intent")
        confidence = payload.get("confidence")
        if intent not in _ALLOWED_INTENTS:
            return _fallback()
        return IntentResult(intent=intent, confidence=float(confidence), source="llm")
    except (TypeError, ValueError, ValidationError):
        logger.info("LLM router returned invalid response", exc_info=True)
        return _fallback()
    except Exception:
        logger.exception("LLM router failed")
        return _fallback()


def _parse_json_response(response: str) -> dict[str, Any]:
    content = response.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("LLM router response must be a JSON object.")
    return payload


def _fallback() -> IntentResult:
    return IntentResult(intent="unsupported", confidence=0.0, source="fallback")

