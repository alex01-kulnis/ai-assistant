from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.support_agent import SupportAgentError
from app.services.llm_service import OllamaLLMError, OllamaLLMService
from app.tools.agent_tools import INSUFFICIENT_SUMMARY_TEXT_MESSAGE

logger = logging.getLogger(__name__)


class SummarizationAgent(BaseAgent):
    name = "summarization_agent"

    def __init__(self, llm_service: OllamaLLMService) -> None:
        self.llm_service = llm_service

    async def run(self, state: AgentState) -> AgentState:
        logger.info(
            "agent_started",
            extra={
                "request_id": state.request_id,
                "conversation_id": state.conversation_id,
                "intent": state.intent,
                "selected_agent": self.name,
            },
        )
        state.current_agent = self.name
        state.selected_agent = self.name
        state.add_trace_step(self.name, "started")

        source_text, source = self._select_source_text(state)
        state.add_trace_step(self.name, "source_selected", {"source": source})
        if source_text is None:
            state.answer = INSUFFICIENT_SUMMARY_TEXT_MESSAGE
            state.add_trace_step(self.name, "completed", {"used_llm": False})
            return state

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты support assistant. Суммаризируй только предоставленный текст. "
                    "Ответ должен быть коротким, структурированным и на русском языке."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Сделай краткую выжимку в структуре:\n"
                    "1. Суть обращения\n"
                    "2. Основная проблема\n"
                    "3. Важные детали\n"
                    "4. Рекомендуемый следующий шаг\n\n"
                    f"TEXT:\n{source_text}"
                ),
            },
        ]
        try:
            state.answer = (
                await self.llm_service.generate_chat_response(messages, temperature=0.1)
            ).strip()
        except OllamaLLMError as exc:
            raise SupportAgentError(f"LLM request failed: {exc}", status_code=502) from exc

        state.add_trace_step(self.name, "completed", {"used_llm": True})
        logger.info(
            "agent_completed",
            extra={
                "request_id": state.request_id,
                "conversation_id": state.conversation_id,
                "intent": state.intent,
                "selected_agent": self.name,
            },
        )
        return state

    def _select_source_text(self, state: AgentState) -> tuple[str | None, str]:
        if state.selected_text and state.selected_text.strip():
            return state.selected_text.strip(), "selected_text"

        message_text = state.message_text.strip()
        if len(message_text.split()) >= 8:
            return message_text, "message_text"

        return None, "insufficient_text"

