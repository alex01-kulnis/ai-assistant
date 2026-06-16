from __future__ import annotations

import json
import logging

from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.support_agent import SupportAgentError
from app.services.llm_service import OllamaLLMError, OllamaLLMService
from app.tools.customer_tools import (
    calculate_churn_score,
    get_mock_customer_profile,
    recommend_next_best_action,
)

logger = logging.getLogger(__name__)


class CustomerAnalysisAgent(BaseAgent):
    name = "customer_analysis_agent"

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
                "customer_id": state.customer_id,
            },
        )
        state.current_agent = self.name
        state.selected_agent = self.name
        state.add_trace_step(self.name, "started")

        profile = await get_mock_customer_profile(state.customer_id)
        state.add_trace_step(self.name, "get_customer_profile_called")

        churn_score = calculate_churn_score(profile)
        state.add_trace_step(self.name, "calculate_churn_score_called")

        next_best_action = recommend_next_best_action(profile, churn_score)
        state.add_trace_step(self.name, "recommend_next_best_action_called")

        state.tool_results = {
            "customer_profile": profile,
            "churn_score": churn_score,
            "next_best_action": next_best_action,
        }

        if profile.get("error") == "missing_customer_id":
            state.answer = (
                "Для точного анализа клиента нужен customer_id. "
                "Без него можно описать только общий подход: собрать профиль клиента, "
                "проверить частоту покупок, недавние обращения, активность и жалобы, "
                "после чего выбрать удерживающее действие. Текущая оценка в MVP является "
                "rule-based baseline, а не ML-моделью."
            )
            state.add_trace_step(self.name, "answer_generated", {"used_llm": False})
            return state

        try:
            state.answer = (
                await self.llm_service.generate_chat_response(
                    self._build_llm_messages(
                        profile=profile,
                        churn_score=churn_score,
                        next_best_action=next_best_action,
                    ),
                    temperature=0.2,
                )
            ).strip()
        except OllamaLLMError as exc:
            raise SupportAgentError(f"LLM request failed: {exc}", status_code=502) from exc

        state.add_trace_step(self.name, "answer_generated", {"used_llm": True})
        logger.info(
            "agent_completed",
            extra={
                "request_id": state.request_id,
                "conversation_id": state.conversation_id,
                "intent": state.intent,
                "selected_agent": self.name,
                "customer_id": state.customer_id,
                "risk_level": churn_score.get("risk_level"),
            },
        )
        return state

    def _build_llm_messages(
        self,
        *,
        profile: dict,
        churn_score: dict,
        next_best_action: dict,
    ) -> list[dict[str, str]]:
        data = {
            "customer_profile": profile,
            "churn_score": churn_score,
            "next_best_action": next_best_action,
        }
        return [
            {
                "role": "system",
                "content": (
                    "Ты CRM/support analyst. Сформируй понятный бизнес-ответ на русском. "
                    "Не делай вид, что это реальная ML-модель: явно укажи, что оценка "
                    "является rule-based baseline / предварительной оценкой."
                ),
            },
            {
                "role": "user",
                "content": (
                    "На основе mock CRM данных подготовь ответ в структуре:\n"
                    "1. Краткий статус клиента\n"
                    "2. Риск / сигналы\n"
                    "3. Рекомендуемое действие\n"
                    "4. Почему это действие\n"
                    "5. Ограничения / что проверить перед запуском\n\n"
                    f"DATA:\n{json.dumps(data, ensure_ascii=False)}"
                ),
            },
        ]

