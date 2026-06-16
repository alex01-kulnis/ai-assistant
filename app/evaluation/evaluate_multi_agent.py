from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from app.agents.base import BaseAgent
from app.agents.customer_analysis_agent import CustomerAnalysisAgent
from app.agents.state import AgentState
from app.agents.summarization_agent import SummarizationAgent
from app.agents.supervisor import SupervisorAgent
from app.routing.classifier import classify_intent
from app.schemas.chat import ChatRequest

DATASET_PATH = Path(__file__).with_name("multi_agent_dataset.jsonl")


class EvaluationRagAgent(BaseAgent):
    name = "rag_agent"

    async def run(self, state: AgentState) -> AgentState:
        state.current_agent = self.name
        state.selected_agent = self.name
        state.answer = "Evaluation RAG answer"
        state.sources = [
            {
                "document_id": "eval-doc",
                "filename": "eval.txt",
                "page_number": None,
                "chunk_index": 0,
                "score": 1.0,
            }
        ]
        state.add_trace_step(self.name, "evaluation_stub_completed")
        return state


class EvaluationLLMService:
    model_name = "evaluation-fake-llm"

    async def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        return (
            "1. Краткий статус клиента: evaluation stub.\n"
            "2. Риск / сигналы: rule-based baseline.\n"
            "3. Рекомендуемое действие: проверить следующий шаг.\n"
            "4. Почему это действие: основано на mock-сигналах.\n"
            "5. Ограничения / что проверить перед запуском: это не реальная ML-модель."
        )


async def main() -> None:
    examples = _load_dataset(DATASET_PATH)
    failures: list[dict[str, Any]] = []
    correct_intent = 0
    correct_agent = 0
    needs_human_review_count = 0
    validation_error_count = 0

    llm_service = EvaluationLLMService()
    supervisor = SupervisorAgent(
        rag_agent=EvaluationRagAgent(),
        summarization_agent=SummarizationAgent(llm_service=llm_service),  # type: ignore[arg-type]
        customer_analysis_agent=CustomerAnalysisAgent(llm_service=llm_service),  # type: ignore[arg-type]
    )

    for index, example in enumerate(examples, start=1):
        request = ChatRequest(**example["request"])
        intent_result = await classify_intent(request, llm_service=llm_service)  # type: ignore[arg-type]
        state = AgentState(
            request_id=str(uuid.uuid4()),
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            message_text=request.message_text,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            router_source=intent_result.source,
            customer_id=request.customer_id,
            ticket_id=request.ticket_id,
            document_id=request.document_id,
            action=request.action,
            selected_text=request.selected_text,
        )
        state = await supervisor.run(state)

        expected_intent = example["expected_intent"]
        expected_agent = example["expected_agent"]
        intent_matches = intent_result.intent == expected_intent
        agent_matches = state.selected_agent == expected_agent
        correct_intent += int(intent_matches)
        correct_agent += int(agent_matches)
        needs_human_review_count += int(state.needs_human_review)
        validation_error_count += len(state.validation_errors)

        if not intent_matches or not agent_matches:
            failures.append(
                {
                    "index": index,
                    "question": request.message_text,
                    "expected_intent": expected_intent,
                    "actual_intent": intent_result.intent,
                    "expected_agent": expected_agent,
                    "actual_agent": state.selected_agent,
                    "validation_errors": state.validation_errors,
                }
            )

    total = len(examples)
    report = {
        "total": total,
        "correct_intent": correct_intent,
        "correct_agent": correct_agent,
        "intent_accuracy": _accuracy(correct_intent, total),
        "agent_selection_accuracy": _accuracy(correct_agent, total),
        "needs_human_review_count": needs_human_review_count,
        "validation_error_count": validation_error_count,
        "failed_cases": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                examples.append(json.loads(stripped_line))
    return examples


def _accuracy(correct: int, examples: int) -> float:
    return round(correct / examples, 4) if examples else 0.0


if __name__ == "__main__":
    asyncio.run(main())
