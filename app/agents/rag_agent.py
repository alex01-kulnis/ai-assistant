from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.support_agent import SupportAgent

logger = logging.getLogger(__name__)


class RagAgent(BaseAgent):
    name = "rag_agent"

    def __init__(self, support_agent: SupportAgent, session: AsyncSession) -> None:
        self.support_agent = support_agent
        self.session = session

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

        response = await self.support_agent.chat(
            conversation_id=state.conversation_id,
            message=state.message_text,
            session=self.session,
        )
        state.conversation_id = response.conversation_id
        state.message_id = response.message_id
        state.answer = response.answer
        state.sources = list(response.sources)
        state.add_trace_step(
            self.name,
            "retrieval_generation_completed",
            {"sources_count": len(state.sources)},
        )

        logger.info(
            "agent_completed",
            extra={
                "request_id": state.request_id,
                "conversation_id": state.conversation_id,
                "message_id": state.message_id,
                "intent": state.intent,
                "selected_agent": self.name,
                "sources_count": len(state.sources),
            },
        )
        return state
