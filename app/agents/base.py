from __future__ import annotations

from abc import ABC, abstractmethod

from app.agents.state import AgentState


class BaseAgent(ABC):
    name: str

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        ...

