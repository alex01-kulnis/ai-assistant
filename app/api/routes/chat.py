from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.support_agent import SupportAgent, SupportAgentError
from app.db.session import get_db_session
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_workflow_service import ChatWorkflowService
from app.services.llm_service import OllamaLLMService

router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_support_agent() -> SupportAgent:
    return SupportAgent()


def get_llm_service() -> OllamaLLMService:
    return OllamaLLMService()


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    agent: Annotated[SupportAgent, Depends(get_support_agent)],
    llm_service: Annotated[OllamaLLMService, Depends(get_llm_service)],
) -> ChatResponse:
    workflow = ChatWorkflowService(support_agent=agent, llm_service=llm_service)
    try:
        return await workflow.process(request=request, session=session)
    except SupportAgentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
