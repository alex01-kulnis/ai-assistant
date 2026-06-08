from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.langchain_support_agent import (
    LangChainSupportAgent,
    get_langchain_support_agent,
)
from app.agents.support_agent import SupportAgentError
from app.db.session import get_db_session
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat/langchain", response_model=ChatResponse, response_model_exclude_none=True)
async def chat_langchain(
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    agent: Annotated[LangChainSupportAgent, Depends(get_langchain_support_agent)],
) -> ChatResponse:
    try:
        return await agent.chat(
            conversation_id=request.conversation_id,
            message=request.message,
            session=session,
        )
    except SupportAgentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
