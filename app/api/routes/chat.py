from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.support_agent import SupportAgent, SupportAgentError
from app.db.session import get_db_session
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_support_agent() -> SupportAgent:
    return SupportAgent()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    agent: Annotated[SupportAgent, Depends(get_support_agent)],
) -> ChatResponse:
    try:
        return await agent.chat(
            conversation_id=request.conversation_id,
            message=request.message,
            session=session,
        )
    except SupportAgentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
