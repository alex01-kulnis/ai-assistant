from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.services.llm_service import OllamaLLMService

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


def get_llm_service() -> OllamaLLMService:
    return OllamaLLMService()


@router.get("/health")
async def llm_health(
    service: Annotated[OllamaLLMService, Depends(get_llm_service)],
) -> dict[str, str | bool]:
    is_available = await service.health_check()
    return {
        "provider": "ollama",
        "model": service.model_name,
        "available": is_available,
        "status": "ok" if is_available else "unavailable",
    }
