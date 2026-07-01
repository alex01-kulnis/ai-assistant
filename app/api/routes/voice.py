from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.support_agent import SupportAgent
from app.api.routes.chat import get_llm_service, get_support_agent
from app.core.config import Settings, get_settings
from app.core.tracing import set_span_attributes
from app.db.session import get_db_session
from app.services.chat_workflow_service import ChatWorkflowService
from app.services.llm_service import OllamaLLMService
from app.voice.audio_converter import SUPPORTED_AUDIO_EXTENSIONS, AudioConversionError
from app.voice.schemas import VoiceChatResponse, VoiceTranscriptionResult
from app.voice.service import VoiceService
from app.voice.speech_to_text import SpeechToTextError

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

SUPPORTED_AUDIO_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/mp4",
    "audio/m4a",
    "audio/ogg",
    "audio/opus",
    "application/ogg",
    "application/octet-stream",
}

CONTENT_TYPE_EXTENSION_FALLBACKS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/mp4": ".mp4",
    "audio/m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "application/ogg": ".ogg",
    "application/octet-stream": ".ogg",
}


def get_voice_service() -> VoiceService:
    return VoiceService()


@router.post("/transcribe", response_model=VoiceTranscriptionResult)
async def transcribe_voice(
    file: Annotated[UploadFile, File()],
    voice_service: Annotated[VoiceService, Depends(get_voice_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VoiceTranscriptionResult:
    _ensure_voice_enabled(settings)
    audio_path = await _save_upload(file=file, settings=settings)
    try:
        return voice_service.transcribe_audio(audio_path, cleanup=True)
    except AudioConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SpeechToTextError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    support_agent: Annotated[SupportAgent, Depends(get_support_agent)],
    llm_service: Annotated[OllamaLLMService, Depends(get_llm_service)],
    voice_service: Annotated[VoiceService, Depends(get_voice_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VoiceChatResponse:
    _ensure_voice_enabled(settings)
    audio_path = await _save_upload(file=file, settings=settings)
    chat_workflow = ChatWorkflowService(
        support_agent=support_agent,
        llm_service=llm_service,
    )
    try:
        return await voice_service.handle_voice_message(
            audio_path=audio_path,
            session=session,
            chat_workflow=chat_workflow,
            cleanup=True,
        )
    except AudioConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SpeechToTextError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _save_upload(file: UploadFile, settings: Settings) -> Path:
    content_type = file.content_type or "application/octet-stream"
    suffix = Path(file.filename or "").suffix.casefold()
    if not _is_supported_audio(content_type=content_type, suffix=suffix):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    with tracer.start_as_current_span("voice.save_upload") as span:
        content = await file.read()
        max_size_bytes = settings.VOICE_MAX_AUDIO_SIZE_MB * 1024 * 1024
        set_span_attributes(
            span,
            {
                "content_type": content_type,
                "file_size": len(content),
                "max_size_bytes": max_size_bytes,
            },
        )
        if len(content) > max_size_bytes:
            raise HTTPException(status_code=413, detail="Audio file is too large.")

        tmp_dir = Path(settings.VOICE_AUDIO_TMP_DIR)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        output_suffix = suffix or CONTENT_TYPE_EXTENSION_FALLBACKS.get(content_type, ".ogg")
        output_path = tmp_dir / f"upload-{uuid.uuid4().hex}{output_suffix}"
        output_path.write_bytes(content)
        logger.info(
            "voice file received",
            extra={
                "file_size": len(content),
                "content_type": content_type,
                "original_audio_path": str(output_path),
            },
        )
        return output_path


def _ensure_voice_enabled(settings: Settings) -> None:
    if not settings.VOICE_ENABLED:
        raise HTTPException(status_code=503, detail="Voice input is disabled.")


def _is_supported_audio(*, content_type: str, suffix: str) -> bool:
    return content_type in SUPPORTED_AUDIO_CONTENT_TYPES or suffix in SUPPORTED_AUDIO_EXTENSIONS
