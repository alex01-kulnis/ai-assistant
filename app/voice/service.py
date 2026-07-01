from __future__ import annotations

import logging
from pathlib import Path

from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tracing import set_span_attributes
from app.schemas.chat import ChatRequest
from app.services.chat_workflow_service import ChatWorkflowService
from app.voice.audio_converter import AudioConverter
from app.voice.schemas import VoiceChatResponse, VoiceTranscriptionResult
from app.voice.speech_to_text import LocalSpeechToTextService

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class EmptyTranscriptError(RuntimeError):
    pass


class VoiceService:
    def __init__(
        self,
        *,
        audio_converter: AudioConverter | None = None,
        speech_to_text: LocalSpeechToTextService | None = None,
        convert_to_wav: bool | None = None,
        keep_audio_files: bool | None = None,
    ) -> None:
        from app.core.config import get_settings

        settings = get_settings()
        self.audio_converter = audio_converter or AudioConverter()
        self.speech_to_text = speech_to_text or LocalSpeechToTextService()
        self.convert_to_wav = (
            settings.VOICE_CONVERT_TO_WAV if convert_to_wav is None else convert_to_wav
        )
        self.keep_audio_files = (
            settings.VOICE_KEEP_AUDIO_FILES if keep_audio_files is None else keep_audio_files
        )

    def transcribe_audio(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        cleanup: bool = True,
    ) -> VoiceTranscriptionResult:
        converted_path: Path | None = None
        try:
            wav_path = audio_path
            if self.convert_to_wav:
                with tracer.start_as_current_span("voice.convert_audio") as span:
                    converted_path = self.audio_converter.convert_to_wav(audio_path)
                    wav_path = converted_path
                    set_span_attributes(
                        span,
                        {
                            "original_audio_path": str(audio_path),
                            "converted_wav_path": str(wav_path),
                        },
                    )
            logger.info(
                "voice file prepared for transcription",
                extra={
                    "original_audio_path": str(audio_path),
                    "converted_wav_path": str(wav_path),
                },
            )
            with tracer.start_as_current_span("voice.transcribe") as span:
                result = self.speech_to_text.transcribe(wav_path, language=language)
                set_span_attributes(
                    span,
                    {
                        "stt_provider": result.stt_provider,
                        "stt_model": result.stt_model,
                        "stt_latency_ms": result.latency_ms,
                        "transcript_length": len(result.text),
                    },
                )
            logger.info(
                "voice transcription completed",
                extra={
                    "stt_provider": result.stt_provider,
                    "stt_model": result.stt_model,
                    "stt_latency_ms": result.latency_ms,
                    "transcript_length": len(result.text),
                },
            )
            return result
        finally:
            if cleanup and not self.keep_audio_files:
                self._cleanup_paths(audio_path, converted_path)

    async def handle_voice_message(
        self,
        *,
        audio_path: Path,
        session: AsyncSession,
        chat_workflow: ChatWorkflowService,
        user_id: str | None = None,
        conversation_id: str | None = None,
        language: str | None = None,
        cleanup: bool = True,
    ) -> VoiceChatResponse:
        with tracer.start_as_current_span("voice.chat") as span:
            transcription = self.transcribe_audio(audio_path, language=language, cleanup=cleanup)
            if not transcription.text:
                set_span_attributes(span, {"voice_chat_status": "transcription_failed"})
                return VoiceChatResponse(
                    transcript="",
                    status="transcription_failed",
                    answer=None,
                    stt_provider=transcription.stt_provider,
                    stt_model=transcription.stt_model,
                    stt_latency_ms=transcription.latency_ms,
                )

            input_audio_path = str(audio_path) if self.keep_audio_files else None
            chat_response = await chat_workflow.process(
                request=ChatRequest(
                    conversation_id=conversation_id,
                    message=transcription.text,
                    user_id=user_id,
                    input_mode="voice",
                    input_audio_path=input_audio_path,
                    input_transcript=transcription.text,
                    stt_provider=transcription.stt_provider,
                    stt_model=transcription.stt_model,
                    stt_latency_ms=transcription.latency_ms,
                ),
                session=session,
            )
            status = "needs_human_review" if chat_response.needs_human_review else "success"
            set_span_attributes(
                span,
                {
                    "voice_chat_status": status,
                    "agent_run_id": chat_response.agent_run_id,
                    "transcript_length": len(transcription.text),
                },
            )
            logger.info(
                "voice chat completed",
                extra={
                    "status": status,
                    "agent_run_id": chat_response.agent_run_id,
                    "transcript_length": len(transcription.text),
                },
            )
            return VoiceChatResponse(
                transcript=transcription.text,
                status=status,
                answer=chat_response.answer,
                agent_run_id=chat_response.agent_run_id,
                review_reason=_build_review_reason(chat_response.validation_errors, status),
                stt_provider=transcription.stt_provider,
                stt_model=transcription.stt_model,
                stt_latency_ms=transcription.latency_ms,
            )

    def _cleanup_paths(self, *paths: Path | None) -> None:
        for path in paths:
            if path is not None and path.exists():
                try:
                    path.unlink()
                except OSError:
                    logger.warning(
                        "Failed to delete temporary audio file",
                        extra={"path": str(path)},
                    )


def _build_review_reason(validation_errors: list[str] | None, status: str) -> str | None:
    if status != "needs_human_review":
        return None
    if validation_errors:
        return validation_errors[0]
    return "high_risk_action"
