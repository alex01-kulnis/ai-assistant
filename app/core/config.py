from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="SupportOps AI Agent", validation_alias="APP_NAME")
    app_env: str = Field(default="local", validation_alias="APP_ENV")
    debug: bool = Field(default=True, validation_alias="DEBUG")

    api_prefix: str = "/api/v1"

    postgres_user: str = Field(default="supportops", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(
        default="supportops",
        validation_alias="POSTGRES_PASSWORD",
    )
    postgres_db: str = Field(default="supportops", validation_alias="POSTGRES_DB")
    database_url: str = Field(
        default="postgresql+asyncpg://supportops:supportops@localhost:5433/supportops",
        validation_alias="DATABASE_URL",
    )

    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_collection_name: str = Field(
        default="support_knowledge_base",
        validation_alias="QDRANT_COLLECTION_NAME",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="qwen2.5:7b", validation_alias="OLLAMA_MODEL")
    embedding_model_name: str = Field(
        default="intfloat/multilingual-e5-small",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    telegram_bot_token: str | None = Field(default=None, validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_user_ids: str | None = Field(
        default=None,
        validation_alias="TELEGRAM_ALLOWED_USER_IDS",
    )
    telegram_use_backend_http: bool = Field(
        default=False,
        validation_alias="TELEGRAM_USE_BACKEND_HTTP",
    )
    telegram_backend_chat_url: str = Field(
        default="http://localhost:8000/api/v1/chat",
        validation_alias="TELEGRAM_BACKEND_CHAT_URL",
    )
    telegram_enabled: bool = Field(default=False, validation_alias="TELEGRAM_ENABLED")
    telegram_webhook_secret: str | None = Field(
        default=None,
        validation_alias="TELEGRAM_WEBHOOK_SECRET",
    )
    telegram_api_base_url: str = Field(
        default="https://api.telegram.org",
        validation_alias="TELEGRAM_API_BASE_URL",
    )
    telegram_file_base_url: str = Field(
        default="https://api.telegram.org/file",
        validation_alias="TELEGRAM_FILE_BASE_URL",
    )
    voice_enabled: bool = Field(default=True, validation_alias="VOICE_ENABLED")
    voice_stt_provider: str = Field(
        default="local_whisper",
        validation_alias="VOICE_STT_PROVIDER",
    )
    voice_stt_model: str = Field(default="base", validation_alias="VOICE_STT_MODEL")
    voice_stt_device: str = Field(default="cpu", validation_alias="VOICE_STT_DEVICE")
    voice_stt_compute_type: str = Field(
        default="int8",
        validation_alias="VOICE_STT_COMPUTE_TYPE",
    )
    voice_default_language: str = Field(default="ru", validation_alias="VOICE_DEFAULT_LANGUAGE")
    voice_audio_tmp_dir: str = Field(default="tmp/audio", validation_alias="VOICE_AUDIO_TMP_DIR")
    voice_max_audio_size_mb: int = Field(
        default=25,
        validation_alias="VOICE_MAX_AUDIO_SIZE_MB",
    )
    voice_keep_audio_files: bool = Field(
        default=False,
        validation_alias="VOICE_KEEP_AUDIO_FILES",
    )
    voice_convert_to_wav: bool = Field(
        default=True,
        validation_alias="VOICE_CONVERT_TO_WAV",
    )
    tracing_enabled: bool = Field(default=False, validation_alias="TRACING_ENABLED")
    otel_service_name: str = Field(
        default="supportops-ai-agent",
        validation_alias="OTEL_SERVICE_NAME",
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_environment: str = Field(default="local", validation_alias="OTEL_ENVIRONMENT")

    @property
    def EMBEDDING_MODEL_NAME(self) -> str:
        return self.embedding_model_name

    @property
    def QDRANT_COLLECTION_NAME(self) -> str:
        return self.qdrant_collection_name

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return self.ollama_base_url

    @property
    def OLLAMA_MODEL(self) -> str:
        return self.ollama_model

    @property
    def TELEGRAM_BOT_TOKEN(self) -> str | None:
        return self.telegram_bot_token

    @property
    def TELEGRAM_ALLOWED_USER_IDS(self) -> str | None:
        return self.telegram_allowed_user_ids

    @property
    def TELEGRAM_USE_BACKEND_HTTP(self) -> bool:
        return self.telegram_use_backend_http

    @property
    def TELEGRAM_BACKEND_CHAT_URL(self) -> str:
        return self.telegram_backend_chat_url

    @property
    def TELEGRAM_ENABLED(self) -> bool:
        return self.telegram_enabled

    @property
    def TELEGRAM_WEBHOOK_SECRET(self) -> str | None:
        return self.telegram_webhook_secret

    @property
    def TELEGRAM_API_BASE_URL(self) -> str:
        return self.telegram_api_base_url

    @property
    def TELEGRAM_FILE_BASE_URL(self) -> str:
        return self.telegram_file_base_url

    @property
    def VOICE_ENABLED(self) -> bool:
        return self.voice_enabled

    @property
    def VOICE_STT_PROVIDER(self) -> str:
        return self.voice_stt_provider

    @property
    def VOICE_STT_MODEL(self) -> str:
        return self.voice_stt_model

    @property
    def VOICE_STT_DEVICE(self) -> str:
        return self.voice_stt_device

    @property
    def VOICE_STT_COMPUTE_TYPE(self) -> str:
        return self.voice_stt_compute_type

    @property
    def VOICE_DEFAULT_LANGUAGE(self) -> str:
        return self.voice_default_language

    @property
    def VOICE_AUDIO_TMP_DIR(self) -> str:
        return self.voice_audio_tmp_dir

    @property
    def VOICE_MAX_AUDIO_SIZE_MB(self) -> int:
        return self.voice_max_audio_size_mb

    @property
    def VOICE_KEEP_AUDIO_FILES(self) -> bool:
        return self.voice_keep_audio_files

    @property
    def VOICE_CONVERT_TO_WAV(self) -> bool:
        return self.voice_convert_to_wav

    @property
    def TRACING_ENABLED(self) -> bool:
        return self.tracing_enabled

    @property
    def OTEL_SERVICE_NAME(self) -> str:
        return self.otel_service_name

    @property
    def OTEL_EXPORTER_OTLP_ENDPOINT(self) -> str:
        return self.otel_exporter_otlp_endpoint

    @property
    def OTEL_ENVIRONMENT(self) -> str:
        return self.otel_environment


@lru_cache
def get_settings() -> Settings:
    return Settings()
