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
