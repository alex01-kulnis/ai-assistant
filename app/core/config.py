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
        default="postgresql+asyncpg://supportops:supportops@localhost:5432/supportops",
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
