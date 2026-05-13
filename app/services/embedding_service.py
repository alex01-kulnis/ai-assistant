from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from app.core.config import get_settings


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any:
        ...


class EmbeddingService:
    def __init__(
        self,
        model_name: str | None = None,
        model: EmbeddingModel | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = model

    @property
    def model(self) -> EmbeddingModel:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [f"passage: {text}" for text in texts]
        return self._encode(prefixed_texts)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([f"query: {query}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        return [[float(value) for value in vector] for vector in embeddings]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
