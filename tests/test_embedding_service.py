from typing import Any

from app.services.embedding_service import EmbeddingService, get_embedding_service


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "sentences": sentences,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [[float(index)] * 384 for index, _ in enumerate(sentences)]


def test_embed_documents_returns_embeddings() -> None:
    model = FakeEmbeddingModel()
    service = EmbeddingService(model_name="test-model", model=model)

    embeddings = service.embed_documents(["First chunk", "Second chunk"])

    assert len(embeddings) == 2
    assert all(len(vector) == 384 for vector in embeddings)
    assert model.calls[0]["sentences"] == [
        "passage: First chunk",
        "passage: Second chunk",
    ]
    assert model.calls[0]["normalize_embeddings"] is True


def test_embed_query_returns_single_vector() -> None:
    model = FakeEmbeddingModel()
    service = EmbeddingService(model_name="test-model", model=model)

    vector = service.embed_query("How do I reset my password?")

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert model.calls[0]["sentences"] == ["query: How do I reset my password?"]
    assert model.calls[0]["normalize_embeddings"] is True


def test_embedding_vector_dimension_is_384() -> None:
    model = FakeEmbeddingModel()
    service = EmbeddingService(model_name="test-model", model=model)

    [vector] = service.embed_documents(["Dimension check"])

    assert len(vector) == 384


def test_get_embedding_service_returns_cached_instance() -> None:
    get_embedding_service.cache_clear()

    first_service = get_embedding_service()
    second_service = get_embedding_service()

    assert first_service is second_service
