from __future__ import annotations

from typing import Any

from app.vectorstore.qdrant_store import QdrantVectorStore


class FakeQdrantClient:
    def __init__(self, scroll_records: list[Any] | None = None) -> None:
        self.delete_calls: list[dict[str, Any]] = []
        self.scroll_records = scroll_records or []
        self.collection_exists_calls = 0
        self.create_collection_calls = 0

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls += 1
        return True

    def create_collection(self, **kwargs: Any) -> None:
        self.create_collection_calls += 1

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)

    def scroll(self, **kwargs: Any) -> tuple[list[Any], None]:
        return self.scroll_records, None


def test_delete_points_empty_list_returns_zero_without_client_delete() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, collection_name="test_collection")

    deleted_count = store.delete_points([])

    assert deleted_count == 0
    assert client.delete_calls == []


def test_delete_points_calls_qdrant_delete_with_wait() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, collection_name="test_collection")

    deleted_count = store.delete_points(["point-1", "point-2"])

    assert deleted_count == 2
    assert len(client.delete_calls) == 1
    delete_call = client.delete_calls[0]
    assert delete_call["collection_name"] == "test_collection"
    assert delete_call["wait"] is True
    assert delete_call["points_selector"].points == ["point-1", "point-2"]


def test_delete_points_by_document_id_uses_payload_filter() -> None:
    client = FakeQdrantClient(scroll_records=[object(), object()])
    store = QdrantVectorStore(client=client, collection_name="test_collection")

    deleted_count = store.delete_points_by_document_id("document-1")

    assert deleted_count == 2
    assert len(client.delete_calls) == 1
    delete_call = client.delete_calls[0]
    points_filter = delete_call["points_selector"].filter
    assert delete_call["collection_name"] == "test_collection"
    assert delete_call["wait"] is True
    assert points_filter.must[0].key == "document_id"
    assert points_filter.must[0].match.value == "document-1"
