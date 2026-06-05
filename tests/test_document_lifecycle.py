from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.documents import get_document_ingestion_service
from app.db.session import get_db_session
from app.main import app
from app.models.document import Document, DocumentChunk
from app.schemas.document import DocumentDeleteResponse, DocumentReindexResponse
from app.services.document_ingestion_service import DocumentIndexingError, DocumentIngestionService
from app.services.document_parser import DocumentParsingError


class FakeEmbeddingService:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vector = [0.0] * 384
        vector[0] = 1.0
        return [vector[:] for _ in texts]


class FakeVectorStore:
    def __init__(self, *, fail_delete: bool = False, fallback_count: int = 0) -> None:
        self.fail_delete = fail_delete
        self.fallback_count = fallback_count
        self.deleted_point_ids: list[list[str]] = []
        self.deleted_document_ids: list[str] = []
        self.upserted_chunks: list[Any] = []

    def upsert_chunks(self, chunks: list[Any]) -> list[str]:
        self.upserted_chunks.extend(chunks)
        return [f"new-point-{index}" for index, _chunk in enumerate(chunks)]

    def delete_points(self, point_ids: list[str]) -> int:
        if self.fail_delete:
            raise RuntimeError("qdrant unavailable")
        self.deleted_point_ids.append(point_ids)
        return len(point_ids)

    def delete_points_by_document_id(self, document_id: str) -> int:
        self.deleted_document_ids.append(document_id)
        return self.fallback_count


class FailingParser:
    def parse(self, file_path: str) -> Any:
        raise DocumentParsingError("parser failed")


class FakeScalarResult:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def all(self) -> list[Any]:
        return list(self.items)


class FakeExecuteResult:
    def __init__(
        self,
        *,
        scalar_items: list[Any] | None = None,
        rows: list[Any] | None = None,
    ) -> None:
        self.scalar_items = scalar_items or []
        self.rows = rows or []

    def scalars(self) -> FakeScalarResult:
        return FakeScalarResult(self.scalar_items)

    def all(self) -> list[Any]:
        return list(self.rows)


class FakeLifecycleSession:
    def __init__(self, document: Document | None, chunks: list[DocumentChunk]) -> None:
        self.document = document
        self.chunks = chunks
        self.added: list[Any] = []
        self.commits = 0
        self.rollbacks = 0
        self.document_deleted = False
        self.return_document_rows = False

    def add(self, instance: Any) -> None:
        self.added.append(instance)
        if isinstance(instance, Document):
            self.document = instance

    def add_all(self, instances: list[Any]) -> None:
        self.added.extend(instances)
        self.chunks.extend(
            instance for instance in instances if isinstance(instance, DocumentChunk)
        )

    async def get(self, model: type, primary_key: Any) -> Any:
        if model is Document and self.document is not None and self.document.id == primary_key:
            return self.document
        return None

    async def execute(self, statement: Any) -> FakeExecuteResult:
        if self.return_document_rows and self.document is not None:
            return FakeExecuteResult(
                rows=[
                    SimpleNamespace(
                        id=self.document.id,
                        filename=self.document.filename,
                        content_type=self.document.content_type,
                        status=self.document.status,
                        created_at=self.document.created_at,
                        chunks_count=len(self.chunks),
                    )
                ]
            )
        return FakeExecuteResult(scalar_items=self.chunks)

    async def delete(self, instance: Any) -> None:
        if isinstance(instance, DocumentChunk):
            self.chunks = [chunk for chunk in self.chunks if chunk.id != instance.id]
        if isinstance(instance, Document):
            self.document_deleted = True
            self.document = None

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


@dataclass
class FakeDeleteService:
    response: DocumentDeleteResponse | None = None
    error: DocumentIndexingError | None = None

    async def delete_document(self, *, document_id: str, session: Any) -> DocumentDeleteResponse:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


@dataclass
class FakeReindexService:
    response: DocumentReindexResponse | None = None
    error: DocumentIndexingError | None = None

    async def reindex_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str | None,
        content: bytes,
        session: Any,
    ) -> DocumentReindexResponse:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def make_document() -> Document:
    return Document(
        id=uuid.uuid4(),
        filename="old_policy.txt",
        content_type="text/plain",
        status="indexed",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def make_chunk(document_id: uuid.UUID, point_id: str = "old-point-1") -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=0,
        text="Old policy text",
        page_number=None,
        qdrant_point_id=point_id,
    )


def make_service(
    vector_store: FakeVectorStore | None = None,
    parser: Any | None = None,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        parser=parser,
        embedding_service=FakeEmbeddingService(),  # type: ignore[arg-type]
        vector_store=vector_store or FakeVectorStore(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_delete_document_deletes_qdrant_points_chunks_and_document() -> None:
    document = make_document()
    chunk = make_chunk(document.id)
    session = FakeLifecycleSession(document=document, chunks=[chunk])
    vector_store = FakeVectorStore()
    service = make_service(vector_store=vector_store)

    response = await service.delete_document(
        document_id=str(document.id),
        session=session,  # type: ignore[arg-type]
    )

    assert response.document_id == str(document.id)
    assert response.filename == "old_policy.txt"
    assert response.deleted_chunks_count == 1
    assert response.deleted_qdrant_points_count == 1
    assert response.status == "deleted"
    assert vector_store.deleted_point_ids == [["old-point-1"]]
    assert vector_store.deleted_document_ids == [str(document.id)]
    assert session.chunks == []
    assert session.document_deleted is True


@pytest.mark.asyncio
async def test_delete_document_returns_service_error_when_document_not_found() -> None:
    service = make_service()
    session = FakeLifecycleSession(document=None, chunks=[])

    with pytest.raises(DocumentIndexingError) as exc_info:
        await service.delete_document(
            document_id=str(uuid.uuid4()),
            session=session,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_does_not_delete_postgres_rows_when_qdrant_delete_fails() -> None:
    document = make_document()
    chunk = make_chunk(document.id)
    session = FakeLifecycleSession(document=document, chunks=[chunk])
    service = make_service(vector_store=FakeVectorStore(fail_delete=True))

    with pytest.raises(DocumentIndexingError) as exc_info:
        await service.delete_document(
            document_id=str(document.id),
            session=session,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 500
    assert session.chunks == [chunk]
    assert session.document_deleted is False


@pytest.mark.asyncio
async def test_reindex_document_replaces_chunks_and_updates_document_metadata() -> None:
    document = make_document()
    old_chunk = make_chunk(document.id)
    session = FakeLifecycleSession(document=document, chunks=[old_chunk])
    vector_store = FakeVectorStore()
    service = make_service(vector_store=vector_store)

    response = await service.reindex_document(
        document_id=str(document.id),
        filename="new_policy.md",
        content_type="text/markdown",
        content=b"# Updated policy\n\nNew refund instructions.",
        session=session,  # type: ignore[arg-type]
    )

    assert response.document_id == str(document.id)
    assert response.filename == "new_policy.md"
    assert response.chunks_count == 1
    assert response.status == "indexed"
    assert document.filename == "new_policy.md"
    assert document.content_type == "text/markdown"
    assert document.status == "indexed"
    assert vector_store.deleted_point_ids == [["old-point-1"]]
    assert vector_store.upserted_chunks[0].filename == "new_policy.md"
    assert len(session.chunks) == 1
    assert session.chunks[0].qdrant_point_id == "new-point-0"

    session.return_document_rows = True
    documents = await service.list_documents(session=session)  # type: ignore[arg-type]
    assert documents[0].chunks_count == 1
    assert documents[0].filename == "new_policy.md"


@pytest.mark.asyncio
async def test_reindex_document_unknown_id_returns_404() -> None:
    service = make_service()
    session = FakeLifecycleSession(document=None, chunks=[])

    with pytest.raises(DocumentIndexingError) as exc_info:
        await service.reindex_document(
            document_id=str(uuid.uuid4()),
            filename="new_policy.txt",
            content_type="text/plain",
            content=b"new",
            session=session,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_reindex_document_marks_document_failed_when_parser_fails() -> None:
    document = make_document()
    old_chunk = make_chunk(document.id)
    session = FakeLifecycleSession(document=document, chunks=[old_chunk])
    vector_store = FakeVectorStore()
    service = make_service(vector_store=vector_store, parser=FailingParser())

    with pytest.raises(DocumentIndexingError) as exc_info:
        await service.reindex_document(
            document_id=str(document.id),
            filename="new_policy.txt",
            content_type="text/plain",
            content=b"new",
            session=session,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 400
    assert document.status == "failed"
    assert session.chunks == []
    assert vector_store.deleted_point_ids == [["old-point-1"]]


async def override_session() -> Any:
    yield object()


def test_delete_document_endpoint_returns_expected_response() -> None:
    document_id = str(uuid.uuid4())
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_document_ingestion_service] = lambda: FakeDeleteService(
        response=DocumentDeleteResponse(
            document_id=document_id,
            filename="old_policy.txt",
            deleted_chunks_count=1,
            deleted_qdrant_points_count=1,
        )
    )
    client = TestClient(app)

    try:
        response = client.delete(f"/api/v1/documents/{document_id}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "filename": "old_policy.txt",
        "deleted_chunks_count": 1,
        "deleted_qdrant_points_count": 1,
        "status": "deleted",
    }


def test_delete_document_endpoint_returns_404() -> None:
    document_id = str(uuid.uuid4())
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_document_ingestion_service] = lambda: FakeDeleteService(
        error=DocumentIndexingError("Document not found.", status_code=404)
    )
    client = TestClient(app)

    try:
        response = client.delete(f"/api/v1/documents/{document_id}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}


def test_delete_document_endpoint_returns_400_for_invalid_uuid() -> None:
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_document_ingestion_service] = lambda: FakeDeleteService(
        error=DocumentIndexingError("document_id must be a valid UUID.", status_code=400)
    )
    client = TestClient(app)

    try:
        response = client.delete("/api/v1/documents/not-a-uuid")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {"detail": "document_id must be a valid UUID."}


def test_reindex_document_endpoint_returns_expected_response() -> None:
    document_id = str(uuid.uuid4())
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_document_ingestion_service] = lambda: FakeReindexService(
        response=DocumentReindexResponse(
            document_id=document_id,
            filename="new_policy.txt",
            chunks_count=1,
            status="indexed",
        )
    )
    client = TestClient(app)

    try:
        response = client.post(
            f"/api/v1/documents/{document_id}/reindex",
            files={"file": ("new_policy.txt", b"Updated text", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "filename": "new_policy.txt",
        "chunks_count": 1,
        "status": "indexed",
    }
