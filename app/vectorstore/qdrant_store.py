from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings
from app.schemas.vector import RetrievedChunk, VectorChunkInput


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        vector_size: int = 384,
    ) -> None:
        settings = get_settings()
        self.client = client or QdrantClient(url=settings.qdrant_url)
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: list[VectorChunkInput]) -> list[str]:
        if not chunks:
            return []

        self.ensure_collection()

        point_ids: list[str] = []
        points: list[PointStruct] = []

        for chunk in chunks:
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.vector,
                    payload={
                        "document_id": str(chunk.document_id),
                        "chunk_id": str(chunk.chunk_id),
                        "chunk_index": chunk.chunk_index,
                        "filename": chunk.filename,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return point_ids

    def search(self, query_vector: list[float], limit: int = 5) -> list[RetrievedChunk]:
        self.ensure_collection()

        if hasattr(self.client, "search"):
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            search_results = response.points

        return [self._to_retrieved_chunk(point) for point in search_results]

    def _to_retrieved_chunk(self, point: Any) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            point_id=str(point.id),
            score=float(point.score),
            text=str(payload["text"]),
            document_id=str(payload["document_id"]),
            chunk_id=str(payload["chunk_id"]),
            filename=str(payload["filename"]),
            page_number=payload.get("page_number"),
            chunk_index=int(payload["chunk_index"]),
        )


@lru_cache
def get_qdrant_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()
