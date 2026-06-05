from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

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
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            check_compatibility=False,
        )
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

    def delete_points(self, point_ids: list[str]) -> int:
        if not point_ids:
            return 0

        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )

        return len(point_ids)

    def delete_points_by_document_id(self, document_id: str) -> int:
        self.ensure_collection()

        document_filter = self._document_id_filter(document_id)
        points_count = self._count_points_by_filter(document_filter)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=document_filter),
            wait=True,
        )

        return points_count

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

    def _document_id_filter(self, document_id: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    def _count_points_by_filter(self, points_filter: Filter) -> int:
        points_count = 0
        offset: Any = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=points_filter,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points_count += len(records)
            if offset is None:
                return points_count


@lru_cache
def get_qdrant_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()
