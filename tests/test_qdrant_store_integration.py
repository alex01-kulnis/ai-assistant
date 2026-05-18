import uuid

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.schemas.vector import VectorChunkInput  # noqa: E402
from app.vectorstore.qdrant_store import QdrantVectorStore  # noqa: E402


@pytest.mark.integration
def test_qdrant_upsert_and_search_finds_chunk() -> None:
    settings = get_settings()
    collection_name = f"{settings.QDRANT_COLLECTION_NAME}_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=settings.qdrant_url, timeout=5, check_compatibility=False)
    store = QdrantVectorStore(client=client, collection_name=collection_name)

    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant is not available: {exc}")

    store.ensure_collection()

    document_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    vector = [0.0] * 384
    vector[0] = 1.0

    try:
        point_ids = store.upsert_chunks(
            [
                VectorChunkInput(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    chunk_index=0,
                    filename="faq.txt",
                    page_number=None,
                    text="Password reset instructions",
                    vector=vector,
                )
            ]
        )

        results = store.search(query_vector=vector, limit=1)
    finally:
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

    assert len(point_ids) == 1
    assert len(results) == 1
    assert results[0].point_id == point_ids[0]
    assert results[0].text == "Password reset instructions"
    assert results[0].document_id == str(document_id)
    assert results[0].chunk_id == str(chunk_id)
    assert results[0].filename == "faq.txt"
    assert results[0].page_number is None
    assert results[0].chunk_index == 0
