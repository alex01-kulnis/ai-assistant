import pytest

from app.services.chunking_service import TextChunkingService
from app.services.document_parser import ParsedPage


def test_split_text_returns_single_chunk_when_text_is_shorter_than_chunk_size() -> None:
    chunks = TextChunkingService().split_text(
        "Short support answer.",
        chunk_size=100,
        overlap=10,
    )

    assert chunks == ["Short support answer."]


def test_split_text_uses_overlap() -> None:
    chunks = TextChunkingService().split_text(
        "abcdefghij",
        chunk_size=4,
        overlap=2,
    )

    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_split_text_filters_empty_chunks() -> None:
    chunks = TextChunkingService().split_text(
        "     ",
        chunk_size=3,
        overlap=1,
    )

    assert chunks == []


def test_split_text_rejects_overlap_greater_than_or_equal_to_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        TextChunkingService().split_text(
            "abcdefghij",
            chunk_size=4,
            overlap=4,
        )


def test_split_pages_keeps_page_number_and_assigns_chunk_indexes() -> None:
    pages = [
        ParsedPage(page_number=1, text="abcdef"),
        ParsedPage(page_number=2, text="ghijkl"),
    ]

    chunks = TextChunkingService().split_pages(
        pages,
        chunk_size=4,
        overlap=1,
    )

    assert [chunk.text for chunk in chunks] == ["abcd", "def", "ghij", "jkl"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]


def test_split_pages_skips_empty_pages() -> None:
    pages = [
        ParsedPage(page_number=1, text=""),
        ParsedPage(page_number=2, text="abc"),
    ]

    chunks = TextChunkingService().split_pages(
        pages,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "abc"
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 2
