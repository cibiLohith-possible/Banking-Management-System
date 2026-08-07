"""Unit tests for text chunking module."""

import pytest

from app.chunker import PDFChunk, TextChunker
from app.pdf_loader import PDFPage


def test_short_text_produces_one_chunk():
    """Verify that text shorter than chunk_size produces exactly one chunk."""
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    page = PDFPage(document_name="test.pdf", page_number=1, text="Short text snippet.")

    chunks = chunker.chunk_page(page)

    assert len(chunks) == 1
    assert isinstance(chunks[0], PDFChunk)
    assert chunks[0].text == "Short text snippet."


def test_dict_input_format_supported():
    """Verify that input as a dictionary matching the schema works correctly."""
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    page_dict = {
        "document_name": "manual.pdf",
        "page_number": 3,
        "text": "Content inside dictionary input.",
    }

    chunks = chunker.chunk_page(page_dict)

    assert len(chunks) == 1
    assert chunks[0].document_name == "manual.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].text == "Content inside dictionary input."


def test_long_text_produces_multiple_chunks():
    """Verify that text longer than chunk_size splits into multiple chunks."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    long_text = "A" * 250
    page = PDFPage(document_name="report.pdf", page_number=2, text=long_text)

    chunks = chunker.chunk_page(page)

    assert len(chunks) > 1
    total_reconstructed = sum(len(c.text) for c in chunks)
    assert total_reconstructed >= 250  # Accounts for overlapping characters


def test_chunk_overlap_works_correctly():
    """Verify that consecutive chunks overlap by the specified character count."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    # 0..149 characters
    text = "".join(f"{i:02d}" for i in range(75))  # 150 chars total
    page = PDFPage(document_name="doc.pdf", page_number=1, text=text)

    chunks = chunker.chunk_page(page)

    assert len(chunks) == 2
    # First chunk: text[0:100], Second chunk: text[80:150]
    # Overlap is text[80:100] (20 chars)
    overlap = chunks[0].text[-20:]
    assert chunks[1].text.startswith(overlap)


def test_document_name_and_page_number_preserved():
    """Verify metadata (document_name and page_number) is preserved across all chunks."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "X" * 120
    page = PDFPage(document_name="architecture_spec.pdf", page_number=5, text=text)

    chunks = chunker.chunk_page(page)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_name == "architecture_spec.pdf"
        assert chunk.page_number == 5


def test_chunk_id_is_unique_and_deterministic():
    """Verify that chunk_ids are unique per chunk and deterministic across executions."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "Z" * 130
    page = PDFPage(document_name="guide.pdf", page_number=1, text=text)

    chunks1 = chunker.chunk_page(page)
    chunks2 = chunker.chunk_page(page)

    # Check uniqueness within single pass
    chunk_ids = [c.chunk_id for c in chunks1]
    assert len(chunk_ids) == len(set(chunk_ids))

    # Check determinism across runs
    assert chunk_ids == [c.chunk_id for c in chunks2]
    assert chunk_ids[0] == "guide.pdf_p1_c0"
    assert chunk_ids[1] == "guide.pdf_p1_c1"


def test_empty_or_whitespace_text_ignored():
    """Verify that empty or whitespace-only text returns no chunks."""
    chunker = TextChunker()

    empty_page = PDFPage(document_name="empty.pdf", page_number=1, text="   \n\t  ")
    assert chunker.chunk_page(empty_page) == []


def test_invalid_chunk_size_raises_value_error():
    """Verify that non-positive chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        TextChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        TextChunker(chunk_size=-100)


def test_invalid_chunk_overlap_raises_value_error():
    """Verify that negative chunk_overlap raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be greater than or equal to 0"):
        TextChunker(chunk_size=1000, chunk_overlap=-10)


def test_chunk_overlap_greater_than_or_equal_chunk_size_raises_value_error():
    """Verify that chunk_overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="must be strictly smaller than chunk_size"):
        TextChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="must be strictly smaller than chunk_size"):
        TextChunker(chunk_size=500, chunk_overlap=600)


def test_chunk_pages_sequence():
    """Verify batch processing of multiple pages."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    pages = [
        PDFPage(document_name="doc.pdf", page_number=1, text="Page 1 content"),
        PDFPage(document_name="doc.pdf", page_number=2, text="Page 2 content"),
    ]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
