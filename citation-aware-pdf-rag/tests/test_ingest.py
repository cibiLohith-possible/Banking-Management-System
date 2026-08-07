"""Tests for the ingestion pipeline (scripts/ingest.py).

Strategy
--------
* Real PDF generation via PyMuPDF (no binary fixtures committed).
* Real PDF loader, chunker, and embedding model — no mocks for those.
* QdrantStore is mocked to isolate external service dependency.
  The mock captures every insert() call so we can assert on metadata.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import fitz
import pytest

# Ensure project root is importable when pytest runs from any CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.chunker import PDFChunk, TextChunker
from app.embeddings import EmbeddingModel
from app.pdf_loader import PDFLoader, PDFPage
from scripts.ingest import (
    chunk_pages,
    discover_pdfs,
    embed_chunks,
    ingest_pdf,
    load_pdf,
    run_ingestion,
    store_chunks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def embedding_model() -> EmbeddingModel:
    """Session-scoped real embedding model — loaded once for the whole test session."""
    return EmbeddingModel()


@pytest.fixture()
def pdf_dir(tmp_path: Path) -> Path:
    """Return a temporary directory to use as the PDF store."""
    return tmp_path / "pdfs"


def _write_pdf(directory: Path, filename: str, pages: list[str]) -> Path:
    """Create a real PDF with PyMuPDF and return its path.

    Args:
        directory: Target directory (created if absent).
        filename: Filename including .pdf extension.
        pages: List of text strings — one per page.
    """
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / filename
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# discover_pdfs
# ---------------------------------------------------------------------------


def test_discover_pdfs_finds_pdf_files(pdf_dir: Path):
    """PDF files in the directory are discovered."""
    _write_pdf(pdf_dir, "alpha.pdf", ["Page 1"])
    _write_pdf(pdf_dir, "beta.pdf", ["Page 1"])
    found = discover_pdfs(pdf_dir)
    assert len(found) == 2
    assert all(p.suffix == ".pdf" for p in found)


def test_discover_pdfs_ignores_non_pdf_files(pdf_dir: Path):
    """Non-PDF files are not returned."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / "notes.txt").write_text("hello")
    (pdf_dir / "data.csv").write_text("a,b,c")
    _write_pdf(pdf_dir, "real.pdf", ["Content"])
    found = discover_pdfs(pdf_dir)
    assert len(found) == 1
    assert found[0].name == "real.pdf"


def test_discover_pdfs_empty_directory_returns_empty_list(pdf_dir: Path):
    """An existing but empty directory yields an empty list (no exception)."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    assert discover_pdfs(pdf_dir) == []


def test_discover_pdfs_missing_directory_raises(tmp_path: Path):
    """Missing directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="PDF directory not found"):
        discover_pdfs(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------


def test_load_pdf_returns_pages(pdf_dir: Path):
    """load_pdf returns PDFPage objects for each non-empty page."""
    pdf_path = _write_pdf(pdf_dir, "doc.pdf", ["First page content", "Second page content"])
    pages = load_pdf(pdf_path)
    assert len(pages) == 2
    assert all(isinstance(p, PDFPage) for p in pages)


def test_load_pdf_preserves_document_name(pdf_dir: Path):
    """document_name matches the PDF filename."""
    pdf_path = _write_pdf(pdf_dir, "myreport.pdf", ["Some text"])
    pages = load_pdf(pdf_path)
    assert pages[0].document_name == "myreport.pdf"


# ---------------------------------------------------------------------------
# chunk_pages
# ---------------------------------------------------------------------------


def test_chunk_pages_converts_pages_to_chunks(pdf_dir: Path):
    """Pages become PDFChunk objects."""
    pdf_path = _write_pdf(pdf_dir, "doc.pdf", ["Hello world content on page one"])
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) >= 1
    assert all(isinstance(c, PDFChunk) for c in chunks)


def test_chunk_pages_metadata_preserved(pdf_dir: Path):
    """document_name and page_number survive chunking."""
    pdf_path = _write_pdf(pdf_dir, "preserve.pdf", ["Page one text"])
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
    assert chunks[0].document_name == "preserve.pdf"
    assert chunks[0].page_number == 1


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------


def test_embed_chunks_produces_correct_count(pdf_dir: Path, embedding_model: EmbeddingModel):
    """Number of embeddings equals number of chunks."""
    pdf_path = _write_pdf(pdf_dir, "embed.pdf", ["Text for embedding"])
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
    embeddings = embed_chunks(chunks, embedding_model)
    assert len(embeddings) == len(chunks)


def test_embed_chunks_returns_float_lists(pdf_dir: Path, embedding_model: EmbeddingModel):
    """Each embedding is a list of floats with the correct dimension."""
    pdf_path = _write_pdf(pdf_dir, "floats.pdf", ["Floating point check"])
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
    embeddings = embed_chunks(chunks, embedding_model)
    assert all(isinstance(v, float) for v in embeddings[0])
    assert len(embeddings[0]) == embedding_model.dimension


# ---------------------------------------------------------------------------
# store_chunks (mock Qdrant)
# ---------------------------------------------------------------------------


def test_store_chunks_calls_insert_with_correct_data(pdf_dir: Path, embedding_model: EmbeddingModel):
    """store_chunks calls QdrantStore.insert with serialised chunks and vectors."""
    pdf_path = _write_pdf(pdf_dir, "store.pdf", ["Store test content"])
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
    embeddings = embed_chunks(chunks, embedding_model)

    mock_store = MagicMock()
    mock_store.insert.return_value = len(chunks)

    count = store_chunks(chunks, embeddings, mock_store)

    assert count == len(chunks)
    mock_store.insert.assert_called_once()
    call_args = mock_store.insert.call_args
    inserted_chunks, inserted_vectors = call_args[0]
    assert len(inserted_chunks) == len(chunks)
    assert len(inserted_vectors) == len(embeddings)
    # Verify required metadata keys are present
    for ch in inserted_chunks:
        assert "chunk_id" in ch
        assert "document_name" in ch
        assert "page_number" in ch
        assert "text" in ch


# ---------------------------------------------------------------------------
# ingest_pdf (mock Qdrant)
# ---------------------------------------------------------------------------


def test_ingest_pdf_returns_correct_counts(pdf_dir: Path, embedding_model: EmbeddingModel):
    """ingest_pdf returns pages, chunks, stored counts for a valid PDF."""
    pdf_path = _write_pdf(pdf_dir, "pipeline.pdf", ["Page one", "Page two"])

    mock_store = MagicMock()
    mock_store.insert.side_effect = lambda chunks, vecs: len(chunks)

    counts = ingest_pdf(pdf_path, embedding_model, mock_store, 1000, 200)

    assert counts["pages"] == 2
    assert counts["chunks"] >= 2
    assert counts["stored"] == counts["chunks"]


def test_ingest_pdf_raises_on_corrupted_file(pdf_dir: Path, embedding_model: EmbeddingModel):
    """ingest_pdf raises ValueError for a corrupted PDF."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    bad_pdf = pdf_dir / "corrupt.pdf"
    bad_pdf.write_bytes(b"this is not a real pdf")

    mock_store = MagicMock()
    with pytest.raises(ValueError):
        ingest_pdf(bad_pdf, embedding_model, mock_store, 1000, 200)


# ---------------------------------------------------------------------------
# run_ingestion (mock Qdrant, real everything else)
# ---------------------------------------------------------------------------


def test_run_ingestion_processes_multiple_pdfs(pdf_dir: Path, embedding_model: EmbeddingModel, capsys):
    """run_ingestion handles multiple PDFs and prints progress."""
    _write_pdf(pdf_dir, "doc1.pdf", ["First document content"])
    _write_pdf(pdf_dir, "doc2.pdf", ["Second document content"])

    mock_store = MagicMock()
    mock_store.insert.side_effect = lambda chunks, vecs: len(chunks)

    with (
        patch("scripts.ingest.EmbeddingModel", return_value=embedding_model),
        patch("scripts.ingest.QdrantStore", return_value=mock_store),
    ):
        run_ingestion(pdf_dir=pdf_dir)

    captured = capsys.readouterr()
    assert "doc1.pdf" in captured.out
    assert "doc2.pdf" in captured.out
    assert "Ingestion complete" in captured.out


def test_run_ingestion_empty_directory_exits_cleanly(pdf_dir: Path, capsys):
    """Empty PDF directory prints a message and returns without error."""
    pdf_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("scripts.ingest.EmbeddingModel"),
        patch("scripts.ingest.QdrantStore"),
    ):
        run_ingestion(pdf_dir=pdf_dir)

    captured = capsys.readouterr()
    assert "No PDF files found" in captured.out


def test_run_ingestion_missing_directory_raises(tmp_path: Path):
    """Missing pdf_dir raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_ingestion(pdf_dir=tmp_path / "nonexistent")


def test_run_ingestion_non_pdf_files_are_ignored(pdf_dir: Path, embedding_model: EmbeddingModel, capsys):
    """Non-PDF files in the directory do not cause errors or processing."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / "readme.txt").write_text("ignore me")
    (pdf_dir / "image.png").write_bytes(b"\x89PNG")
    _write_pdf(pdf_dir, "only.pdf", ["Only PDF content"])

    mock_store = MagicMock()
    mock_store.insert.side_effect = lambda chunks, vecs: len(chunks)

    with (
        patch("scripts.ingest.EmbeddingModel", return_value=embedding_model),
        patch("scripts.ingest.QdrantStore", return_value=mock_store),
    ):
        run_ingestion(pdf_dir=pdf_dir)

    captured = capsys.readouterr()
    assert "only.pdf" in captured.out
    assert "readme.txt" not in captured.out
    assert "image.png" not in captured.out


# ---------------------------------------------------------------------------
# Idempotency / no duplicate chunk IDs
# ---------------------------------------------------------------------------


def test_reingest_same_pdf_produces_identical_chunk_ids(pdf_dir: Path, embedding_model: EmbeddingModel):
    """Re-ingesting the same PDF yields exactly the same set of chunk_ids both times."""
    pdf_path = _write_pdf(pdf_dir, "idem.pdf", ["Idempotency test content page one"])

    mock_store = MagicMock()
    all_inserted_chunk_ids: list[list[str]] = []

    def capture_insert(chunks, vecs):
        all_inserted_chunk_ids.append([c["chunk_id"] for c in chunks])
        return len(chunks)

    mock_store.insert.side_effect = capture_insert

    # First ingestion
    ingest_pdf(pdf_path, embedding_model, mock_store, 1000, 200)
    # Second ingestion (same PDF)
    ingest_pdf(pdf_path, embedding_model, mock_store, 1000, 200)

    assert len(all_inserted_chunk_ids) == 2
    # Both runs produce the same chunk_ids
    assert set(all_inserted_chunk_ids[0]) == set(all_inserted_chunk_ids[1])
