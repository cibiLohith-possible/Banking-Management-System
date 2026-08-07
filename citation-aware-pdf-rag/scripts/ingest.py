"""Ingestion script: PDF files → loader → chunks → embeddings → Qdrant.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --pdf-dir path/to/pdfs
    python scripts/ingest.py --chunk-size 800 --chunk-overlap 150
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `app.*` imports work when
# this script is run directly (e.g. `python scripts/ingest.py`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.chunker import PDFChunk, TextChunker
from app.embeddings import EmbeddingModel
from app.pdf_loader import PDFLoader, PDFPage
from app.qdrant_store import QdrantStore


# ---------------------------------------------------------------------------
# Core ingestion helpers
# ---------------------------------------------------------------------------


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """Return all .pdf files found in pdf_dir.

    Args:
        pdf_dir: Directory to scan.

    Returns:
        Sorted list of PDF file paths (may be empty).

    Raises:
        FileNotFoundError: If pdf_dir does not exist.
        NotADirectoryError: If pdf_dir is not a directory.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {pdf_dir}")

    return sorted(pdf_dir.glob("*.pdf"))


def load_pdf(pdf_path: Path) -> list[PDFPage]:
    """Load all non-empty pages from a single PDF.

    Args:
        pdf_path: Path to a .pdf file.

    Returns:
        List of PDFPage objects.

    Raises:
        ValueError: If the file is invalid or corrupted.
    """
    return PDFLoader(pdf_path).load()


def chunk_pages(pages: list[PDFPage], chunk_size: int, chunk_overlap: int) -> list[PDFChunk]:
    """Chunk all pages from a document.

    Args:
        pages: Extracted PDF pages.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.

    Returns:
        Flat list of PDFChunk objects.
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_pages(pages)


def embed_chunks(chunks: list[PDFChunk], model: EmbeddingModel) -> list[list[float]]:
    """Generate embeddings for a batch of chunks.

    Args:
        chunks: Chunks to embed.
        model: Initialised EmbeddingModel instance.

    Returns:
        List of embedding vectors in the same order as chunks.
    """
    texts = [c.text for c in chunks]
    return model.embed_batch(texts)


def store_chunks(
    chunks: list[PDFChunk],
    embeddings: list[list[float]],
    store: QdrantStore,
) -> int:
    """Upsert chunk metadata + embeddings into Qdrant.

    Args:
        chunks: PDFChunk objects with metadata.
        embeddings: Corresponding embedding vectors.
        store: Initialised QdrantStore instance.

    Returns:
        Number of points upserted.
    """
    chunk_dicts = [asdict(c) for c in chunks]
    return store.insert(chunk_dicts, embeddings)


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------


def ingest_pdf(
    pdf_path: Path,
    model: EmbeddingModel,
    store: QdrantStore,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, int]:
    """Run the full ingestion pipeline for a single PDF file.

    Args:
        pdf_path: Path to the PDF.
        model: Shared EmbeddingModel instance.
        store: Shared QdrantStore instance.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between chunks.

    Returns:
        Dict with counts: pages, chunks, stored.

    Raises:
        ValueError: If the PDF is invalid or corrupted.
    """
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)

    if not chunks:
        return {"pages": len(pages), "chunks": 0, "stored": 0}

    embeddings = embed_chunks(chunks, model)
    stored = store_chunks(chunks, embeddings, store)

    return {"pages": len(pages), "chunks": len(chunks), "stored": stored}


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------


def run_ingestion(
    pdf_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> None:
    """Discover and ingest all PDFs in pdf_dir into Qdrant.

    Args:
        pdf_dir: Directory containing PDF files.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between chunks.
    """
    # ---- 1. Discover PDFs -------------------------------------------------
    pdf_files = discover_pdfs(pdf_dir)

    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'. Nothing to ingest.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in '{pdf_dir}'\n")

    # ---- 2. Initialise shared components ----------------------------------
    print("Loading embedding model...")
    model = EmbeddingModel()
    print(f"Embedding model ready  (dimension={model.dimension})\n")

    store = QdrantStore(embedding_dimension=model.dimension)
    store.create_collection()  # no-op if collection already exists

    # ---- 3. Ingest each PDF -----------------------------------------------
    total_pages = total_chunks = total_stored = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            counts = ingest_pdf(pdf_path, model, store, chunk_size, chunk_overlap)
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            print(f"  Skipping '{pdf_path.name}' due to above error.\n")
            continue

        total_pages += counts["pages"]
        total_chunks += counts["chunks"]
        total_stored += counts["stored"]

        print(f"  Pages extracted : {counts['pages']}")
        print(f"  Chunks created  : {counts['chunks']}")
        print(f"  Stored in Qdrant: {counts['stored']}\n")

    # ---- 4. Summary -------------------------------------------------------
    print("=" * 45)
    print("Ingestion complete.")
    print(f"  Total pages   : {total_pages}")
    print(f"  Total chunks  : {total_chunks}")
    print(f"  Total stored  : {total_stored}")
    print("=" * 45)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into Qdrant for Citation-Aware RAG."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "pdfs",
        help="Directory containing PDF files (default: data/pdfs/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum character length per chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap in characters between consecutive chunks (default: 200)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ingestion(
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
