"""Text chunking module with metadata preservation and chunk overlap support."""

from dataclasses import dataclass
from typing import Any, Sequence

from app.pdf_loader import PDFPage


@dataclass(frozen=True)
class PDFChunk:
    """Represents a chunk of text extracted from a PDF page.

    Attributes:
        chunk_id: A unique and deterministic identifier for the chunk.
        document_name: The filename of the source PDF.
        page_number: The 1-based page number from which the chunk originated.
        text: The extracted chunk text.
    """

    chunk_id: str
    document_name: str
    page_number: int
    text: str


class TextChunker:
    """Splits PDF page content into smaller overlapping chunks while preserving metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize TextChunker with specified chunk size and overlap parameters.

        Args:
            chunk_size: Maximum character length for each chunk (must be > 0).
            chunk_overlap: Overlap in characters between consecutive chunks (must be >= 0 and < chunk_size).

        Raises:
            ValueError: If chunk parameters are invalid.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be greater than or equal to 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_fields(self, page: PDFPage | dict[str, Any]) -> tuple[str, int, str]:
        """Extract document_name, page_number, and text from PDFPage object or dict."""
        if isinstance(page, dict):
            return page["document_name"], int(page["page_number"]), str(page["text"])
        return page.document_name, page.page_number, page.text

    def chunk_page(self, page: PDFPage | dict[str, Any]) -> list[PDFChunk]:
        """Split a single PDF page into chunks.

        Args:
            page: A PDFPage instance or a dictionary with document_name, page_number, and text.

        Returns:
            A list of PDFChunk instances. Returns empty list for empty or whitespace-only text.
        """
        doc_name, page_num, raw_text = self._extract_fields(page)
        text = raw_text.strip()

        if not text:
            return []

        # Short text remaining as one chunk
        if len(text) <= self.chunk_size:
            chunk_id = f"{doc_name}_p{page_num}_c0"
            return [
                PDFChunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    page_number=page_num,
                    text=text,
                )
            ]

        chunks: list[PDFChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{doc_name}_p{page_num}_c{chunk_idx}"
                chunks.append(
                    PDFChunk(
                        chunk_id=chunk_id,
                        document_name=doc_name,
                        page_number=page_num,
                        text=chunk_text,
                    )
                )
                chunk_idx += 1

            if end >= len(text):
                break

            start += step

        return chunks

    def chunk_pages(self, pages: Sequence[PDFPage | dict[str, Any]]) -> list[PDFChunk]:
        """Split a sequence of PDF pages into chunks.

        Args:
            pages: Sequence of PDFPage objects or dictionaries.

        Returns:
            A list of generated PDFChunk instances.
        """
        all_chunks: list[PDFChunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_page(page))
        return all_chunks
