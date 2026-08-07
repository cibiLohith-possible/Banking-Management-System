"""Citation formatting and verification module.

Ensures citations are derived strictly from Qdrant metadata rather than LLM generation.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from app.qdrant_store import SearchResult


@dataclass(frozen=True)
class Citation:
    """Represents a citation source derived from Qdrant metadata.

    Attributes:
        document_name: Source PDF filename.
        page_number: 1-based page number.
        retrieved_text: Exact retrieved text snippet from Qdrant.
    """

    document_name: str
    page_number: int
    retrieved_text: str

    def format_text(self) -> str:
        """Return standard human-readable citation text representation.

        Format:
            Document: <document_name>
            Page: <page_number>
            Retrieved Text:
            "<retrieved_text>"
        """
        return (
            f"Document: {self.document_name}\n"
            f"Page: {self.page_number}\n"
            f"Retrieved Text:\n"
            f'"{self.retrieved_text}"'
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Citation to dictionary representation for API responses."""
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "retrieved_text": self.retrieved_text,
        }


class CitationFormatter:
    """Formats and deduplicates citations from retrieved search results."""

    @staticmethod
    def extract_citations(
        results: Sequence[SearchResult | dict[str, Any]]
    ) -> list[Citation]:
        """Extract citations from search results, deduplicating while preserving order.

        Args:
            results: Sequence of SearchResult objects or dictionaries.

        Returns:
            Deduplicated list of Citation instances in original relevance order.
        """
        citations: list[Citation] = []
        seen_keys: set[tuple[str, int, str]] = set()

        for res in results:
            if isinstance(res, SearchResult):
                doc_name = res.document_name
                page_num = res.page_number
                text = res.text
            elif isinstance(res, dict):
                doc_name = str(res.get("document_name", ""))
                page_num = int(res.get("page_number", 0))
                text = str(res.get("text", ""))
            else:
                continue

            if not doc_name or not text:
                continue

            dedup_key = (doc_name, page_num, text)
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                citations.append(
                    Citation(
                        document_name=doc_name,
                        page_number=page_num,
                        retrieved_text=text,
                    )
                )

        return citations

    @classmethod
    def format_citations_string(
        cls, results: Sequence[SearchResult | dict[str, Any]]
    ) -> str:
        """Format all citations into a single multi-source human-readable string.

        Args:
            results: Sequence of SearchResult objects or dicts.

        Returns:
            Formatted citation block or empty string if no valid sources.
        """
        citations = cls.extract_citations(results)
        if not citations:
            return ""

        formatted_blocks = [c.format_text() for c in citations]
        return "\n\n".join(formatted_blocks)
