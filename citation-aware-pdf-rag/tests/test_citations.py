"""Unit tests for citation system."""

from app.citation import Citation, CitationFormatter
from app.qdrant_store import SearchResult


def test_citation_format_text():
    """Verify exact formatting of individual Citation instances."""
    citation = Citation(
        document_name="employee_handbook.pdf",
        page_number=17,
        retrieved_text="Full-time employees receive 24 days of annual leave.",
    )

    formatted = citation.format_text()

    assert "Document: employee_handbook.pdf" in formatted
    assert "Page: 17" in formatted
    assert 'Retrieved Text:\n"Full-time employees receive 24 days of annual leave."' in formatted


def test_citation_extraction_and_deduplication():
    """Verify that citations are extracted and deduplicated while preserving order."""
    results = [
        SearchResult("c1", "guide.pdf", 1, "First snippet content", 0.90),
        SearchResult("c2", "guide.pdf", 1, "First snippet content", 0.88),  # Duplicate
        SearchResult("c3", "guide.pdf", 2, "Second snippet content", 0.80),
    ]

    citations = CitationFormatter.extract_citations(results)

    assert len(citations) == 2
    assert citations[0].document_name == "guide.pdf"
    assert citations[0].page_number == 1
    assert citations[0].retrieved_text == "First snippet content"
    assert citations[1].page_number == 2
    assert citations[1].retrieved_text == "Second snippet content"


def test_citation_empty_sources_handled():
    """Verify that empty sources produce empty list and empty string."""
    citations = CitationFormatter.extract_citations([])
    formatted_str = CitationFormatter.format_citations_string([])

    assert citations == []
    assert formatted_str == ""
