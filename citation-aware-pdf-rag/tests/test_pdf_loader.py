"""Tests for PDF loader module using PyMuPDF."""

from pathlib import Path

import fitz
import pytest

from app.pdf_loader import PDFLoader, PDFPage


@pytest.fixture
def create_sample_pdf(tmp_path: Path):
    """Fixture to programmatically generate test PDF files using PyMuPDF."""

    def _generator(
        filename: str = "sample.pdf",
        pages_content: list[str] | None = None,
    ) -> Path:
        if pages_content is None:
            pages_content = ["Page 1 content", "Page 2 content", "Page 3 content"]

        pdf_path = tmp_path / filename
        doc = fitz.open()

        for content in pages_content:
            page = doc.new_page()
            if content:  # Insert text if non-empty; leave empty otherwise
                page.insert_text((50, 50), content)

        doc.save(pdf_path)
        doc.close()
        return pdf_path

    return _generator


def test_load_single_and_multiple_pages(create_sample_pdf):
    """Verify that multiple pages in a PDF can be loaded into PDFPage objects."""
    pdf_path = create_sample_pdf("multi_page.pdf", ["First page text", "Second page text", "Third page text"])
    loader = PDFLoader(pdf_path)
    pages = loader.load()

    assert len(pages) == 3
    assert all(isinstance(p, PDFPage) for p in pages)
    assert pages[0].text == "First page text"
    assert pages[1].text == "Second page text"
    assert pages[2].text == "Third page text"


def test_document_filename_preserved(create_sample_pdf):
    """Verify that document filename metadata is preserved."""
    pdf_path = create_sample_pdf("document_alpha.pdf", ["Some text content"])
    loader = PDFLoader(pdf_path)
    pages = loader.load()

    assert len(pages) == 1
    assert pages[0].document_name == "document_alpha.pdf"


def test_page_numbers_are_one_based(create_sample_pdf):
    """Verify that page numbers start from 1 instead of 0."""
    pdf_path = create_sample_pdf("pages.pdf", ["Page A", "Page B", "Page C"])
    loader = PDFLoader(pdf_path)
    pages = loader.load()

    assert [p.page_number for p in pages] == [1, 2, 3]


def test_empty_pages_are_skipped(create_sample_pdf):
    """Verify that completely empty pages are skipped while preserving correct 1-based page numbers."""
    pdf_path = create_sample_pdf("with_empty_page.pdf", ["Page 1 text", "", "Page 3 text"])
    loader = PDFLoader(pdf_path)
    pages = loader.load()

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].text == "Page 1 text"
    assert pages[1].page_number == 3
    assert pages[1].text == "Page 3 text"


def test_missing_file_raises_file_not_found_error(tmp_path: Path):
    """Verify that providing a non-existent path raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        PDFLoader(non_existent)


def test_non_pdf_file_extension_raises_value_error(tmp_path: Path):
    """Verify that non-PDF file extension raises ValueError."""
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("Hello world")

    with pytest.raises(ValueError, match="Specified file is not a PDF"):
        PDFLoader(txt_file)


def test_corrupted_pdf_file_raises_value_error(tmp_path: Path):
    """Verify that invalid/corrupted PDF file contents raise ValueError upon loading."""
    fake_pdf = tmp_path / "corrupted.pdf"
    fake_pdf.write_bytes(b"Not a real PDF stream content")

    loader = PDFLoader(fake_pdf)
    with pytest.raises(ValueError, match="Invalid or corrupted PDF file"):
        loader.load()
