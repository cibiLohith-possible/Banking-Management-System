"""PDF document loading and extraction module using PyMuPDF."""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PDFPage:
    """Represents an extracted page from a PDF document.

    Attributes:
        document_name: The filename of the source PDF.
        page_number: The 1-based index of the page in the PDF.
        text: The extracted text content of the page.
    """

    document_name: str
    page_number: int
    text: str


class PDFLoader:
    """Loads PDF documents and extracts page content with metadata using PyMuPDF."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the PDFLoader with a file path.

        Args:
            file_path: Path to the target PDF file.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            ValueError: If the path is not a file or does not have a .pdf extension.
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Specified path is not a file: {self.file_path}")

        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Specified file is not a PDF: {self.file_path}")

    def load(self) -> list[PDFPage]:
        """Extract text from each non-empty page of the PDF.

        Returns:
            A list of PDFPage instances containing document_name, 1-based page_number,
            and extracted text for non-empty pages.

        Raises:
            ValueError: If the file cannot be opened or parsed as a valid PDF.
        """
        pages: list[PDFPage] = []
        doc_name = self.file_path.name

        try:
            with fitz.open(self.file_path) as doc:
                if not doc.is_pdf and doc.page_count == 0:
                    raise ValueError(f"File could not be opened as a valid PDF: {self.file_path}")

                for page_idx in range(doc.page_count):
                    page = doc.load_page(page_idx)
                    text = page.get_text("text", sort=True).strip()

                    # Skip completely empty pages
                    if not text:
                        continue

                    # 1-based page numbering
                    page_number = page_idx + 1

                    pages.append(
                        PDFPage(
                            document_name=doc_name,
                            page_number=page_number,
                            text=text,
                        )
                    )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid or corrupted PDF file '{self.file_path}': {e}") from e

        return pages
