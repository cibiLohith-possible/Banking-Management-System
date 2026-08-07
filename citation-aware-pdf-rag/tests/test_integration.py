"""End-to-end integration tests for Citation-Aware PDF RAG."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from app.chunker import TextChunker
from app.embeddings import EmbeddingModel
from app.pdf_loader import PDFLoader
from app.qdrant_store import QdrantStore, SearchResult
from app.rag_pipeline import RAGPipeline
from app.retriever import Retriever


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Generate a 2-page sample PDF dynamically with PyMuPDF."""
    pdf_path = tmp_path / "employee_policy.pdf"
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "All employees receive 25 annual leave days per year.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "Standard work hours are 9:00 AM to 5:00 PM Monday through Friday.")

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_full_pipeline_flow(sample_pdf: Path):
    """Verify PDF loading -> chunking -> embedding -> retrieval -> RAG pipeline."""
    # 1. Load PDF
    loader = PDFLoader(sample_pdf)
    pages = loader.load()
    assert len(pages) == 2
    assert pages[0].document_name == "employee_policy.pdf"

    # 2. Chunk Pages
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) == 2

    # 3. Embedding
    embed_model = EmbeddingModel()
    texts = [c.text for c in chunks]
    embeddings = embed_model.embed_batch(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == embed_model.dimension

    # 4. RAG Pipeline with mocked LLM response
    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = [
        SearchResult(
            chunk_id=chunks[0].chunk_id,
            document_name=chunks[0].document_name,
            page_number=chunks[0].page_number,
            text=chunks[0].text,
            score=0.92,
        )
    ]

    retriever = Retriever(embedding_model=embed_model, qdrant_store=mock_qdrant)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Employees receive 25 days of annual leave."

    pipeline = RAGPipeline(retriever=retriever, llm=mock_llm)

    result = pipeline.query("How many leave days do employees get?")

    # Verify result structure & citation metadata
    assert result["answer"] == "Employees receive 25 days of annual leave."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_name"] == "employee_policy.pdf"
    assert result["sources"][0]["page_number"] == 1
    assert "25 annual leave days" in result["sources"][0]["retrieved_text"]
