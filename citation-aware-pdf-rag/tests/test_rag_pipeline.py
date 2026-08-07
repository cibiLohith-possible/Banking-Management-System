"""Unit tests for complete RAG Pipeline."""

from unittest.mock import MagicMock
import pytest

from app.llm import NO_INFO_FALLBACK
from app.rag_pipeline import RAGPipeline
from app.qdrant_store import SearchResult


def test_rag_pipeline_known_question_flow():
    """Verify full RAG flow when relevant context exists."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        SearchResult("c1", "policy.pdf", 3, "Leave is 20 days per year.", 0.85)
    ]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Employees are entitled to 20 days of leave per year."

    pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm)

    result = pipeline.query("How much leave is provided?")

    assert result["answer"] == "Employees are entitled to 20 days of leave per year."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_name"] == "policy.pdf"
    assert result["sources"][0]["page_number"] == 3
    assert result["sources"][0]["retrieved_text"] == "Leave is 20 days per year."
    mock_retriever.retrieve.assert_called_once()
    mock_llm.generate.assert_called_once()


def test_rag_pipeline_unknown_question_bypasses_llm():
    """Verify that when no relevant chunks match, LLM is NOT called and fallback is returned."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []  # No chunks retrieved

    mock_llm = MagicMock()

    pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm)

    result = pipeline.query("What is the rocket launch schedule?")

    assert result["answer"] == NO_INFO_FALLBACK
    assert result["sources"] == []
    assert result["retrieved_chunks"] == []
    mock_retriever.retrieve.assert_called_once()
    # Critical requirement: LLM must NOT be called for unknown questions!
    mock_llm.generate.assert_not_called()


def test_rag_pipeline_empty_question_bypasses_llm():
    """Verify that empty or whitespace query returns fallback immediately without retrieval or LLM calls."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm)

    result = pipeline.query("   \n  ")

    assert result["answer"] == NO_INFO_FALLBACK
    assert result["sources"] == []
    mock_retriever.retrieve.assert_not_called()
    mock_llm.generate.assert_not_called()
