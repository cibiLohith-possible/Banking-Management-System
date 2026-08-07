"""Unit tests for semantic retriever module."""

from unittest.mock import MagicMock
import pytest

from app.qdrant_store import SearchResult
from app.retriever import Retriever


def test_retriever_question_type_validation():
    """Verify that non-string queries raise TypeError."""
    retriever = Retriever()
    with pytest.raises(TypeError, match="Query must be a string"):
        retriever.retrieve(query=123)  # type: ignore


def test_retriever_empty_question_validation():
    """Verify that empty or whitespace-only queries raise ValueError."""
    retriever = Retriever()
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve(query="   \n  ")


def test_retriever_top_k_validation():
    """Verify that top_k <= 0 raises ValueError."""
    retriever = Retriever()
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.retrieve(query="valid query", top_k=0)


def test_retriever_search_and_threshold_filtering():
    """Verify embedding generation, search delegation, and score threshold filtering."""
    mock_embed = MagicMock()
    mock_embed.embed_text.return_value = [0.1] * 384
    mock_embed.dimension = 384

    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = [
        SearchResult("c1", "doc.pdf", 1, "High relevance text", 0.85),
        SearchResult("c2", "doc.pdf", 2, "Medium relevance text", 0.65),
        SearchResult("c3", "doc.pdf", 3, "Low relevance text", 0.25),
    ]

    retriever = Retriever(embedding_model=mock_embed, qdrant_store=mock_qdrant)

    # Filter with min_score = 0.50
    results = retriever.retrieve("What is the policy?", top_k=5, min_score=0.50)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[1].chunk_id == "c2"
    mock_embed.embed_text.assert_called_once_with("What is the policy?")
    mock_qdrant.search.assert_called_once_with(query_vector=[0.1] * 384, top_k=5)


def test_retriever_returns_empty_when_no_result_meets_threshold():
    """Verify empty list is returned when all results fall below threshold."""
    mock_embed = MagicMock()
    mock_embed.embed_text.return_value = [0.1] * 384
    mock_embed.dimension = 384

    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = [
        SearchResult("c1", "doc.pdf", 1, "Irrelevant snippet", 0.15),
    ]

    retriever = Retriever(embedding_model=mock_embed, qdrant_store=mock_qdrant)
    results = retriever.retrieve("Specific query", min_score=0.50)

    assert results == []
