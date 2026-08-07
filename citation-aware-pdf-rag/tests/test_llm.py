"""Unit tests for OpenRouter LLM module."""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.llm import NO_INFO_FALLBACK, OpenRouterLLM
from app.qdrant_store import SearchResult


def test_llm_prompt_construction():
    """Verify that prompt incorporates question and formatted context blocks correctly."""
    llm = OpenRouterLLM(api_key="test_key", model="meta-llama/llama-3.3-70b-instruct:free")
    context = [
        SearchResult("c1", "handbook.pdf", 12, "Annual leave is 20 days.", 0.88),
    ]

    prompt = llm.build_prompt("How many leave days?", context)

    assert "You are a document question-answering assistant." in prompt
    assert "Document: handbook.pdf" in prompt
    assert "Page: 12" in prompt
    assert "Annual leave is 20 days." in prompt
    assert "Question: How many leave days?" in prompt


def test_llm_generate_success():
    """Verify successful OpenRouter API call using mock HTTP response."""
    llm = OpenRouterLLM(api_key="mock_api_key", model="meta-llama/llama-3.3-70b-instruct:free")
    context = [
        SearchResult("c1", "handbook.pdf", 12, "Annual leave is 20 days.", 0.88),
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Employees get 20 days of annual leave.",
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        answer = llm.generate("How many leave days?", context)

        assert answer == "Employees get 20 days of annual leave."
        mock_post.assert_called_once()
        # Verify API key is passed in headers
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer mock_api_key"


def test_llm_missing_api_key_returns_fallback():
    """Verify that missing API key returns fallback message cleanly."""
    llm = OpenRouterLLM(api_key="")
    context = [SearchResult("c1", "handbook.pdf", 1, "Text", 0.9)]

    answer = llm.generate("Question?", context)
    assert NO_INFO_FALLBACK in answer
    assert "missing" in answer.lower() or "invalid" in answer.lower()


def test_llm_empty_context_returns_fallback_without_api_call():
    """Verify that empty context returns fallback immediately without calling LLM HTTP API."""
    llm = OpenRouterLLM(api_key="valid_key")

    with patch("httpx.Client.post") as mock_post:
        answer = llm.generate("Any question?", context_chunks=[])
        assert answer == NO_INFO_FALLBACK
        mock_post.assert_not_called()


def test_llm_http_timeout_returns_fallback():
    """Verify that HTTP timeout returns fallback message cleanly."""
    llm = OpenRouterLLM(api_key="valid_key")
    context = [SearchResult("c1", "handbook.pdf", 1, "Text", 0.9)]

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timeout")):
        answer = llm.generate("Question?", context)
        assert NO_INFO_FALLBACK in answer
        assert "timed out" in answer.lower()


def test_llm_http_error_returns_fallback():
    """Verify that HTTP error response returns fallback message cleanly."""
    llm = OpenRouterLLM(api_key="valid_key")
    context = [SearchResult("c1", "handbook.pdf", 1, "Text", 0.9)]

    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("httpx.Client.post", side_effect=httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_resp)):
        answer = llm.generate("Question?", context)
        assert NO_INFO_FALLBACK in answer
