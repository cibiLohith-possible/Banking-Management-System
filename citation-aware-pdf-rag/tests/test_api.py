"""Unit tests for FastAPI endpoints in main.py."""

from unittest.mock import MagicMock, patch
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
import pytest

from main import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


def test_get_root(client: TestClient):
    """Verify GET / returns application metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "description" in data


def test_get_health(client: TestClient):
    """Verify GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_ask_valid_question(client: TestClient):
    """Verify POST /ask processes valid questions and returns expected schema."""
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = {
        "answer": "Working hours are 9 AM to 6 PM.",
        "sources": [
            {
                "document_name": "handbook.pdf",
                "page_number": 5,
                "retrieved_text": "Working hours are 9 AM to 6 PM.",
            }
        ],
    }

    with patch("main.get_pipeline", return_value=mock_pipeline):
        response = client.post("/ask", json={"question": "What are the working hours?"})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == "Working hours are 9 AM to 6 PM."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_name"] == "handbook.pdf"
        assert data["sources"][0]["page_number"] == 5


def test_post_ask_empty_question_returns_400(client: TestClient):
    """Verify POST /ask returns 400 Bad Request when question is empty."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()
