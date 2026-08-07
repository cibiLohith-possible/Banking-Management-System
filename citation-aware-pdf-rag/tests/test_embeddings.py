"""Unit tests for embedding generation module."""

from typing import Any
import pytest

from app.embeddings import EmbeddingModel


@pytest.fixture(scope="session")
def embedding_model():
    """Session-scoped fixture to initialize the EmbeddingModel once for all tests."""
    return EmbeddingModel()


def test_model_initialization(embedding_model: EmbeddingModel):
    """Verify that the embedding model initializes correctly."""
    assert embedding_model is not None
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


def test_embedding_dimension_is_correct(embedding_model: EmbeddingModel):
    """Verify that all-MiniLM-L6-v2 produces vectors of dimension 384."""
    assert embedding_model.dimension == 384


def test_embed_single_text(embedding_model: EmbeddingModel):
    """Verify single text produces a list of floats matching the model dimension."""
    text = "Employees receive annual leave."
    embedding = embedding_model.embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) == embedding_model.dimension
    assert all(isinstance(val, float) for val in embedding)


def test_embed_batch_texts(embedding_model: EmbeddingModel):
    """Verify batch processing produces equal number of embeddings as inputs."""
    texts = [
        "Employees receive annual leave.",
        "Working hours are from 9 AM to 6 PM.",
    ]
    embeddings = embedding_model.embed_batch(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == embedding_model.dimension
    assert len(embeddings[1]) == embedding_model.dimension
    assert all(isinstance(val, float) for val in embeddings[0])
    assert all(isinstance(val, float) for val in embeddings[1])


def test_input_ordering_is_preserved(embedding_model: EmbeddingModel):
    """Verify that batch embedding preserves the exact input order."""
    text1 = "First text content."
    text2 = "Second text content."

    batch_embeddings = embedding_model.embed_batch([text1, text2])
    single_embedding1 = embedding_model.embed_text(text1)
    single_embedding2 = embedding_model.embed_text(text2)

    assert batch_embeddings[0] == pytest.approx(single_embedding1, abs=1e-5)
    assert batch_embeddings[1] == pytest.approx(single_embedding2, abs=1e-5)


def test_empty_batch_handled_cleanly(embedding_model: EmbeddingModel):
    """Verify that an empty list/sequence returns an empty list."""
    assert embedding_model.embed_batch([]) == []


def test_invalid_single_text_type_rejected(embedding_model: EmbeddingModel):
    """Verify non-string single text raises TypeError."""
    invalid_inputs: list[Any] = [123, None, ["text"], {"key": "val"}]
    for invalid in invalid_inputs:
        with pytest.raises(TypeError, match="must be a string"):
            embedding_model.embed_text(invalid)


def test_invalid_batch_element_type_rejected(embedding_model: EmbeddingModel):
    """Verify non-string elements inside batch list raise TypeError."""
    invalid_batch = ["Valid string", 12345, "Another valid string"]  # type: ignore
    with pytest.raises(TypeError, match="must be a string"):
        embedding_model.embed_batch(invalid_batch)


def test_single_string_passed_to_embed_batch_rejected(embedding_model: EmbeddingModel):
    """Verify passing a raw string instead of a sequence of strings to embed_batch raises TypeError."""
    with pytest.raises(TypeError):
        embedding_model.embed_batch("not a list of strings")  # type: ignore
