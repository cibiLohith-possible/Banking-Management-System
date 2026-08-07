"""Tests for Qdrant vector database integration.

Tests require a running Qdrant instance at http://localhost:6333.
If Qdrant is not available, all tests are skipped with a clear message.
"""

import pytest

from app.qdrant_store import QdrantStore, SearchResult

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "test_pdf_rag_integration"
EMBEDDING_DIM = 8  # Small dimension for fast in-test vectors


def _make_vector(val: float = 1.0, dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a normalised unit vector of given dimension."""
    raw = [val] * dim
    magnitude = sum(x ** 2 for x in raw) ** 0.5
    return [x / magnitude for x in raw]


def _qdrant_available() -> bool:
    """Return True if a local Qdrant instance is reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False)
        client.get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Local Qdrant not available at http://localhost:6333 — skipping integration tests",
)


@pytest.fixture(scope="module")
def store():
    """Create a QdrantStore backed by a dedicated test collection.

    Deletes the test collection before and after the test module runs so
    repeated test runs are always clean.
    """
    s = QdrantStore(
        embedding_dimension=EMBEDDING_DIM,
        url=QDRANT_URL,
        api_key=None,
        collection_name=TEST_COLLECTION,
    )
    # Clean slate before tests
    s.delete_collection()
    s.create_collection()
    yield s
    # Teardown: remove test collection
    s.delete_collection()


SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc.pdf_p1_c0",
        "document_name": "doc.pdf",
        "page_number": 1,
        "text": "Employees receive 20 days of annual leave.",
    },
    {
        "chunk_id": "doc.pdf_p2_c0",
        "document_name": "doc.pdf",
        "page_number": 2,
        "text": "Working hours are from 9 AM to 6 PM.",
    },
]

SAMPLE_VECTORS = [
    _make_vector(1.0),
    _make_vector(0.5),
]

# ---------------------------------------------------------------------------
# Tests — initialization
# ---------------------------------------------------------------------------


def test_qdrant_store_initialization_valid():
    """QdrantStore can be created with a valid embedding dimension."""
    # We only test construction, not live connection.
    store = QdrantStore(
        embedding_dimension=384,
        url=QDRANT_URL,
        api_key=None,
        collection_name="dummy",
    )
    assert store.embedding_dimension == 384
    assert store.collection_name == "dummy"


def test_qdrant_store_invalid_dimension():
    """QdrantStore raises ValueError for non-positive embedding dimension."""
    with pytest.raises(ValueError, match="embedding_dimension must be a positive integer"):
        QdrantStore(embedding_dimension=0)

    with pytest.raises(ValueError, match="embedding_dimension must be a positive integer"):
        QdrantStore(embedding_dimension=-1)

    with pytest.raises(ValueError, match="embedding_dimension must be a positive integer"):
        QdrantStore(embedding_dimension="384")  # type: ignore


# ---------------------------------------------------------------------------
# Tests — collection management (require live Qdrant)
# ---------------------------------------------------------------------------


@requires_qdrant
def test_collection_is_created(store: QdrantStore):
    """Collection should exist after create_collection is called."""
    assert store.collection_exists() is True


@requires_qdrant
def test_collection_can_be_checked(store: QdrantStore):
    """collection_exists returns False for a non-existent collection name."""
    ghost = QdrantStore(
        embedding_dimension=EMBEDDING_DIM,
        url=QDRANT_URL,
        collection_name="nonexistent_xyz_abc_99",
    )
    assert ghost.collection_exists() is False


@requires_qdrant
def test_collection_recreate(store: QdrantStore):
    """Recreating the collection does not raise and leaves the collection present."""
    store.create_collection(recreate=True)
    assert store.collection_exists() is True


# ---------------------------------------------------------------------------
# Tests — insertion
# ---------------------------------------------------------------------------


@requires_qdrant
def test_insert_returns_correct_count(store: QdrantStore):
    """insert() returns the number of points upserted."""
    count = store.insert(SAMPLE_CHUNKS, SAMPLE_VECTORS)
    assert count == 2


@requires_qdrant
def test_empty_insert_returns_zero(store: QdrantStore):
    """Inserting an empty list returns 0 without error."""
    count = store.insert([], [])
    assert count == 0


@requires_qdrant
def test_insert_mismatched_lengths_raises(store: QdrantStore):
    """insert() raises ValueError when chunks and embeddings differ in length."""
    with pytest.raises(ValueError, match="must match embeddings length"):
        store.insert(SAMPLE_CHUNKS, [_make_vector()])  # only 1 vector for 2 chunks


@requires_qdrant
def test_insert_missing_metadata_raises(store: QdrantStore):
    """insert() raises ValueError when a chunk is missing required metadata."""
    bad_chunk = [{"chunk_id": "x", "text": "oops"}]  # missing document_name, page_number
    with pytest.raises(ValueError, match="missing required fields"):
        store.insert(bad_chunk, [_make_vector()])


@requires_qdrant
def test_insert_wrong_vector_dimension_raises(store: QdrantStore):
    """insert() raises ValueError when a vector has the wrong dimension."""
    wrong_vector = [[1.0, 2.0, 3.0]]  # dim=3 instead of EMBEDDING_DIM
    with pytest.raises(ValueError, match="dimension"):
        store.insert(SAMPLE_CHUNKS[:1], wrong_vector)


# ---------------------------------------------------------------------------
# Tests — search
# ---------------------------------------------------------------------------


@requires_qdrant
def test_search_returns_results(store: QdrantStore):
    """search() returns a non-empty list for a valid query."""
    results = store.search(query_vector=_make_vector(1.0), top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0


@requires_qdrant
def test_search_result_has_correct_fields(store: QdrantStore):
    """Each SearchResult contains score and all required citation metadata."""
    results = store.search(query_vector=_make_vector(1.0), top_k=1)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, SearchResult)
    assert isinstance(r.score, float)
    assert isinstance(r.document_name, str)
    assert isinstance(r.page_number, int)
    assert isinstance(r.text, str)
    assert isinstance(r.chunk_id, str)


@requires_qdrant
def test_search_metadata_preserved(store: QdrantStore):
    """search() returns results whose metadata matches what was inserted."""
    results = store.search(query_vector=_make_vector(1.0), top_k=5)
    found_chunk_ids = {r.chunk_id for r in results}
    # Both inserted chunk_ids should appear in the results
    assert "doc.pdf_p1_c0" in found_chunk_ids
    assert "doc.pdf_p2_c0" in found_chunk_ids


@requires_qdrant
def test_search_top_k_respected(store: QdrantStore):
    """search() returns at most top_k results."""
    results = store.search(query_vector=_make_vector(1.0), top_k=1)
    assert len(results) <= 1


@requires_qdrant
def test_search_invalid_top_k_raises(store: QdrantStore):
    """search() raises ValueError for top_k <= 0."""
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.search(query_vector=_make_vector(1.0), top_k=0)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.search(query_vector=_make_vector(1.0), top_k=-5)


@requires_qdrant
def test_search_wrong_dimension_raises(store: QdrantStore):
    """search() raises ValueError when the query vector has the wrong dimension."""
    wrong_dim_vector = [0.1, 0.2, 0.3]  # dim=3 instead of EMBEDDING_DIM
    with pytest.raises(ValueError, match="dimension"):
        store.search(query_vector=wrong_dim_vector)
