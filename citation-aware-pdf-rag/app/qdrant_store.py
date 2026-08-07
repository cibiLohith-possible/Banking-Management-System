import uuid
from dataclasses import dataclass
from typing import Any

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.http import models

from app import config


@dataclass
class SearchResult:
    """Represents a single search result returned from Qdrant.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        document_name: Source PDF filename.
        page_number: 1-based page number of the source page.
        text: The chunk text content.
        score: Cosine similarity score in range [0.0, 1.0].
    """

    chunk_id: str
    document_name: str
    page_number: int
    text: str
    score: float


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert a chunk_id string into a deterministic UUID-compatible string.

    Qdrant point IDs must be either unsigned integers or UUID strings.
    We derive a deterministic UUID v5 from the chunk_id using the DNS namespace.

    Args:
        chunk_id: The string chunk identifier from the chunker.

    Returns:
        A deterministic UUID string derived from chunk_id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantStore:
    """Manages vector storage and retrieval in Qdrant.

    Supports both local Qdrant (http://localhost:6333) and
    Qdrant Cloud (via QDRANT_URL + QDRANT_API_KEY env vars).
    """

    def __init__(
        self,
        embedding_dimension: int,
        url: str = config.QDRANT_URL,
        api_key: str | None = config.QDRANT_API_KEY,
        collection_name: str = config.QDRANT_COLLECTION,
    ) -> None:
        """Initialize QdrantStore and establish connection.

        Args:
            embedding_dimension: The dimensionality of the embedding vectors.
            url: Qdrant server URL (default from config/env).
            api_key: Optional Qdrant Cloud API key (default from config/env).
            collection_name: Name of the Qdrant collection.

        Raises:
            ValueError: If embedding_dimension is not a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be a positive integer, got {embedding_dimension!r}"
            )

        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name

        self._client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """Check whether the configured collection exists in Qdrant.

        Returns:
            True if the collection exists, False otherwise.
        """
        existing = self._client.get_collections().collections
        return any(c.name == self.collection_name for c in existing)

    def create_collection(self, *, recreate: bool = False) -> None:
        """Create the Qdrant collection if it does not already exist.

        Args:
            recreate: If True, delete any existing collection with the same
                      name before creating a fresh one.

        Raises:
            RuntimeError: If collection creation fails unexpectedly.
        """
        if recreate and self.collection_exists():
            self._client.delete_collection(self.collection_name)

        if not self.collection_exists():
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def delete_collection(self) -> None:
        """Delete the configured collection if it exists."""
        if self.collection_exists():
            self._client.delete_collection(self.collection_name)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def insert(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        """Insert chunks and their embeddings into Qdrant.

        Each chunk dict must contain:
            - chunk_id (str)
            - document_name (str)
            - page_number (int)
            - text (str)

        Args:
            chunks: List of chunk metadata dicts.
            embeddings: Corresponding embedding vectors (same length as chunks).

        Returns:
            Number of points successfully upserted.

        Raises:
            ValueError: If chunks and embeddings differ in length,
                        required metadata fields are missing,
                        or a vector has the wrong dimension.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks length ({len(chunks)}) must match embeddings length ({len(embeddings)})"
            )

        required_fields = {"chunk_id", "document_name", "page_number", "text"}
        points: list[models.PointStruct] = []

        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            missing = required_fields - chunk.keys()
            if missing:
                raise ValueError(
                    f"Chunk at index {idx} is missing required fields: {missing}"
                )

            if len(vector) != self.embedding_dimension:
                raise ValueError(
                    f"Vector at index {idx} has dimension {len(vector)}, "
                    f"expected {self.embedding_dimension}"
                )

            point_id = _chunk_id_to_point_id(chunk["chunk_id"])

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "document_name": chunk["document_name"],
                        "page_number": chunk["page_number"],
                        "text": chunk["text"],
                    },
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Find the top-k most similar chunks for a given query vector.

        Args:
            query_vector: The embedding vector of the search query.
            top_k: Number of results to return (must be > 0).

        Returns:
            List of SearchResult instances ordered by descending similarity score.

        Raises:
            ValueError: If top_k <= 0 or query_vector dimension is wrong.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        if len(query_vector) != self.embedding_dimension:
            raise ValueError(
                f"query_vector dimension {len(query_vector)} does not match "
                f"collection dimension {self.embedding_dimension}"
            )

        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    document_name=payload.get("document_name", ""),
                    page_number=int(payload.get("page_number", 0)),
                    text=payload.get("text", ""),
                    score=float(hit.score),
                )
            )
        return results
