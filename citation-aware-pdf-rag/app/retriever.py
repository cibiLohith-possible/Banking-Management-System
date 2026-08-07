"""Semantic retrieval module connecting query embedding generation and Qdrant search."""

from typing import Sequence

from app.embeddings import EmbeddingModel
from app.qdrant_store import QdrantStore, SearchResult


class Retriever:
    """Retrieves contextually relevant document chunks for a natural-language query."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        qdrant_store: QdrantStore | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Initialize the Retriever.

        Args:
            embedding_model: Optional EmbeddingModel instance. If None, one will be created.
            qdrant_store: Optional QdrantStore instance. If None, one will be created using model dimension.
            min_score: Minimum similarity score threshold [0.0, 1.0] for returned results.
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.qdrant_store = qdrant_store or QdrantStore(
            embedding_dimension=self.embedding_model.dimension
        )
        self.min_score = max(0.0, min_score)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks for a user query string.

        Args:
            query: Non-empty search query string.
            top_k: Maximum number of results to return (must be > 0).
            min_score: Optional override for minimum similarity score threshold.

        Returns:
            List of SearchResult instances sorted by descending score that satisfy min_score.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or top_k <= 0.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}")

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        threshold = self.min_score if min_score is None else max(0.0, min_score)

        # 1. Embed query
        query_vector = self.embedding_model.embed_text(clean_query)

        # 2. Search Qdrant
        results = self.qdrant_store.search(query_vector=query_vector, top_k=top_k)

        # 3. Filter by similarity threshold
        filtered_results = [r for r in results if r.score >= threshold]

        return filtered_results
