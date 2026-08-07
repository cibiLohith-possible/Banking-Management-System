"""End-to-end RAG pipeline connecting retriever, LLM, and citation formatter."""

from typing import Any, Sequence

from app.citation import Citation, CitationFormatter
from app.llm import NO_INFO_FALLBACK, OpenRouterLLM
from app.retriever import Retriever
from app.qdrant_store import SearchResult


class RAGPipeline:
    """Orchestrates end-to-end question answering with ground truth citations."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: OpenRouterLLM | None = None,
        citation_formatter: CitationFormatter | None = None,
    ) -> None:
        """Initialize the RAGPipeline with required components.

        Args:
            retriever: Optional Retriever instance.
            llm: Optional OpenRouterLLM instance.
            citation_formatter: Optional CitationFormatter instance.
        """
        self.retriever = retriever or Retriever()
        self.llm = llm or OpenRouterLLM()
        self.citation_formatter = citation_formatter or CitationFormatter()

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Execute RAG flow for a natural-language question.

        Flow:
            1. Validate question.
            2. Retrieve relevant chunks from Qdrant.
            3. If no relevant chunks: Return fallback response immediately without calling LLM.
            4. If relevant chunks exist: Query OpenRouter free model with grounded prompt.
            5. Generate citations independently from Qdrant metadata.

        Args:
            question: User question string.
            top_k: Maximum number of context chunks to retrieve.
            min_score: Minimum similarity score threshold for retrieved context.

        Returns:
            Dictionary containing answer, sources list, and raw retrieved_chunks.
        """
        if not isinstance(question, str) or not question.strip():
            return {
                "answer": NO_INFO_FALLBACK,
                "sources": [],
                "retrieved_chunks": [],
            }

        # 1. Retrieve relevant context chunks
        try:
            results = self.retriever.retrieve(
                query=question, top_k=top_k, min_score=min_score
            )
        except Exception:
            return {
                "answer": NO_INFO_FALLBACK,
                "sources": [],
                "retrieved_chunks": [],
            }

        # 2. Early exit if no context retrieved
        if not results:
            return {
                "answer": NO_INFO_FALLBACK,
                "sources": [],
                "retrieved_chunks": [],
            }

        # 3. Generate answer using OpenRouter
        answer = self.llm.generate(question=question, context_chunks=results)

        # 4. Generate citations independently from Qdrant metadata
        citations = self.citation_formatter.extract_citations(results)
        sources = [c.to_dict() for c in citations]

        # Format raw chunks info
        retrieved_chunks = [
            {
                "chunk_id": r.chunk_id,
                "document_name": r.document_name,
                "page_number": r.page_number,
                "text": r.text,
                "score": r.score,
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }
