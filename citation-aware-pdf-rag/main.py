"""FastAPI Web Service for Citation-Aware PDF RAG."""

from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.llm import NO_INFO_FALLBACK
from app.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Citation-Aware PDF RAG",
    description="Retrieval-Augmented Generation system with guaranteed citation tracking.",
    version="1.0.0",
)

# Global pipeline instance (initialized lazily to avoid heavy loading on import)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Dependency helper to get or initialize the RAGPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# --- Pydantic Schemas ---


class AskRequest(BaseModel):
    """Request model for /ask endpoint."""

    question: str = Field(
        ...,
        description="Natural language question to ask about the ingested documents.",
        examples=["What is the leave policy?"],
    )


class SourceCitation(BaseModel):
    """Source citation model."""

    document_name: str = Field(..., description="Filename of the source PDF document.")
    page_number: int = Field(..., description="1-based page number where context was found.")
    retrieved_text: str = Field(..., description="Exact text snippet retrieved from context.")


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""

    answer: str = Field(..., description="Generated answer grounded in document context.")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="List of verified document sources supporting the answer.",
    )


# --- Routes ---


@app.get("/", tags=["Info"])
def read_root() -> dict[str, str]:
    """Root endpoint returning application description."""
    return {
        "name": "Citation-Aware PDF RAG",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation API providing verified PDF citations.",
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    tags=["Q&A"],
)
def ask_question(request: AskRequest) -> AskResponse:
    """Process a user question, retrieve document context, and return citation-backed answer."""
    question_text = request.question.strip() if request.question else ""

    if not question_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty or whitespace-only.",
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.query(question=question_text)

        sources = [
            SourceCitation(
                document_name=s["document_name"],
                page_number=s["page_number"],
                retrieved_text=s["retrieved_text"],
            )
            for s in result.get("sources", [])
        ]

        return AskResponse(
            answer=result.get("answer", NO_INFO_FALLBACK),
            sources=sources,
        )

    except HTTPException:
        raise
    except Exception:
        # Catch unexpected failures safely without revealing internal stack traces
        return AskResponse(
            answer=NO_INFO_FALLBACK,
            sources=[],
        )
