"""OpenRouter LLM client integration for grounded question answering."""

import os
from typing import Any, Sequence

import httpx

from app import config
from app.qdrant_store import SearchResult

NO_INFO_FALLBACK = "The information is not available in the supplied documents."


class OpenRouterLLM:
    """Interfaces with OpenRouter API using free models for strictly grounded QA."""

    DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize OpenRouterLLM client.

        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY / env).
            model: OpenRouter free model name (defaults to config.OPENROUTER_MODEL / env).
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or getattr(
            config, "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
        )
        self.timeout = timeout

    def build_prompt(self, question: str, context_chunks: Sequence[SearchResult | dict[str, Any]]) -> str:
        """Construct the grounded prompt incorporating retrieved context.

        Args:
            question: The user's query.
            context_chunks: List of retrieved SearchResult objects or dicts.

        Returns:
            Formatted prompt string.
        """
        formatted_blocks: list[str] = []
        for idx, chunk in enumerate(context_chunks, 1):
            if isinstance(chunk, SearchResult):
                doc_name = chunk.document_name
                page_num = chunk.page_number
                text = chunk.text
                score = chunk.score
            else:
                doc_name = chunk.get("document_name", "Unknown")
                page_num = chunk.get("page_number", 0)
                text = chunk.get("text", "")
                score = chunk.get("score", 0.0)

            formatted_blocks.append(
                f"[Source {idx}]\n"
                f"Document: {doc_name}\n"
                f"Page: {page_num}\n"
                f"Relevance Score: {score:.4f}\n"
                f"Content:\n{text}\n"
            )

        context_str = "\n---\n".join(formatted_blocks)

        prompt = (
            "You are a document question-answering assistant.\n"
            "Answer ONLY using the supplied context.\n"
            "Do not use outside knowledge.\n"
            "Do not make assumptions.\n"
            "Do not fabricate information.\n"
            "If the answer cannot be found in the supplied context, respond:\n"
            f"{NO_INFO_FALLBACK}\n\n"
            f"=== CONTEXT ===\n"
            f"{context_str}\n"
            f"=== END CONTEXT ===\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt

    def generate(
        self,
        question: str,
        context_chunks: Sequence[SearchResult | dict[str, Any]],
    ) -> str:
        """Generate a grounded answer for the user question given retrieved context.

        Args:
            question: The user query string.
            context_chunks: Retrieved document context chunks.

        Returns:
            Clean answer string or fallback response if information is not found.
        """
        if not context_chunks:
            return NO_INFO_FALLBACK

        if not self.api_key:
            # Handle missing API key safely without revealing secrets
            return (
                f"{NO_INFO_FALLBACK} (OpenRouter API key is missing or invalid)"
            )

        prompt = self.build_prompt(question, context_chunks)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/citation-aware-pdf-rag",
            "X-Title": "Citation-Aware PDF RAG",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.DEFAULT_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    return NO_INFO_FALLBACK

                content = choices[0].get("message", {}).get("content", "").strip()
                return content if content else NO_INFO_FALLBACK

        except httpx.TimeoutException:
            return f"{NO_INFO_FALLBACK} (LLM request timed out)"
        except httpx.HTTPStatusError as e:
            # Mask API keys and sensitive headers in exceptions
            status_code = e.response.status_code if e.response else 500
            return f"{NO_INFO_FALLBACK} (LLM service returned HTTP {status_code})"
        except Exception:
            return NO_INFO_FALLBACK
