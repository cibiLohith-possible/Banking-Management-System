"""Application configuration using environment variables with sensible local defaults."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenRouter LLM Configuration ---
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY", None)
OPENROUTER_MODEL: str = os.getenv(
    "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
)

# --- Qdrant Configuration ---
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "pdf_documents")
