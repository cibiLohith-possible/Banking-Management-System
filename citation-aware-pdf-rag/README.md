# Citation-Aware PDF RAG

## Overview
**Citation-Aware PDF RAG** is a production-grade Retrieval-Augmented Generation (RAG) system built in Python 3.11+. It indexes PDF documents into a local vector database and provides natural-language question answering backed by **verifiable, ground-truth citations** (Document Name, Page Number, and Exact Text Snippet).

## Problem Statement
Standard LLM applications frequently suffer from two critical flaws:
1. **Hallucination**: Generating facts not supported by source documents.
2. **Unverifiable Citations**: Allowing LLMs to invent source metadata or page numbers.

This system guarantees zero citation fabrication by bypassing LLM metadata generation entirely—citations are extracted directly from Qdrant metadata and coupled with strict grounding prompts.

## Features
- **Page-Aware PDF Loading**: Extracts text page-by-page using PyMuPDF while recording document filenames and 1-based page numbers.
- **Deterministic Text Chunking**: Splits large pages into overlapping text chunks with unique, deterministic IDs (`{document_name}_p{page}_c{index}`).
- **Local Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally (384-dim vector space) without paid API calls.
- **Qdrant Vector Database**: Stores embeddings and metadata with Cosine similarity search and automatic point overwrite (idempotent ingestion).
- **Strict Grounding & Fallback**: If no relevant context meets the similarity threshold, the LLM is **not called**, and the system returns: `"The information is not available in the supplied documents."`
- **FastAPI Web Service**: High-performance RESTful API endpoints (`/`, `/health`, `/ask`) with OpenAPI documentation.

## Architecture

```
         +-------------------+
         |   PDF Documents   |
         +---------+---------+
                   |
                   v
         +-------------------+
         |     PyMuPDF       | (Page-aware extraction)
         +---------+---------+
                   |
                   v
         +-------------------+
         |  Text Chunking    | (1000 char size, 200 overlap)
         +---------+---------+
                   |
                   v
         +-------------------+
         | Local Embeddings  | (all-MiniLM-L6-v2)
         +---------+---------+
                   |
                   v
         +-------------------+
         | Qdrant Store      | (Vector DB + Payload metadata)
         +---------+---------+
                   |
                   v
         +-------------------+
         | Semantic Retriever| (Cosine similarity search)
         +---------+---------+
                   |
                   v
         +-------------------+
         | OpenRouter Free   | (Strictly grounded prompt)
         |     LLM           |
         +---------+---------+
                   |
                   v
         +-------------------+
         | Answer + Verified | (Citations directly from Qdrant)
         |    Citations      |
         +-------------------+
```

## Technology Stack
- **Python 3.11+**
- **PyMuPDF (`pymupdf`)** for PDF document processing
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for local embeddings
- **Qdrant (`qdrant-client`)** for vector storage and retrieval
- **OpenRouter API (`httpx`)** for free LLM inference (`meta-llama/llama-3.3-70b-instruct:free`)
- **FastAPI & Uvicorn** for REST API service
- **Pytest** for unit and integration testing

## Project Structure
```
citation-aware-pdf-rag/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration from env / defaults
│   ├── pdf_loader.py       # Page-level PDF text loader
│   ├── chunker.py          # Deterministic text chunker with overlap
│   ├── embeddings.py       # Local SentenceTransformer embeddings wrapper
│   ├── qdrant_store.py     # Qdrant client vector & metadata storage
│   ├── retriever.py        # Semantic similarity search & threshold filter
│   ├── llm.py              # OpenRouter free LLM client with grounding
│   ├── citation.py         # Ground-truth citation formatter & deduplicator
│   └── rag_pipeline.py     # Complete RAG pipeline orchestration
├── data/
│   └── pdfs/               # Target directory for PDF ingestion
│       └── .gitkeep
├── scripts/
│   └── ingest.py           # CLI script to batch process & index PDFs
├── tests/
│   ├── __init__.py
│   ├── test_pdf_loader.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_qdrant_store.py
│   ├── test_ingest.py
│   ├── test_retriever.py
│   ├── test_llm.py
│   ├── test_citations.py
│   ├── test_rag_pipeline.py
│   ├── test_api.py
│   └── test_integration.py
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Exact dependencies
├── .env.example            # Environment configuration template
├── .gitignore              # Git exclusion rules
├── README.md               # Project documentation
└── LICENSE                 # MIT License
```

## Requirements
- Python 3.11+
- Qdrant Vector Database (Local Docker container or Qdrant Cloud instance)
- OpenRouter API Key (Free tier key from [openrouter.ai](https://openrouter.ai))

## Installation

1. **Clone the repository and navigate to project root**:
   ```bash
   cd citation-aware-pdf-rag
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Configuration
Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
```

`.env` configuration example:
```dotenv
OPENROUTER_API_KEY=your_openrouter_free_api_key
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=pdf_documents
```

## Qdrant Setup

### Local Docker Setup (Recommended)
Run Qdrant locally in a Docker container:
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Qdrant Cloud Setup (Optional)
Set `QDRANT_URL` to your cloud endpoint and set `QDRANT_API_KEY` in `.env`.

## PDF Ingestion
Place your PDF files into `data/pdfs/` and run the ingestion pipeline:
```bash
python scripts/ingest.py
```

Console output preview:
```text
Found 2 PDF file(s) in 'data/pdfs/'

Loading embedding model...
Embedding model ready  (dimension=384)

Processing: employee_handbook.pdf
  Pages extracted : 15
  Chunks created  : 42
  Stored in Qdrant: 42

=============================================
Ingestion complete.
  Total pages   : 15
  Total chunks  : 42
  Total stored  : 42
=============================================
```

## Running the API
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
Interactive OpenAPI documentation will be available at `http://127.0.0.1:8000/docs`.

## API Usage

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Ask a Question (`POST /ask`)
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the annual leave policy?"}'
```

Response format:
```json
{
  "answer": "Full-time employees receive 25 days of paid annual leave per year.",
  "sources": [
    {
      "document_name": "employee_handbook.pdf",
      "page_number": 12,
      "retrieved_text": "All full-time employees are entitled to 25 days of paid annual leave per calendar year."
    }
  ]
}
```

## Citation Format
Citations are generated strictly from Qdrant metadata:
```text
Answer:
Full-time employees receive 25 days of paid annual leave per year.

Source 1:
Document: employee_handbook.pdf
Page: 12
Retrieved Text:
"All full-time employees are entitled to 25 days of paid annual leave per calendar year."
```

## Unknown Questions
If a question cannot be answered using the retrieved document context, the system bypasses the LLM call entirely and returns:
```json
{
  "answer": "The information is not available in the supplied documents.",
  "sources": []
}
```

## Testing
Run the comprehensive test suite with pytest:
```bash
pytest -v
```

## Security
- API keys are read from environment variables and never logged or committed to repository tracking.
- `.env` and virtual environment folders are excluded via `.gitignore`.
- Exception details and stack traces are suppressed in user-facing HTTP responses.

## Limitations
- **Scanned / Image PDFs**: Current version parses text-based PDFs via PyMuPDF; scanned documents without OCR text layers are not supported.
- **Complex Tables**: Complex multi-column table layouts may require advanced table parsing.

## Future Improvements
- Optical Character Recognition (OCR) support for scanned PDF pages.
- Hybrid Search (combining BM25 keyword search with Qdrant vector search).
- Cross-encoder Reranking for improved top-k retrieval precision.
- Streaming responses for real-time frontend rendering.
