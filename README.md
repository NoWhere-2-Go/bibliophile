z# Bibliophile: Book Recommendation RAG System

> A Retrieval-Augmented Generation (RAG) engine for intelligent book recommendations using local LLMs.

## Overview

Bibliophile implements a three-phase book recommendation system:

- **Phase 1 (Offline)**: Ingest books, chunk them intelligently, compute embeddings, and build a searchable vector index ✅ COMPLETE
- **Phase 2 (Runtime)**: Embed user queries and retrieve relevant book passages  
- **Phase 3 (Generation)**: Augment prompts with retrieved context and use local LLMs to generate personalized recommendations

This repository contains a complete **Phase 1** implementation with enhanced components and comprehensive tooling.

## Quick Start

### 1. Setup

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Start Ollama

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull embedding model
ollama pull nomic-embed-text
```

### 3. Ingest Books

```bash
source .venv/bin/activate
python app.py ingest ./data ./chroma_db
```

### 4. Query the Index

```bash
python app.py query ./chroma_db "books about space exploration" -k 5
```

## Prerequisites

- **Python 3.11+**
- **Ollama** running locally with `nomic-embed-text` model
- **macOS/Linux** with zsh or bash

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## CLI Usage

### Ingest Command

```bash
python app.py ingest INPUT_DIR INDEX_DIR [OPTIONS]

Options:
  --chunk-size INT        Token-based chunk size (default: 512)
  --overlap INT           Token overlap between chunks (default: 64)
  --batch-size INT        Batch size for embeddings (default: 64)
  --embedding-model STR   Model name in Ollama (default: nomic-embed-text)
  --ollama-base-url URL   Ollama service URL (default: http://localhost:11434)
  -v, --verbose           Enable debug logging
```

**Example:**
```bash
python app.py ingest ./data ./chroma_db --chunk-size 512 --overlap 64
```

### Query Command

```bash
python app.py query INDEX_DIR "QUERY_TEXT" [OPTIONS]

Options:
  -k INT                  Number of results (default: 5)
  --embedding-model STR   Model name in Ollama (default: nomic-embed-text)
  --ollama-base-url URL   Ollama service URL (default: http://localhost:11434)
  -v, --verbose           Enable debug logging
```

**Example:**
```bash
python app.py query ./chroma_db "books like Dune" -k 10
```

## Project Structure

```
bibliophile/
├── app.py                 # CLI interface
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── .env.example          # Configuration template
│
├── rag/                  # RAG module
│   ├── __init__.py       # Package exports
│   ├── ingest.py         # Phase 1: Text chunking & metadata extraction
│   ├── embeddings.py     # Phase 1: Embedding computation (Ollama)
│   ├── vectorstore.py    # Phase 1: ChromaDB persistence
│   └── retriever.py      # Phase 2: Query retrieval & prompt building
│
├── data/                 # Sample book data
│   └── sample_book.txt
│
└── chroma_db/            # Vector store (created during ingest)
```

## Key Components

### 1. **Ingestion** (`rag/ingest.py`)

- Reads text files (UTF-8 with latin-1 fallback)
- Extracts metadata from filenames and content
- Token-based chunking (512 tokens, 64 overlap)
- Preserves metadata with each chunk

```python
from rag.ingest import ingest_directory

docs = ingest_directory("./data", chunk_tokens=512, overlap=64)
```

### 2. **Embeddings** (`rag/embeddings.py`)

- Local Ollama-based embeddings (no external APIs)
- Model: `nomic-embed-text` (768-dimensional)
- Batch processing for efficiency
- Retry logic with exponential backoff
- Health checks

```python
from rag.embeddings import EmbeddingModel, batch_embed

model = EmbeddingModel(model_name="nomic-embed-text")
embeddings = batch_embed(model, texts, batch_size=64)
```

### 3. **Vector Store** (`rag/vectorstore.py`)

- ChromaDB with DuckDB persistence
- Automatic disk persistence
- Metadata filtering support
- Collection statistics

```python
from rag.vectorstore import ChromaVectorStore

store = ChromaVectorStore(persist_directory="./chroma_db")
results = store.search(query_vector, top_k=5)
```

### 4. **Retriever** (`rag/retriever.py`)

- Query embedding using same model as ingestion
- Top-k ANN search with similarity scoring
- Augmented prompt construction
- Human-readable result formatting

```python
from rag.retriever import Retriever

retriever = Retriever(embedder, store)
results = retriever.retrieve("query", top_k=5)
prompt = retriever.build_prompt("query", results)
```

## Features

✅ **Phase 1 Complete:**
- Multi-file ingestion from directories
- Intelligent token-based text chunking
- Metadata extraction from filenames & content
- Batch embedding computation via Ollama
- Persistent vector storage (ChromaDB)
- Comprehensive error handling & logging
- CLI with progress reporting
- Health checks and diagnostics

✅ **Optimizations:**
- Batch processing for embeddings
- Metadata preservation with chunks
- Graceful encoding fallbacks
- Exponential backoff retry logic
- Service health verification

## File Format

### Supported Input
- Text files (`.txt`) with UTF-8 or latin-1 encoding

### Optional Naming Convention

For automatic metadata extraction, name files as:

```
"Author Name - Book Title (year).txt"
```

Examples:
```
Frank Herbert - Dune (1965).txt
Isaac Asimov - Foundation (1951).txt
Arthur C Clarke - 2001 A Space Odyssey (1968).txt
```

### Metadata Preserved Per Chunk

```python
{
    "title": "Dune",
    "author": "Frank Herbert",
    "year": "1965",
    "genre": "Science Fiction",
    "source_name": "dune.txt",
    "source_path": "/path/to/dune.txt",
    "chunk_index": 0,
    "chunk_size_tokens": 512
}
```

## Troubleshooting

### Embedding Service Unreachable

```bash
# Verify Ollama is running
pgrep -i ollama

# Start Ollama if needed
ollama serve

# Test the endpoint
curl -X POST 'http://localhost:11434/api/embed' \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text","input":["test"]}'
```

### Model Not Found

```bash
# Pull the embedding model
ollama pull nomic-embed-text

# List available models
ollama list
```

### Connection Refused

```bash
# Verify port 11434 is open
nc -zv localhost 11434

# Try alternative base URL
export OLLAMA_BASE_URL=http://127.0.0.1:11434
python app.py ingest ./data ./chroma_db
```

## Configuration

### Environment Variables

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export EMBEDDING_MODEL=nomic-embed-text
```

Or create a `.env` file (see `.env.example` for template).

## Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Ingest 100KB text | ~2-3s | ~50MB |
| Embed 100 chunks | ~5-10s | ~100MB |
| Query & retrieve | ~0.5-1s | ~30MB |

*Measured on M1 Mac with Ollama local inference*

## Next Steps

- **Phase 2** (upcoming): Runtime query pipeline with filtering
- **Phase 3** (upcoming): LLM-based recommendation generation
- **Future**: PDF/EPUB support, hybrid search, reranking

## References

- [Ollama Embedding API](https://github.com/ollama/ollama/blob/main/docs/api.md#embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [tiktoken](https://github.com/openai/tiktoken)
- [RAG Architecture](./rag-architecture.html)

## License

MIT

