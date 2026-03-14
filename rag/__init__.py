"""RAG (Retrieval-Augmented Generation) module for book recommendations.

Phase 1 (Offline): Ingestion, chunking, embedding, and indexing
Phase 2 (Runtime): Query retrieval and context building
Phase 3 (Generation): LLM-based recommendation generation
"""

from .ingest import ingest_directory, chunk_text_by_tokens, extract_book_metadata
from .embeddings import EmbeddingModel, batch_embed
from .vectorstore import ChromaVectorStore
from .retriever import Retriever

__all__ = [
    "ingest_directory",
    "chunk_text_by_tokens",
    "extract_book_metadata",
    "EmbeddingModel",
    "batch_embed",
    "ChromaVectorStore",
    "Retriever",
]
