#!/usr/bin/env python3
"""
Example script demonstrating Phase 1 workflow programmatically.

This shows how to use the RAG modules directly in Python code.
"""
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from rag.ingest import ingest_directory, extract_book_metadata
from rag.embeddings import EmbeddingModel, batch_embed
from rag.vectorstore import ChromaVectorStore
from rag.retriever import Retriever


def example_complete_workflow():
    """Run a complete Phase 1 workflow: ingest -> embed -> store -> query."""
    
    logger.info("=" * 70)
    logger.info("BIBLIOPHILE RAG - PHASE 1 COMPLETE WORKFLOW EXAMPLE")
    logger.info("=" * 70)
    
    # Configuration
    input_dir = "./data"
    index_dir = "./chroma_db_example"
    ollama_base_url = "http://localhost:11434"
    embedding_model = "nomic-embed-text"
    
    # Verify input directory exists
    if not Path(input_dir).exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    # Step 1: Ingest and chunk documents
    logger.info("\n[1/6] INGESTION: Reading and chunking text files")
    logger.info(f"      Input directory: {input_dir}")
    try:
        docs = ingest_directory(input_dir, chunk_tokens=512, overlap=64)
        logger.info(f"      ✓ Created {len(docs)} chunks from {len(set(d['meta'].get('source_name') for d in docs))} files")
        
        # Show sample metadata
        if docs:
            sample_meta = docs[0]["meta"]
            logger.info(f"      Sample chunk metadata: title={sample_meta.get('title')}, author={sample_meta.get('author')}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return False
    
    if not docs:
        logger.error("No documents ingested")
        return False
    
    # Step 2: Verify embedding service
    logger.info("\n[2/6] EMBEDDING: Verifying Ollama service")
    logger.info(f"      Service URL: {ollama_base_url}")
    logger.info(f"      Model: {embedding_model}")
    
    model = EmbeddingModel(
        model_name=embedding_model,
        base_url=ollama_base_url
    )
    
    if not model.health_check():
        logger.error("Embedding service not available")
        logger.error("Start Ollama: ollama serve")
        logger.error("Pull model: ollama pull nomic-embed-text")
        return False
    
    # Step 3: Compute embeddings
    logger.info("\n[3/6] EMBEDDING: Computing embeddings for all chunks")
    texts = [d["text"] for d in docs]
    try:
        vectors = batch_embed(model, texts, batch_size=64)
        logger.info(f"      ✓ Embeddings shape: {vectors.shape}")
        logger.info(f"      ✓ Embedding dimension: {vectors.shape[1]} (768 for nomic-embed-text)")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return False
    
    # Step 4: Create vector store
    logger.info("\n[4/6] STORAGE: Creating ChromaDB vector store")
    logger.info(f"      Persist directory: {index_dir}")
    try:
        store = ChromaVectorStore(persist_directory=index_dir)
        logger.info(f"      ✓ Vector store initialized")
    except Exception as e:
        logger.error(f"Store initialization failed: {e}")
        return False
    
    # Step 5: Add documents to store
    logger.info("\n[5/6] STORAGE: Adding documents to vector store")
    import uuid
    try:
        ids = [str(uuid.uuid4()) for _ in texts]
        metadatas = [dict(**d["meta"]) for d in docs]
        documents = texts
        
        added = store.add(
            ids=ids,
            vectors=vectors.tolist(),
            metadatas=metadatas,
            documents=documents
        )
        store.persist()
        logger.info(f"      ✓ Added {added} documents to store")
        
        stats = store.get_collection_stats()
        logger.info(f"      Store stats: {stats}")
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        return False
    
    # Step 6: Query the store
    logger.info("\n[6/6] RETRIEVAL: Querying the vector store")
    
    # Test queries
    test_queries = [
        "books with themes of power and politics",
        "science fiction adventures",
        "desert planets and strange ecology"
    ]
    
    try:
        retriever = Retriever(model, store)
        
        for query in test_queries:
            logger.info(f"\n      Query: '{query}'")
            results = retriever.retrieve(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                score = result.get("score", 0.0)
                title = result.get("metadata", {}).get("title", "Unknown")
                author = result.get("metadata", {}).get("author", "")
                logger.info(f"        [{i}] {title} (relevance: {score:.1%})")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return False
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ PHASE 1 WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    
    return True


def example_metadata_extraction():
    """Demonstrate metadata extraction from filenames."""
    
    logger.info("\n" + "=" * 70)
    logger.info("METADATA EXTRACTION EXAMPLES")
    logger.info("=" * 70)
    
    test_cases = [
        ("Frank Herbert - Dune (1965).txt", "Frank Herbert", "Dune"),
        ("Isaac Asimov - Foundation (1951).txt", "Isaac Asimov", "Foundation"),
        ("dune.txt", "", "dune"),
    ]
    
    for filename, expected_author, expected_title in test_cases:
        meta = extract_book_metadata(filename, "sample content")
        logger.info(f"\nFilename: {filename}")
        logger.info(f"  Title: {meta.get('title')}")
        logger.info(f"  Author: {meta.get('author')}")
        logger.info(f"  Year: {meta.get('year')}")


if __name__ == "__main__":
    try:
        # Run example metadata extraction
        example_metadata_extraction()
        
        # Run complete workflow
        success = example_complete_workflow()
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

