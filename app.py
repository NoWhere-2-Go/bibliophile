"""CLI for Phase 1: Ingest text files into a Chroma vector store and query."""
import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from rag.ingest import ingest_directory
from rag.embeddings import EmbeddingModel, batch_embed
from rag.vectorstore import ChromaVectorStore
from rag.retriever import Retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_ingest(args):
    """Ingest books from a directory into the vector store."""
    logger.info("=" * 60)
    logger.info("Phase 1: OFFLINE INGESTION PIPELINE")
    logger.info("=" * 60)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    index_dir = Path(args.index_dir)
    chunk_size = args.chunk_size
    overlap = args.overlap
    
    # Step 1: Read and chunk documents
    logger.info(f"\n[1/5] Reading and chunking text files from {input_dir}")
    try:
        docs = ingest_directory(
            str(input_dir),
            chunk_tokens=chunk_size,
            overlap=overlap
        )
        logger.info(f"✓ Created {len(docs)} chunks")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)
    
    if not docs:
        logger.error("No documents found to ingest")
        sys.exit(1)
    
    texts = [d["text"] for d in docs]
    
    # Step 2: Verify embedding service
    logger.info(f"\n[2/5] Checking embedding service at {args.ollama_base_url}")
    model = EmbeddingModel(
        model_name=args.embedding_model,
        base_url=args.ollama_base_url
    )
    if not model.health_check():
        logger.error("Embedding service is not available. Start Ollama first:")
        logger.error(f"  ollama serve")
        logger.error(f"Then verify: curl -X POST '{args.ollama_base_url}/api/embed' \\")
        logger.error(f"  -H 'Content-Type: application/json' \\")
        logger.error(f"  -d '{{\"model\":\"{args.embedding_model}\",\"input\":[\"test\"]}}'")
        sys.exit(1)
    
    # Step 3: Compute embeddings
    logger.info(f"\n[3/5] Computing {len(texts)} embeddings (batch size: {args.batch_size})")
    try:
        vectors = batch_embed(model, texts, batch_size=args.batch_size)
        logger.info(f"✓ Embeddings shape: {vectors.shape}")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        sys.exit(1)
    
    # Step 4: Prepare for storage
    logger.info(f"\n[4/5] Preparing documents for storage")
    ids = [str(uuid.uuid4()) for _ in texts]
    metadatas = [dict(**d["meta"]) for d in docs]
    documents = texts
    logger.info(f"✓ Prepared {len(ids)} document records")
    
    # Step 5: Store in ChromaDB
    logger.info(f"\n[5/5] Storing in ChromaDB at {index_dir}")
    try:
        store = ChromaVectorStore(persist_directory=str(index_dir))
        added = store.add(
            ids=ids,
            vectors=vectors.tolist(),
            metadatas=metadatas,
            documents=documents
        )
        store.persist()
        logger.info(f"✓ Successfully stored {added} documents")
    except Exception as e:
        logger.error(f"Storage failed: {e}")
        sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nVector store saved to: {index_dir}")
    logger.info(f"Documents indexed: {added}")
    logger.info(f"Embedding dimension: {vectors.shape[1]}")


def cmd_query(args):
    """Query the vector store."""
    logger.info("=" * 60)
    logger.info("Phase 2: RUNTIME QUERY PIPELINE")
    logger.info("=" * 60)
    
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        logger.error(f"Run 'python app.py ingest <input_dir> <index_dir>' first")
        sys.exit(1)
    
    logger.info(f"\n[1/3] Loading vector store from {index_dir}")
    try:
        store = ChromaVectorStore.load(str(index_dir))
        stats = store.get_collection_stats()
        logger.info(f"✓ Loaded {stats['document_count']} indexed documents")
    except Exception as e:
        logger.error(f"Failed to load store: {e}")
        sys.exit(1)
    
    logger.info(f"\n[2/3] Initializing embedding model")
    model = EmbeddingModel(
        model_name=args.embedding_model,
        base_url=args.ollama_base_url
    )
    if not model.health_check():
        logger.error("Embedding service unavailable")
        sys.exit(1)
    
    logger.info(f"\n[3/3] Retrieving top-{args.k} results for query:")
    logger.info(f"  Query: {args.query}")
    
    try:
        retriever = Retriever(model, store)
        results = retriever.retrieve(args.query, top_k=args.k)
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS (Top {len(results)}):")
        logger.info(f"{'=' * 60}\n")
        
        for i, result in enumerate(results, 1):
            formatted = retriever.format_result(result, i)
            print(formatted)
            print()
        
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline: Phase 1 (Offline) Ingestion & Query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest books
  python app.py ingest ./data ./index_dir
  
  # Query the index
  python app.py query ./index_dir "books about space exploration" -k 10
        """
    )
    
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for Ollama service (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Name of the embedding model in Ollama (default: nomic-embed-text)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    
    # Ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest text files and build vector store"
    )
    ingest_parser.add_argument("input_dir", help="Directory containing text files to ingest")
    ingest_parser.add_argument("index_dir", help="Directory to store the vector index")
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Token-based chunk size (default: 512)"
    )
    ingest_parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Token overlap between chunks (default: 64)"
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding computation (default: 64)"
    )
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Query subcommand
    query_parser = subparsers.add_parser(
        "query",
        help="Query the vector store"
    )
    query_parser.add_argument("index_dir", help="Directory containing the vector index")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    query_parser.set_defaults(func=cmd_query)
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
