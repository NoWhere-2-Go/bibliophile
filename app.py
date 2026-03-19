"""CLI for Phase 1: Ingest text files into a Chroma vector store and query."""
import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from rag.embeddings import EmbeddingModel, batch_embed
from rag.vectorstore import ChromaVectorStore
from rag.retriever import Retriever
from rag.ingest import ingest_directory_streaming, ingest_metadata_stubs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_ingest(args):
    """Ingest books from a directory into the vector store."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    index_dir = Path(args.index_dir)

    logger.info(f"[1/4] Checking embedding service at {args.ollama_base_url}")
    model = EmbeddingModel(
        model_name=args.embedding_model,
        base_url=args.ollama_base_url
    )
    if not model.health_check():
        logger.error("Embedding service unavailable. Run 'ollama serve' and try again.")
        sys.exit(1)
    logger.info("✓ Embedding service ready")

    logger.info(f"[2/4] Initializing ChromaDB at {index_dir}")
    try:
        store = ChromaVectorStore(persist_directory=str(index_dir))
        logger.info("✓ Vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        sys.exit(1)

    logger.info(f"[3/4] Ingesting chunks from {input_dir} in batches of {args.batch_size}")

    EMBED_BATCH = args.batch_size
    batch_texts = []
    batch_docs = []
    total_added = 0

    def flush_batch():
        nonlocal total_added
        if not batch_texts:
            return
        try:
            vectors = batch_embed(model, batch_texts, batch_size=EMBED_BATCH)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

        ids = [str(uuid.uuid4()) for _ in batch_texts]
        metadatas = [dict(**d["meta"]) for d in batch_docs]

        try:
            added = store.add(
                ids=ids,
                vectors=vectors.tolist(),
                metadatas=metadatas,
                documents=batch_texts,
            )
            total_added += added
        except Exception as e:
            logger.error(f"Storage failed: {e}")
            raise

        batch_texts.clear()
        batch_docs.clear()

    try:
        if args.workers > 1:
            for chunk in ingest_directory_streaming(
                str(input_dir),
                chunk_tokens=args.chunk_size,
                overlap=args.overlap,
                num_workers=args.workers,
            ):
                batch_texts.append(chunk["text"])
                batch_docs.append(chunk)
                if len(batch_texts) >= EMBED_BATCH:
                    flush_batch()
        else:
            logger.info('Starting ingestion')
            for chunk in ingest_metadata_stubs(
                    str(input_dir),
                    limit=args.limit,
            ):
                batch_texts.append(chunk["text"])
                batch_docs.append(chunk)
                if len(batch_texts) >= EMBED_BATCH:
                    flush_batch()

        flush_batch()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=args.verbose)
        sys.exit(1)

    if total_added == 0:
        logger.error("No documents were ingested. Check your input directory.")
        sys.exit(1)

    logger.info(f"[4/4] Persisting vector store")
    try:
        store.persist()
        logger.info(f"✓ Done — {total_added} chunks stored to {index_dir}")
    except Exception as e:
        logger.error(f"Failed to persist store: {e}")
        sys.exit(1)


def cmd_query(args):
    """Query the vector store."""
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}. Run 'python app.py ingest' first.")
        sys.exit(1)

    logger.info(f"[1/3] Loading vector store from {index_dir}")
    try:
        store = ChromaVectorStore.load(str(index_dir))
        stats = store.get_collection_stats()
        logger.info(f"✓ Loaded {stats['document_count']} documents")
    except Exception as e:
        logger.error(f"Failed to load store: {e}")
        sys.exit(1)

    logger.info(f"[2/3] Initializing embedding model")
    model = EmbeddingModel(
        model_name=args.embedding_model,
        base_url=args.ollama_base_url
    )
    if not model.health_check():
        logger.error("Embedding service unavailable. Run 'ollama serve' and try again.")
        sys.exit(1)
    logger.info("✓ Embedding model ready")

    logger.info(f"[3/3] Querying: {args.query}")
    try:
        retriever = Retriever(model, store)
        results = retriever.retrieve(args.query, top_k=args.k)
        logger.info(f"✓ Retrieved {len(results)} results\n")

        for i, result in enumerate(results, 1):
            print(retriever.format_result(result, i))
            print()
    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline: Phase 1 (Offline) Ingestion & Query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py ingest ./data ./index_dir
  python app.py ingest ./data ./index_dir --chunk-size 256 --overlap 32 --workers 2
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

    ingest_parser = subparsers.add_parser("ingest", help="Ingest text files and build vector store")
    ingest_parser.add_argument("input_dir", help="Directory containing text files to ingest")
    ingest_parser.add_argument("index_dir", help="Directory to store the vector index")
    ingest_parser.add_argument("--chunk-size", type=int, default=512, help="Token-based chunk size (default: 512)")
    ingest_parser.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks (default: 64)")
    ingest_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding and storage (default: 64)")
    ingest_parser.add_argument("--workers", type=int, default=1, help="Parallel workers for file reading/chunking (default: 1)")
    ingest_parser.add_argument("--limit",type=int,default=None,help="Cap the number of files to ingest (default: all)")
    ingest_parser.set_defaults(func=cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Query the vector store")
    query_parser.add_argument("index_dir", help="Directory containing the vector index")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("-k", type=int, default=5, help="Number of results to return (default: 5)")
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