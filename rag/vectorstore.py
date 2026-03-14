from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import os
import logging
from chromadb.client import Client

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Wrapper for a ChromaDB collection. Uses an embedding function externally —
    here we accept vectors already computed and persist a collection to disk.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "books"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            self.client = Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))
            logger.info(f"Initialized ChromaDB at {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection '{collection_name}'")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection '{collection_name}'")

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict],
        documents: List[str]
    ) -> int:
        """Add documents to the vector store.
        
        Args:
            ids: List of unique string IDs.
            vectors: List of embedding vectors (each is a list of floats).
            metadatas: List of metadata dicts (title, author, etc.).
            documents: List of document texts (chunks).
        
        Returns:
            Number of documents added.
        """
        if not (len(ids) == len(vectors) == len(metadatas) == len(documents)):
            raise ValueError("All arguments must have the same length")
        
        try:
            # ChromaDB requires metadatas to have string values
            cleaned_metadatas = []
            for m in metadatas:
                cleaned = {k: str(v) for k, v in m.items()}
                cleaned_metadatas.append(cleaned)
            
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                metadatas=cleaned_metadatas,
                documents=documents
            )
            logger.info(f"Added {len(ids)} documents to collection")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents using a query vector.
        
        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            where: Optional ChromaDB where filter dict (e.g., {"author": {"$eq": "Frank Herbert"}}).
        
        Returns:
            List of result dicts with keys: score, metadata, document, id.
        """
        try:
            query_kwargs = {
                "query_embeddings": [query_vector],
                "n_results": top_k,
                "include": ["metadatas", "distances", "documents"]
            }
            
            if where:
                query_kwargs["where"] = where
            
            results = self.collection.query(**query_kwargs)
            
            out = []
            # results structure: dict with 'ids', 'distances', 'metadatas', 'documents' lists
            if results["ids"] and len(results["ids"]) > 0:
                for i, dist in enumerate(results["distances"][0]):
                    # ChromaDB returns distances (not similarities)
                    # For L2 distance, convert to similarity score (lower distance = higher similarity)
                    similarity = 1.0 / (1.0 + float(dist))
                    
                    out.append({
                        "id": results["ids"][0][i],
                        "score": similarity,
                        "distance": float(dist),
                        "metadata": results["metadatas"][0][i],
                        "document": results["documents"][0][i],
                    })
            
            logger.debug(f"Search returned {len(out)} results")
            return out
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def get_collection_stats(self) -> Dict:
        """Get basic statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    def persist(self) -> None:
        """Persist the collection to disk. ChromaDB usually does this automatically."""
        try:
            self.client.persist()
            logger.info("Persisted ChromaDB to disk")
        except Exception as e:
            logger.warning(f"Persistence failed (may be automatic): {e}")

    @classmethod
    def load(
        cls,
        persist_directory: str = "./chroma_db",
        collection_name: str = "books"
    ) -> "ChromaVectorStore":
        """Load an existing ChromaDB store from disk.
        
        Args:
            persist_directory: Path to persisted ChromaDB.
            collection_name: Name of the collection to load.
        
        Returns:
            ChromaVectorStore instance.
        """
        inst = cls(persist_directory, collection_name)
        stats = inst.get_collection_stats()
        logger.info(f"Loaded store: {stats}")
        return inst
