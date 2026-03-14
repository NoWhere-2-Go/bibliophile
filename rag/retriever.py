from typing import List, Dict, Optional
import logging
from .embeddings import EmbeddingModel
from .vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retriever that embeds queries and fetches similar documents from the vector store."""
    
    def __init__(self, embedder: EmbeddingModel, store: ChromaVectorStore):
        self.embedder = embedder
        self.store = store
        logger.info("Initialized Retriever")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve top-k documents similar to the query.
        
        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            where: Optional ChromaDB where filter (e.g., filter by author or genre).
        
        Returns:
            List of result dicts with fields: id, score, distance, metadata, document.
        """
        try:
            logger.debug(f"Retrieving top-{top_k} for query: {query[:50]}...")
            qv = self.embedder.embed([query])[0]
            results = self.store.search(qv.tolist(), top_k=top_k, where=where)
            logger.info(f"Retrieved {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    def build_prompt(
        self,
        query: str,
        retrieved: List[Dict],
        system_role: str = "You are a knowledgeable book recommendation assistant."
    ) -> str:
        """Build an augmented prompt for the LLM.
        
        Args:
            query: The user's query.
            retrieved: List of retrieved document results.
            system_role: System prompt / role description.
        
        Returns:
            Formatted prompt string ready for the LLM.
        """
        parts = [system_role, "", "QUERY:", query, "", "RETRIEVED PASSAGES:"]
        
        for i, r in enumerate(retrieved, 1):
            doc = r.get("document", "")
            meta = r.get("metadata", {})
            title = meta.get("title", "Unknown")
            author = meta.get("author", "")
            score = r.get("score", 0.0)
            
            author_str = f" by {author}" if author else ""
            parts.append(f"\n[{i}] {title}{author_str} (relevance: {score:.3f})")
            parts.append(f"    {doc[:200]}...")  # Show preview
        
        parts.append("\n\nTASK:")
        parts.append("Based on the retrieved passages and user query, provide 5 ranked book recommendations with brief rationale.")
        
        return "\n".join(parts)

    def format_result(self, result: Dict, index: int = 1) -> str:
        """Format a single retrieval result for display.
        
        Args:
            result: Result dict from retrieve().
            index: Display index (1-based).
        
        Returns:
            Formatted string for console or UI display.
        """
        meta = result.get("metadata", {})
        doc = result.get("document", "")
        score = result.get("score", 0.0)
        
        title = meta.get("title", "Unknown")
        author = meta.get("author", "")
        source = meta.get("source_name", "")
        
        lines = [
            f"[{index}] {title}",
            f"    Author: {author or '(unknown)'}",
            f"    Relevance: {score:.2%}",
            f"    Source: {source}",
            f"    Excerpt: {doc[:150]}...",
        ]
        return "\n".join(lines)

