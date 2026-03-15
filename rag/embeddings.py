from typing import List, Optional
import numpy as np
import requests
import os
import time
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Embedding client that calls a local Ollama HTTP endpoint.

    Always uses the `nomic-embed-text` model by default. Ollama must be
    running locally (default http://localhost:11434) and host the model.
    
    Can also support other embedding providers if API-compatible.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0
    ):
        self.model = model_name
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.max_retries = max_retries
        self.timeout = timeout
        logger.info(f"Initialized EmbeddingModel: {model_name} at {self.base_url} (timeout={timeout}s)")

    def embed(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Embed a list of texts. Sends to Ollama in optimized batches.
        
        Args:
            texts: List of strings to embed.
            batch_size: Optional batch size for chunking requests to Ollama (default: 16).
        
        Returns:
            Numpy array of embeddings, shape (len(texts), embedding_dim).
        """
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        # Send to Ollama in larger batches to reduce HTTP overhead (default 16 texts per request)
        if batch_size is None:
            batch_size = 16
        
        all_embeddings = []
        endpoints = [
            f"{self.base_url}/api/embed",
            f"{self.base_url}/embed",
        ]
        
        logger.debug(f"embed() called with {len(texts)} texts, chunking into batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i : i + batch_size]
            payload = {"model": self.model, "input": sub_batch}
            
            logger.debug(f"Sending batch {i // batch_size + 1} with {len(sub_batch)} texts to Ollama")
            
            batch_succeeded = False
            for attempt in range(self.max_retries):
                for endpoint in endpoints:
                    try:
                        logger.debug(f"  Attempt {attempt+1}/{self.max_retries} at {endpoint}")
                        resp = requests.post(
                            endpoint,
                            json=payload,
                            timeout=self.timeout
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        
                        # Handle various response formats
                        embeddings = self._extract_embeddings(data)
                        if embeddings is not None:
                            logger.debug(f"✓ Batch {i // batch_size + 1} complete: shape {embeddings.shape}")
                            all_embeddings.append(embeddings)
                            batch_succeeded = True
                            break
                        
                    except requests.exceptions.Timeout:
                        logger.warning(f"Timeout at {endpoint} (attempt {attempt+1}/{self.max_retries})")
                        time.sleep(1)
                        continue
                    except requests.exceptions.ConnectionError as e:
                        logger.warning(f"Connection error at {endpoint}: {e}")
                        time.sleep(1)
                        continue
                    except Exception as e:
                        logger.debug(f"Error at {endpoint}: {type(e).__name__}: {e}")
                        continue
                
                if batch_succeeded:
                    break
            
            if not batch_succeeded:
                raise RuntimeError(
                    f"Failed to embed batch {i // batch_size + 1} after {self.max_retries} attempts. "
                    f"Ensure Ollama is running with model '{self.model}' and responding at {self.base_url}"
                )
        
        if not all_embeddings:
            raise RuntimeError(f"Failed to get any embeddings")
        
        logger.debug(f"Combining {len(all_embeddings)} batches...")
        return np.vstack(all_embeddings)

    def _extract_embeddings(self, data: dict) -> Optional[np.ndarray]:
        """Extract embeddings from various API response formats."""
        try:
            if isinstance(data, dict):
                # Ollama format: {"embeddings": [[...], ...]}
                if "embeddings" in data:
                    embs = data["embeddings"]
                    if embs and isinstance(embs[0], list):
                        return np.array(embs, dtype="float32")
                
                # OpenAI format: {"data": [{"embedding": [...]}, ...]}
                if "data" in data and isinstance(data["data"], list):
                    first = data["data"][0]
                    if isinstance(first, dict) and "embedding" in first:
                        embs = [d["embedding"] for d in data["data"]]
                        return np.array(embs, dtype="float32")
                    if isinstance(first, list):
                        return np.array(data["data"], dtype="float32")
            
            # Raw list format
            if isinstance(data, list) and data and isinstance(data[0], list):
                return np.array(data, dtype="float32")
        except Exception as e:
            logger.debug(f"Error extracting embeddings: {e}")
        
        return None

    def health_check(self) -> bool:
        """Check if the embedding service is available."""
        try:
            logger.info(f"Performing health check on {self.base_url}...")
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": ["test"]},
                timeout=5.0
            )
            resp.raise_for_status()
            logger.info(f"✓ Embedding service is healthy at {self.base_url}")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"✗ Health check timed out after 5 seconds at {self.base_url}")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"✗ Cannot connect to {self.base_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Health check failed: {type(e).__name__}: {e}")
            return False


def batch_embed(
    model: EmbeddingModel,
    texts: List[str],
    batch_size: int = 128
) -> np.ndarray:
    """Embed texts in batches to manage memory and reduce HTTP overhead.
    
    Args:
        model: EmbeddingModel instance.
        texts: List of texts to embed.
        batch_size: Number of texts per batch (default: 128). Increase for faster processing
                   if you have sufficient memory and Ollama can handle it, decrease if
                   seeing timeouts or memory issues.
    
    Returns:
        Numpy array of all embeddings.
    """
    if not texts:
        raise ValueError("No texts to embed")

    out = []
    total = len(texts)
    logger.info(f"Starting batch embedding of {total} texts with batch_size={batch_size}")
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        start_idx = i
        end_idx = min(i + batch_size, total)
        
        logger.info(f"[{start_idx}/{total}] Embedding batch {start_idx}-{end_idx}/{total} ({len(batch)} texts)")
        logger.debug(f"  Batch text lengths: {[len(t) for t in batch]}")
        
        try:
            batch_embeddings = model.embed(batch)
            logger.debug(f"  ✓ Batch complete: shape {batch_embeddings.shape}")
            out.append(batch_embeddings)
        except Exception as e:
            logger.error(f"  ✗ Batch failed: {e}")
            raise
    
    logger.info(f"Combining {len(out)} batches into single array...")
    result = np.vstack(out)
    logger.info(f"✓ Successfully embedded {total} texts: shape {result.shape}")
    return result

