from typing import List, Dict, Optional
import os
import logging
import re

try:
    import tiktoken
except Exception:
    tiktoken = None

logger = logging.getLogger(__name__)


def read_text_file(path: str) -> str:
    """Read a text file with UTF-8 encoding."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fall back to latin-1 for files with encoding issues
        logger.warning(f"UTF-8 decode failed for {path}, trying latin-1")
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def extract_book_metadata(filename: str, text: str) -> Dict[str, str]:
    """Extract book metadata from filename and text content.
    
    Looks for patterns in filename like: "Author - Title (year).txt"
    Also scans first 200 tokens for title/author markers.
    """
    metadata = {
        "source_name": filename,
        "title": "",
        "author": "",
        "year": "",
        "genre": "",
    }
    
    # Try to parse filename: "Author - Title (year).txt" or "Title - Author.txt"
    base = os.path.splitext(filename)[0]
    
    # Pattern: "Author - Title (year)"
    match = re.match(r"(.+?)\s*-\s*(.+?)\s*(?:\((\d{4})\))?$", base)
    if match:
        potential_author = match.group(1).strip()
        potential_title = match.group(2).strip()
        potential_year = match.group(3) or ""
        # Heuristic: if part 1 is likely a name (has capitals) it's author, else title
        if re.search(r"^[A-Z][a-z]+\s+[A-Z]", potential_author):
            metadata["author"] = potential_author
            metadata["title"] = potential_title
            metadata["year"] = potential_year
        else:
            metadata["title"] = potential_author
            metadata["author"] = potential_title
    else:
        # Just use filename as title
        metadata["title"] = base
    
    # Scan first few lines for markers like "Title:", "Author:", etc.
    lines = text.split("\n")[:10]
    text_preview = "\n".join(lines)
    
    title_match = re.search(r"[Tt]itle\s*[:=]\s*(.+?)(?:\n|$)", text_preview)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    author_match = re.search(r"[Aa]uthor\s*[:=]\s*(.+?)(?:\n|$)", text_preview)
    if author_match:
        metadata["author"] = author_match.group(1).strip()
    
    year_match = re.search(r"[Yy]ear\s*[:=]\s*(\d{4})", text_preview)
    if year_match:
        metadata["year"] = year_match.group(1)
    
    genre_match = re.search(r"[Gg]enre\s*[:=]\s*(.+?)(?:\n|$)", text_preview)
    if genre_match:
        metadata["genre"] = genre_match.group(1).strip()
    
    return metadata


def chunk_text_by_tokens(
    text: str,
    metadata: Dict[str, str],
    chunk_tokens: int = 512,
    overlap: int = 64,
    encoding_name: str = "cl100k_base"
) -> List[Dict]:
    """Chunk text by token counts using tiktoken. Returns list of dicts with 'text' and 'meta'.

    Requires `tiktoken`. Raises RuntimeError if tiktoken is not installed.
    
    Args:
        text: The document text to chunk.
        metadata: Book metadata dict (title, author, etc.).
        chunk_tokens: Target chunk size in tokens (~512 = ~2000 chars).
        overlap: Token overlap between chunks (~64 = ~256 chars).
        encoding_name: Tiktoken encoding (usually 'cl100k_base' for GPT models).
    
    Returns:
        List of dicts with keys: 'id', 'text', 'meta'.
    """
    if tiktoken is None:
        raise RuntimeError("tiktoken is required for token-based chunking. Install it in your venv.")

    if not text or not text.strip():
        logger.warning(f"Empty text for {metadata.get('source_name', 'unknown')}")
        return []

    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.error(f"Failed to load encoding {encoding_name}: {e}")
        raise

    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    L = len(tokens)
    
    if L == 0:
        logger.warning(f"No tokens for {metadata.get('source_name', 'unknown')}")
        return []

    while start < L:
        end = min(start + chunk_tokens, L)
        chunk_tokens_slice = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens_slice).strip()
        if chunk_text:
            # Include book metadata with each chunk
            chunk_meta = {
                **metadata,
                "chunk_index": idx,
                "chunk_size_tokens": len(chunk_tokens_slice),
            }
            chunks.append({
                "id": f"{metadata.get('source_name', 'unknown')}-chunk-{idx}",
                "text": chunk_text,
                "meta": chunk_meta
            })
            idx += 1
        
        # Move by (chunk_tokens - overlap) to next position
        start = end - overlap
        if start < 0:
            start = 0

    logger.info(f"Created {len(chunks)} chunks from {metadata.get('source_name', 'unknown')} ({L} tokens)")
    return chunks


def ingest_directory(
    directory: str,
    ext: str = ".txt",
    chunk_tokens: int = 512,
    overlap: int = 64
) -> List[Dict]:
    """Ingest all text files from a directory recursively.
    
    Args:
        directory: Root directory to scan for files.
        ext: File extension to process (default: '.txt').
        chunk_tokens: Token-based chunk size.
        overlap: Token overlap between chunks.
    
    Returns:
        List of chunk dicts ready for embedding and ingestion.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")

    docs = []
    file_count = 0
    
    for root, _, files in os.walk(directory):
        for fn in sorted(files):
            if fn.lower().endswith(ext):
                path = os.path.join(root, fn)
                try:
                    text = read_text_file(path)
                    metadata = extract_book_metadata(fn, text)
                    metadata["source_path"] = path
                    
                    chunks = chunk_text_by_tokens(
                        text,
                        metadata=metadata,
                        chunk_tokens=chunk_tokens,
                        overlap=overlap
                    )
                    docs.extend(chunks)
                    file_count += 1
                except Exception as e:
                    logger.error(f"Failed to ingest {path}: {e}")
                    continue
    
    logger.info(f"Ingested {file_count} files into {len(docs)} chunks")
    return docs
