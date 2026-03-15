from typing import List, Dict, Optional, Tuple
import os
import logging
import re
from multiprocessing import Pool
from functools import partial

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
    Also scans first 10 lines for title/author markers (optimized).
    """
    metadata = {
        "source_name": filename,
        "title": "",
        "author": "",
        "year": "",
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
    
    # Scan first 10 lines only for markers (reduced from 50)
    lines = text.split("\n")[:10]
    text_preview = "\n".join(lines)
    
    # Combined regex pattern to reduce multiple searches
    title_match = re.search(r"[Tt]itle\s*[:=]\s*(.+?)(?:\n|$)", text_preview)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    author_match = re.search(r"[Aa]uthor\s*[:=]\s*(.+?)(?:\n|$)", text_preview)
    if author_match:
        metadata["author"] = author_match.group(1).strip()
    
    year_match = re.search(r"[Yy]ear\s*[:=]\s*(\d{4})", text_preview)
    if year_match:
        metadata["year"] = year_match.group(1)
    
    return metadata


# Global encoding cache to avoid reloading
_ENCODING_CACHE = {}

def _get_encoding(encoding_name: str = "cl100k_base"):
    """Get tiktoken encoding with caching to avoid hangs."""
    if encoding_name not in _ENCODING_CACHE:
        logger.debug(f"Loading encoding: {encoding_name}")
        _ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODING_CACHE[encoding_name]


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
        enc = _get_encoding(encoding_name)
    except Exception as e:
        logger.error(f"Failed to load encoding {encoding_name}: {e}")
        raise

    try:
        tokens = enc.encode(text)
    except Exception as e:
        logger.error(f"Failed to encode text for {metadata.get('source_name', 'unknown')}: {e}")
        raise
    
    chunks = []
    start = 0
    idx = 0
    L = len(tokens)
    
    if L == 0:
        logger.warning(f"No tokens for {metadata.get('source_name', 'unknown')}")
        return []

    # Decode all chunks at once in batches to reduce encoding/decoding overhead
    chunk_indices = []
    temp_start = 0
    while temp_start < L:
        temp_end = min(temp_start + chunk_tokens, L)
        if temp_end <= temp_start:
            break
        chunk_indices.append((temp_start, temp_end))
        temp_start = temp_end - overlap
        if temp_start <= 0 or temp_start >= L:
            if temp_end < L:
                temp_start = temp_end
            else:
                break
    
    # Batch decode all chunks together for efficiency
    for chunk_idx, (start, end) in enumerate(chunk_indices):
        chunk_tokens_slice = tokens[start:end]
        try:
            chunk_text = enc.decode(chunk_tokens_slice).strip()
        except Exception as e:
            logger.warning(f"Failed to decode chunk {chunk_idx}: {e}")
            chunk_text = ""
        
        if chunk_text:
            chunk_meta = {
                **metadata,
                "chunk_index": chunk_idx,
                "chunk_size_tokens": len(chunk_tokens_slice),
            }
            chunks.append({
                "id": f"{metadata.get('source_name', 'unknown')}-chunk-{chunk_idx}",
                "text": chunk_text,
                "meta": chunk_meta
            })

    logger.info(f"Created {len(chunks)} chunks from {metadata.get('source_name', 'unknown')} ({L} tokens)")
    return chunks


def _process_single_file(
    args: Tuple[str, str, int, int]
) -> Tuple[str, List[Dict]]:
    """Worker function for parallel file processing.
    
    Args:
        args: (path, filename, chunk_tokens, overlap)
    
    Returns:
        (filename, chunks_list)
    """
    path, filename, chunk_tokens, overlap = args
    try:
        text = read_text_file(path)
        metadata = extract_book_metadata(filename, text)
        metadata["source_path"] = path
        
        chunks = chunk_text_by_tokens(
            text,
            metadata=metadata,
            chunk_tokens=chunk_tokens,
            overlap=overlap
        )
        return filename, chunks
    except Exception as e:
        logger.error(f"Failed to ingest {path}: {e}")
        return filename, []


def ingest_directory_streaming(
    directory: str,
    ext: str = ".txt",
    chunk_tokens: int = 512,
    overlap: int = 64,
    num_workers: int = 4
):
    """Generator version of ingest_directory. Yields chunks one at a time."""
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")

    files_to_process = []
    for root, _, files in os.walk(directory):
        for fn in sorted(files):
            if fn.lower().endswith(ext):
                path = os.path.join(root, fn)
                files_to_process.append((path, fn, chunk_tokens, overlap))

    logger.info(f"Found {len(files_to_process)} files to process")

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.imap(_process_single_file, files_to_process, chunksize=1)
            for filename, chunks in results:
                yield from chunks
    else:
        for args in files_to_process:
            _, chunks = _process_single_file(args)
            yield from chunks


def ingest_directory(
    directory: str,
    ext: str = ".txt",
    chunk_tokens: int = 512,
    overlap: int = 64,
    num_workers: int = 4
) -> List[Dict]:
    """Eagerly collect all chunks into a list. Use ingest_directory_streaming
    if memory is a concern."""
    return list(ingest_directory_streaming(
        directory, ext, chunk_tokens, overlap, num_workers
    ))