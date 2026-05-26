"""
chunking.py
-----------
Splits document content into overlapping chunks for embedding.
"""

from typing import List, Dict, Any
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[str]:
    """
    Splits text into overlapping word-based chunks.

    Args:
        text: Input text to chunk
        chunk_size: Approximate number of words per chunk
        overlap: Number of words to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunks all documents and attaches metadata to each chunk.

    Args:
        documents: List of dicts with 'title' and 'content' keys
        chunk_size: Words per chunk
        overlap: Overlapping words between chunks

    Returns:
        List of chunk dicts containing text and metadata
    """
    all_chunks = []
    chunk_id = 0

    for doc in documents:
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")

        if not content.strip():
            logger.warning(f"Skipping empty document: {title}")
            continue

        text_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        logger.debug(f"Document '{title}' → {len(text_chunks)} chunk(s)")

        for i, chunk_text_content in enumerate(text_chunks):
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_title": title,
                    "chunk_index": i,
                    "text": chunk_text_content,
                }
            )
            chunk_id += 1

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
