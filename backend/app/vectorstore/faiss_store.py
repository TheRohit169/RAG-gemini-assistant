"""
faiss_store.py
--------------
FAISS-based vector store for storing and searching document embeddings.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSStore:
    """
    Wraps a FAISS IndexFlatIP (inner product / cosine similarity after
    L2-normalisation) and keeps chunk metadata aligned with index positions.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding vector dimension (e.g. 384 for MiniLM)
        """
        self.dimension = dimension
        # IndexFlatIP computes dot product; we normalise vectors → cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []
        logger.info(f"FAISS index initialised (dim={dimension})")

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Adds embeddings and their associated chunk metadata to the index.

        Args:
            embeddings: 2-D float32 array of shape (n, dimension)
            chunks: Matching list of chunk metadata dicts
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Mismatch between embedding count and chunk count")

        # L2-normalise so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normalised = (embeddings / norms).astype(np.float32)

        self.index.add(normalised)
        self.metadata.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Returns the top-k most similar chunks with their cosine similarity scores.

        Args:
            query_embedding: 1-D float32 query vector
            top_k: Number of results to return

        Returns:
            List of (chunk_metadata, score) tuples, highest score first
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty – no results to return")
            return []

        # Normalise query vector
        vec = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))

        logger.debug(f"FAISS search returned {len(results)} result(s)")
        return results

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
