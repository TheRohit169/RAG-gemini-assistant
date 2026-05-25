"""
retrieval_service.py
--------------------
Retrieves relevant document chunks from FAISS using cosine similarity.
"""

from typing import List, Dict, Any, Tuple
import os

from app.vectorstore.faiss_store import FAISSStore
from app.services.embedding_service import EmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
TOP_K = int(os.getenv("TOP_K", "3"))


class RetrievalService:
    """Handles query embedding → FAISS search → threshold filtering."""

    def __init__(self, faiss_store: FAISSStore, embedding_service: EmbeddingService):
        self.store = faiss_store
        self.embedder = embedding_service

    def retrieve(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Embeds the query and retrieves top-k chunks above the similarity threshold.

        Args:
            query: User's natural-language question

        Returns:
            Tuple of (list of chunk dicts, list of similarity scores)
        """
        query_vec = self.embedder.embed_query(query)
        raw_results: List[Tuple[Dict[str, Any], float]] = self.store.search(
            query_vec, top_k=TOP_K
        )

        # Log all scores for transparency
        for chunk, score in raw_results:
            logger.info(
                f"Similarity {score:.4f} | source='{chunk['source_title']}' "
                f"chunk_id={chunk['chunk_id']}"
            )

        # Filter by threshold
        filtered = [
            (chunk, score)
            for chunk, score in raw_results
            if score >= SIMILARITY_THRESHOLD
        ]

        chunks = [c for c, _ in filtered]
        scores = [s for _, s in filtered]

        logger.info(
            f"Retrieved {len(chunks)}/{TOP_K} chunks above threshold {SIMILARITY_THRESHOLD}"
        )
        return chunks, scores
