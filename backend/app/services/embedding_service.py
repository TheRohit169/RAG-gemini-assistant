"""
embedding_service.py
--------------------
Generates sentence embeddings using sentence-transformers (MiniLM-L6-v2).
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Singleton-style service that loads the model once and reuses it."""

    def __init__(self):
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self._model = SentenceTransformer(MODEL_NAME)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embeddings for one or more texts.

        Args:
            texts: A single string or list of strings

        Returns:
            float32 numpy array of shape (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # normalisation done in FAISS store
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string. Returns 1-D float32 array."""
        return self.embed(query)[0]
