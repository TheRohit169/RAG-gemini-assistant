"""
embedding_service.py
--------------------
Generates sentence embeddings using sentence-transformers (MiniLM-L6-v2).
Implements lazy loading to minimize memory footprint during startup.
"""

from typing import List, Union
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "paraphrase-MiniLM-L3-v2"

class EmbeddingService:
    """Singleton-style service that loads the model only when first needed."""

    def __init__(self):
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Lazy loading embedding model: {MODEL_NAME}")
            self._model = SentenceTransformer(MODEL_NAME)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self):
        """Returns the embedding dimension, loading the model if necessary."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embeddings for one or more texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Access self.model to trigger lazy loading if not already loaded
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string. Returns 1-D float32 array."""
        return self.embed(query)[0]