"""
embedding_service.py
--------------------
Generates sentence embeddings using sentence-transformers (MiniLM-L3-v2).
Implements singleton lazy loading to minimize memory footprint.
"""

from typing import List, Union
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "paraphrase-MiniLM-L3-v2"

class EmbeddingService:
    _model = None
    _dimension = None

    def __init__(self):
        pass

    def load_model(self):
        """Internal helper to load the model."""
        from sentence_transformers import SentenceTransformer
        logger.info(f"Lazy loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        self._dimension = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        return model

    def get_model(self):
        """Returns the model instance, loading it only if not already loaded."""
        if EmbeddingService._model is None:
            EmbeddingService._model = self.load_model()
        return EmbeddingService._model

    @property
    def model(self):
        """Compatibility property for existing code."""
        return self.get_model()

    @property
    def dimension(self):
        """Returns the embedding dimension."""
        if self._dimension is None:
            self.get_model()  
        return self._dimension

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embeddings for one or more texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Trigger lazy loading via get_model()
        embeddings = self.get_model().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string. Returns 1-D float32 array."""
        return self.embed(query)[0]