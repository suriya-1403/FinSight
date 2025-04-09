"""
Module for generating text embeddings.
"""

import logging
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

# Setup logging
logger = logging.getLogger(__name__)


class TextEmbedder:
    """Class for generating embeddings from text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the TextEmbedder.

        Args:
            model_name: Name of the embedding model to use.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded successfully")

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the provided texts.

        Args:
            texts: A single text string or a list of text strings.

        Returns:
            Array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise