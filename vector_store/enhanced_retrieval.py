"""
Advanced vector retrieval with hybrid search and reranking.
"""

import logging
import re
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """Class for enhanced retrieval with hybrid search and reranking."""

    def __init__(
            self,
            vector_db_manager,
            text_processor,
            embedder,
            rerank_ratio: float = 0.3,
    ):
        """
        Initialize the enhanced retriever.

        Args:
            vector_db_manager: Vector database manager.
            text_processor: Text processor.
            embedder: Text embedder.
            rerank_ratio: Weight of BM25 score in hybrid search (0-1).
        """
        self.vector_db = vector_db_manager
        self.text_processor = text_processor
        self.embedder = embedder
        self.rerank_ratio = rerank_ratio
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    def _preprocess_for_bm25(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for BM25 retrieval.

        Args:
            texts: List of text strings.

        Returns:
            List of tokenized documents.
        """
        tokenized_docs = []
        for text in texts:
            # Basic preprocessing
            text = text.lower()
            # Remove special characters
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            # Tokenize
            tokens = text.split()
            tokenized_docs.append(tokens)

        return tokenized_docs

    def hybrid_search(
            self,
            query: str,
            n_results: int = 20,
            filter_criteria: Optional[Dict] = None,
            final_results: int = 5,
    ) -> pd.DataFrame:
        """
        Perform hybrid search combining vector similarity and BM25.

        Args:
            query: Query text.
            n_results: Number of initial results to retrieve.
            filter_criteria: Filter criteria for vector search.
            final_results: Number of final results after reranking.

        Returns:
            DataFrame with reranked results.
        """
        # Step 1: Retrieve larger set of candidates from vector DB
        vector_results = self.vector_db.query_news(
            query_text=query,
            n_results=n_results,
            filter_criteria=filter_criteria,
        )

        if not vector_results["ids"][0]:
            return pd.DataFrame()

        # Create initial dataframe
        initial_df = pd.DataFrame({
            "id": vector_results["ids"][0],
            "document": vector_results["documents"][0],
            "metadata": vector_results["metadatas"][0],
            "distance": vector_results["distances"][0] if "distances" in vector_results else [0] * len(
                vector_results["ids"][0]),
        })

        # Step 2: Apply BM25 reranking
        documents = initial_df["document"].tolist()
        tokenized_docs = self._preprocess_for_bm25(documents)

        # Initialize BM25
        bm25 = BM25Okapi(tokenized_docs)

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize BM25 scores
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)

        # Add BM25 scores to dataframe
        initial_df["bm25_score"] = bm25_scores

        # Step 3: Combine scores
        # Convert vector distances to similarity scores (1 - distance)
        initial_df["vector_score"] = 1 - initial_df["distance"]

        # Hybrid score as weighted combination
        initial_df["hybrid_score"] = (
                (1 - self.rerank_ratio) * initial_df["vector_score"] +
                self.rerank_ratio * initial_df["bm25_score"]
        )

        # Step 4: Sort by hybrid score and return top results
        final_df = initial_df.sort_values("hybrid_score", ascending=False).head(final_results)

        return final_df

    def retrieve_with_semantic_expansion(
            self,
            query: str,
            n_results: int = 5,
            filter_criteria: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Retrieve documents with query expansion.

        Args:
            query: Original query text.
            n_results: Number of results to return.
            filter_criteria: Filter criteria.

        Returns:
            DataFrame with results.
        """
        # TODO: Implement this using an LLM to expand the query
        # For now, we'll use the hybrid search
        return self.hybrid_search(query, n_results=n_results, filter_criteria=filter_criteria)