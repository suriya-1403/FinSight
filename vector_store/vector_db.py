"""
Module for managing vector database operations.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import chromadb
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions

from finsight.vector_store.embeddings import TextEmbedder

# Setup logging
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Class for managing ChromaDB vector database operations."""

    def __init__(
            self,
            collection_name: str = "news_embeddings",
            persist_directory: Optional[str] = None,
            embedding_model: Optional[TextEmbedder] = None,
    ):
        """
        Initialize the ChromaDB manager.

        Args:
            collection_name: Name of the collection to use.
            persist_directory: Directory to persist the database.
            embedding_model: Text embedder to use.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize embedder
        self.embedder = embedding_model or TextEmbedder()

        # Initialize ChromaDB
        self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize the ChromaDB client and collection."""
        try:
            if self.persist_directory:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                logger.info(f"Initialized persistent ChromaDB client at {self.persist_directory}")
            else:
                self.client = chromadb.Client()
                logger.info("Initialized in-memory ChromaDB client")

            # Create or get collection
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedder.model_name
            )

            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=sentence_transformer_ef
                )
                logger.info(f"Retrieved existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=sentence_transformer_ef
                )
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def add_news_articles(self, news_df: pd.DataFrame) -> None:
        """
        Add news articles to the vector database.

        Args:
            news_df: DataFrame containing news articles.
        """
        try:
            # Extract data from DataFrame
            ids = []
            documents = []
            metadatas = []

            for idx, row in news_df.iterrows():
                # Generate a unique ID using row index if _id is not available
                doc_id = str(row.get("_id", f"article_{idx}"))

                # Skip empty IDs
                if not doc_id or doc_id.isspace():
                    doc_id = f"article_{idx}"

                # Combine title and content for embedding
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                doc = f"{title}. {content}"

                # Skip empty documents
                if not doc or doc.isspace():
                    continue

                # Create metadata
                try:
                    source_name = row.get("source", {})
                    if isinstance(source_name, dict):
                        source_name = source_name.get("name", "Unknown")
                    else:
                        source_name = "Unknown"

                    sentiment_compound = row.get("sentiment_compound", 0)
                    if not isinstance(sentiment_compound, (int, float)):
                        sentiment_compound = 0

                    metadata = {
                        "source": str(source_name),
                        "publishedAt": str(row.get("publishedAt", "")),
                        "url": str(row.get("url", "")),
                        "sentiment_compound": float(sentiment_compound),
                        "sentiment_label": str(row.get("sentiment_label", "neutral")),
                    }

                    ids.append(doc_id)
                    documents.append(doc)
                    metadatas.append(metadata)
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    continue

            # Add documents to collection
            if ids:
                # Check for duplicates in the IDs
                if len(set(ids)) != len(ids):
                    # If duplicates exist, make them unique
                    unique_ids = []
                    for i, id_val in enumerate(ids):
                        while id_val in unique_ids:
                            id_val = f"{id_val}_{i}"
                        unique_ids.append(id_val)

                    self.collection.add(
                        ids=unique_ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                else:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                logger.info(f"Added {len(ids)} news articles to vector database")
            else:
                logger.info("No news articles to add to vector database")

        except Exception as e:
            logger.error(f"Error adding news articles to vector database: {str(e)}")
            raise

    def query_news(
            self,
            query_text: str,
            n_results: int = 5,
            filter_criteria: Optional[Dict] = None,
    ) -> Dict:
        """
        Query news articles by semantic similarity.

        Args:
            query_text: Text to query.
            n_results: Number of results to return.
            filter_criteria: Optional criteria to filter results.

        Returns:
            Dictionary of query results.
        """
        try:
            # In ChromaDB, we can't use an empty filter
            where_filter = filter_criteria if filter_criteria else None

            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )

            logger.info(f"Query '{query_text}' returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Error querying vector database: {str(e)}")
            raise