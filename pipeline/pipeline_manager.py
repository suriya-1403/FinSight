# finsight/pipeline/pipeline_manager.py
"""
Module for managing data pipelines.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from finsight.data_collection import NewsCollector, MarketDataCollector
from finsight.data_storage import MongoDBConnector
from finsight.preprocessing import TextProcessor
from finsight.analysis import SentimentAnalyzer
from finsight.config import (
    DEFAULT_NEWS_QUERIES,
    DEFAULT_TICKERS,
    DEFAULT_DAYS_BACK
)
from finsight.vector_store import TextEmbedder, ChromaDBManager
from finsight.llm_integration import InsightGenerator
from datetime import timedelta
import pymongo
from finsight.vector_store.enhanced_retrieval import EnhancedRetriever
from finsight.config import NEWS_COLLECTION
from finsight.agents.agent_controller import AgentSystem
from finsight.forecasting.price_forecasting import PriceForecastingEngine
from finsight.visualization.financial_visualizer import FinancialVisualizer
from finsight.agents.base_agents import TechnicalAnalysisAgent

# Setup logging
logger = logging.getLogger(__name__)


class PipelineManager:
    """Class for managing data pipelines."""

    def __init__(self):
        """Initialize the pipeline manager."""
        self.news_collector = NewsCollector()
        self.market_collector = MarketDataCollector()
        self.db_connector = MongoDBConnector()
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.vector_db_manager = ChromaDBManager(
            collection_name="news_embeddings",
            persist_directory="./vector_db"
        )
        self.insight_generator = InsightGenerator()
        self.enhanced_retriever = EnhancedRetriever(
            vector_db_manager=self.vector_db_manager,
            text_processor=self.text_processor,
            embedder=self.vector_db_manager.embedder,
            rerank_ratio=0.3,
        )
        self.agent_system = AgentSystem(model="mistral")
        self.technical_agent = TechnicalAnalysisAgent(
            name="TechnicalAgent",
            model="mistral"
        )
        self.forecasting_engine = PriceForecastingEngine()
        self.visualizer = FinancialVisualizer()

    def store_analysis_in_vector_db(self, analyzed_news: pd.DataFrame) -> None:
        """
        Store analyzed news in the vector database.

        Args:
            analyzed_news: DataFrame of analyzed news articles.
        """
        logger.info("Storing analysis results in vector database...")

        try:
            self.vector_db_manager.add_news_articles(analyzed_news)
            logger.info(f"Added {len(analyzed_news)} articles to vector database")
        except Exception as e:
            logger.error(f"Error storing analysis in vector database: {str(e)}")

    def run_data_collection_pipeline(
            self,
            news_queries: Optional[List[str]] = None,
            tickers: Optional[List[str]] = None,
            days_back: int = None,
            store_data: bool = True
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Run the data collection pipeline.
        """
        news_queries = news_queries or DEFAULT_NEWS_QUERIES
        tickers = tickers or DEFAULT_TICKERS
        days_back = days_back or DEFAULT_DAYS_BACK

        start_time = datetime.now()
        logger.info(f"Starting data collection pipeline at {start_time}")

        # Step 1: Collect news data
        logger.info("Collecting news data...")
        news_df = self.news_collector.collect_financial_news(
            query_terms=news_queries,
            days_back=days_back,
        )

        # Step 2: Collect market data
        logger.info("Collecting market data...")
        market_data = self.market_collector.collect_market_data(
            tickers=tickers,
            period=f"{days_back}d",
        )

        # Step 3: Store data if requested
        if store_data:
            logger.info("Storing collected data...")

            # Connect to database
            self.db_connector.connect()

            # Set up collections
            self.db_connector.setup_collections()

            # Store news data
            news_count = self.db_connector.store_news_data(news_df)
            logger.info(f"Stored {news_count} news articles")

            # Store market data
            market_count = self.db_connector.store_market_data(market_data)
            logger.info(f"Stored {market_count} market data records")

            # Close connection
            self.db_connector.close()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Data collection pipeline completed in {duration} seconds")

        return {
            "news_data": news_df,
            "market_data": market_data
        }

    def run_analysis_pipeline(
            self,
            news_df: Optional[pd.DataFrame] = None,
            store_results: bool = True
    ) -> pd.DataFrame:
        """
        Run the analysis pipeline on news data.
        """
        start_time = datetime.now()
        logger.info(f"Starting analysis pipeline at {start_time}")

        # If no news data provided, get it from the database
        if news_df is None:
            logger.info("No news data provided, fetching from database...")

            # Connect to database
            self.db_connector.connect()

            # TODO: Implement fetching news data from database
            # For now, return early with warning
            logger.warning("Fetching news data from database not implemented yet")
            return pd.DataFrame()

        # Step 1: Preprocess text
        logger.info("Preprocessing news text...")
        preprocessed_df = news_df.copy()

        # Apply text preprocessing
        preprocessed_df["preprocessed_title"] = preprocessed_df["title"].apply(
            self.text_processor.preprocess_text
        )
        preprocessed_df["preprocessed_content"] = preprocessed_df["content"].apply(
            self.text_processor.preprocess_text
        )

        # Step 2: Analyze sentiment
        logger.info("Analyzing sentiment...")
        result_df = self.sentiment_analyzer.analyze_news_dataframe(preprocessed_df)

        # Step 3: Store results if requested
        if store_results:
            logger.info("Storing analysis results...")

            self.store_analysis_in_vector_db(result_df)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Analysis pipeline completed in {duration} seconds")

        return result_df

    def run_full_pipeline(
            self,
            news_queries: Optional[List[str]] = None,
            tickers: Optional[List[str]] = None,
            days_back: int = None
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Run the full data pipeline.
        """
        # Step 1: Run data collection pipeline
        data = self.run_data_collection_pipeline(
            news_queries=news_queries,
            tickers=tickers,
            days_back=days_back,
            store_data=True
        )

        # Step 2: Run analysis pipeline
        analyzed_news = self.run_analysis_pipeline(
            news_df=data["news_data"],
            store_results=True
        )

        # Return results
        return {
            "news_data": analyzed_news,
            "market_data": data["market_data"]
        }

    def query_financial_insights(
            self,
            query: str,
            n_results: int = 5,
            min_sentiment: Optional[float] = None,
            max_sentiment: Optional[float] = None,
            use_enhanced_retrieval: bool = True,
    ) -> pd.DataFrame:
        """
        Query for financial insights based on vector similarity.

        Args:
            query: Query text.
            n_results: Number of results to return.
            min_sentiment: Minimum sentiment score (compound).
            max_sentiment: Maximum sentiment score (compound).
            use_enhanced_retrieval: Whether to use enhanced retrieval.

        Returns:
            DataFrame with results.
        """
        logger.info(f"Querying for financial insights: '{query}'")

        try:
            # Create filter criteria if sentiment bounds are provided
            filter_criteria = {}
            if min_sentiment is not None:
                filter_criteria["sentiment_compound"] = {"$gte": min_sentiment}
            if max_sentiment is not None:
                filter_criteria.setdefault("sentiment_compound", {})
                filter_criteria["sentiment_compound"]["$lte"] = max_sentiment

            # Query using appropriate method
            if use_enhanced_retrieval:
                logger.info("Using enhanced retrieval with hybrid search")
                result_df = self.enhanced_retriever.hybrid_search(
                    query=query,
                    n_results=20,  # Get more initial results for reranking
                    filter_criteria=filter_criteria,
                    final_results=n_results,
                )
            else:
                # Standard vector database query
                results = self.vector_db_manager.query_news(
                    query_text=query,
                    n_results=n_results,
                    filter_criteria=filter_criteria if filter_criteria else None,
                )

                # Process results into a DataFrame
                if not results["ids"][0]:
                    return pd.DataFrame()  # Return empty DataFrame if no results

                result_df = pd.DataFrame({
                    "id": results["ids"][0],
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0],
                    "distance": results["distances"][0] if "distances" in results else [0] * len(results["ids"][0]),
                })

            logger.info(f"Retrieved {len(result_df)} relevant articles")
            return result_df

        except Exception as e:
            logger.error(f"Error querying for financial insights: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def generate_financial_insights(
            self,
            query: str,
            n_results: int = 5,
            min_sentiment: Optional[float] = None,
            max_sentiment: Optional[float] = None,
    ) -> Dict:
        """
        Generate financial insights for a query.

        Args:
            query: Query text.
            n_results: Number of results to consider.
            min_sentiment: Minimum sentiment score.
            max_sentiment: Maximum sentiment score.

        Returns:
            Dictionary with generated insights.
        """
        logger.info(f"Generating financial insights for: '{query}'")

        # First retrieve relevant articles
        results_df = self.query_financial_insights(
            query=query,
            n_results=n_results,
            min_sentiment=min_sentiment,
            max_sentiment=max_sentiment
        )

        # Generate insights using LLM
        if len(results_df) > 0:
            insights = self.insight_generator.generate_insight(query, results_df)
            logger.info(f"Generated insights for '{query}' with {len(results_df)} articles")
        else:
            insights = {
                "query": query,
                "insight": "No relevant articles found to generate insights.",
                "sources": []
            }
            logger.info(f"No articles found for '{query}'")

        return insights

    def fetch_news_from_database(
            self,
            days_back: int = None,
            limit: int = 1000,
            query_filter: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Fetch news data from MongoDB.

        Args:
            days_back: Number of days to look back.
            limit: Maximum number of records to fetch.
            query_filter: Additional query filters.

        Returns:
            DataFrame with news data.
        """
        try:
            # Connect to database
            self.db_connector.connect()

            # Get news collection
            news_collection = self.db_connector.db[NEWS_COLLECTION]

            # Set up query filter
            filter_query = query_filter or {}

            # Add date filter if days_back is provided
            if days_back:
                from_date = datetime.now() - timedelta(days=days_back)
                filter_query.update({
                    "publishedAt": {"$gte": from_date.isoformat()}
                })

            # Fetch data
            cursor = news_collection.find(
                filter_query,
                limit=limit
            ).sort("publishedAt", pymongo.DESCENDING)

            # Convert to DataFrame
            news_df = pd.DataFrame(list(cursor))

            logger.info(f"Fetched {len(news_df)} news articles from database")

            # Close connection
            self.db_connector.close()

            return news_df

        except Exception as e:
            logger.error(f"Error fetching news from database: {str(e)}")
            return pd.DataFrame()

    def run_analysis_pipeline(
            self,
            news_df: Optional[pd.DataFrame] = None,
            store_results: bool = True,
            days_back: int = None
    ) -> pd.DataFrame:
        """
        Run the analysis pipeline on news data.
        """
        start_time = datetime.now()
        logger.info(f"Starting analysis pipeline at {start_time}")

        # If no news data provided, get it from the database
        if news_df is None:
            logger.info("No news data provided, fetching from database...")
            news_df = self.fetch_news_from_database(days_back=days_back)

            if news_df.empty:
                logger.warning("No news data found in database")
                return pd.DataFrame()

        # Step 1: Preprocess text
        logger.info("Preprocessing news text...")
        preprocessed_df = news_df.copy()

        # Apply text preprocessing
        preprocessed_df["preprocessed_title"] = preprocessed_df["title"].apply(
            self.text_processor.preprocess_text
        )
        preprocessed_df["preprocessed_content"] = preprocessed_df["content"].apply(
            self.text_processor.preprocess_text
        )

        # Step 2: Analyze sentiment
        logger.info("Analyzing sentiment...")
        result_df = self.sentiment_analyzer.analyze_news_dataframe(preprocessed_df)

        # Step 3: Store results if requested
        if store_results:
            logger.info("Storing analysis results...")
            self.store_analysis_in_vector_db(result_df)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Analysis pipeline completed in {duration} seconds")

        return result_df

    def generate_multi_agent_insights(
            self,
            query: str,
            n_results: int = 10,
            days_back: int = 30
    ) -> Dict:
        """
        Generate financial insights using the multi-agent system.

        Args:
            query: Query text.
            n_results: Number of news results to consider.
            days_back: Number of days to look back for market data.

        Returns:
            Dictionary with generated insights from multi-agent system.
        """
        logger.info(f"Generating multi-agent insights for: '{query}'")

        # Use the agent system to analyze data
        insights = self.agent_system.analyze_with_agents(
            query=query,
            pipeline_manager=self,
            n_news_results=n_results,
            days_back=days_back
        )

        logger.info(f"Generated multi-agent insights for '{query}'")
        return insights