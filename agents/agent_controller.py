"""
Agent controller interface for integrating with the main pipeline.
"""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from finsight.agents.base_agents import AgentController

# Setup logging
logger = logging.getLogger(__name__)


class AgentSystem:
    """Class for integrating the multi-agent system with the main pipeline."""

    def __init__(
            self,
            model: str = "mistral",
            base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the agent system.

        Args:
            model: LLM model to use.
            base_url: LLM API base URL.
        """
        self.controller = AgentController(
            llm_model=model,
            llm_base_url=base_url
        )

    def analyze_financial_data(
            self,
            query: str,
            news_articles: pd.DataFrame,
            market_data: Dict = None,
            financial_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Analyze financial data using the multi-agent system.

        Args:
            query: User query.
            news_articles: DataFrame of news articles.
            market_data: Dictionary of market data.
            financial_data: Optional dictionary of financial data.

        Returns:
            Dictionary with comprehensive analysis.
        """
        try:
            logger.info(f"Starting multi-agent analysis for query: '{query}'")

            # Convert news DataFrame to list of dictionaries
            news_list = []
            for _, row in news_articles.iterrows():
                news_dict = {
                    "document": row.get("content", ""),
                    "metadata": {
                        "source": row.get("source", {}).get("name", "Unknown") if isinstance(row.get("source"), dict) else "Unknown",
                        "publishedAt": row.get("publishedAt", ""),
                        "url": row.get("url", ""),
                        "sentiment_label": row.get("sentiment_label", "neutral"),
                        "sentiment_compound": row.get("sentiment_compound", 0.0)
                    }
                }
                news_list.append(news_dict)

            # Process with agent controller
            analysis = self.controller.process_query(
                query=query,
                news_articles=news_list,
                market_data=market_data,
                financial_data=financial_data
            )

            # Generate human-readable summary
            summary = self.controller.get_summary(analysis)

            # Return both the structured analysis and the summary
            return {
                "analysis": analysis,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Error in multi-agent analysis: {str(e)}")
            return {
                "error": str(e),
                "analysis": {},
                "summary": f"Error performing multi-agent analysis: {str(e)}"
            }

    def analyze_with_agents(
            self,
            query: str,
            pipeline_manager,
            n_news_results: int = 10,
            days_back: int = 30
    ) -> Dict:
        """
        Analyze using the simplified news sentiment agent.
        """
        try:
            # Step 1: Retrieve relevant news articles
            news_df = pipeline_manager.query_financial_insights(
                query=query,
                n_results=n_news_results
            )

            if news_df.empty:
                return {
                    "error": "No relevant news articles found",
                    "analysis": {},
                    "summary": "Unable to perform analysis: No relevant news articles found for the query."
                }

            # Step 2: Analyze using just the news sentiment agent
            return self.analyze_financial_data(
                query=query,
                news_articles=news_df,
                market_data=None  # No longer needed
            )

        except Exception as e:
            logger.error(f"Error in analyze_with_agents: {str(e)}")
            return {
                "error": str(e),
                "analysis": {},
                "summary": f"Error setting up analysis: {str(e)}"
            }