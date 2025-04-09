"""
AgentController for the Agentic RAG system.
"""

import logging
import json
from typing import Dict, List, Any, Optional
import pandas as pd

from finsight.agent.planner import AgentPlanner
from finsight.agent.tool_manager import ToolManager
from finsight.agent.executor import AgentExecutor

logger = logging.getLogger(__name__)


class AgenticRAGController:
    """Main controller for the Agentic RAG system."""

    def __init__(
            self,
            pipeline_manager,
            model: str = "llama3.2",
            base_url: str = "http://localhost:11434",
            max_iterations: int = 10,
            max_execution_time: int = 300  # 5 minutes
    ):
        """
        Initialize the agentic RAG controller.

        Args:
            pipeline_manager: Reference to the PipelineManager
            model: LLM model to use
            base_url: LLM API base URL
            max_iterations: Maximum iterations for executor
            max_execution_time: Maximum execution time in seconds
        """
        self.pipeline_manager = pipeline_manager

        # Initialize components
        self.planner = AgentPlanner(model=model, base_url=base_url)
        self.tool_manager = ToolManager(pipeline_manager)
        self.executor = AgentExecutor(
            tool_manager=self.tool_manager,
            planner=self.planner,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time
        )

        logger.info("Initialized AgenticRAGController")

    def process_query(self, query: str, context: str = "", return_debug_info: bool = False) -> Dict:
        """
        Process a user query with the agentic RAG system.

        Args:
            query: User query
            context: Optional context information
            return_debug_info: Whether to return debugging information

        Returns:
            Query results
        """
        logger.info(f"Processing query with agentic RAG: '{query}'")

        # Execute the agent
        execution_result = self.executor.execute(query, context)

        # Extract the answer
        answer = execution_result.get("answer", {})

        # Format the response
        response = {
            "query": query,
            "summary": answer.get("summary", "No summary generated"),
            "insights": answer.get("insights", []),
            "visualizations": answer.get("visualizations", []),
            "confidence": answer.get("confidence", 0.0)
        }

        # Include debugging information if requested
        if return_debug_info:
            response["debug"] = {
                "plan": execution_result.get("plan", {}),
                "execution_log": execution_result.get("execution_log", []),
                "execution_time": execution_result.get("execution_time", 0),
                "results": execution_result.get("results", [])
            }

        return response

    def get_visualizations(self, response: Dict) -> List[Dict]:
        """
        Extract visualizations from the response.

        Args:
            response: Response from process_query

        Returns:
            List of visualizations
        """
        return response.get("visualizations", [])

    def get_insights(self, response: Dict) -> List[Dict]:
        """
        Extract insights from the response.

        Args:
            response: Response from process_query

        Returns:
            List of insights
        """
        return response.get("insights", [])

    def format_response_for_display(self, response: Dict) -> Dict:
        """
        Format the response for display in a user interface.

        Args:
            response: Response from process_query

        Returns:
            Formatted response for display
        """
        # Extract summary
        summary = response.get("summary", "No summary available")

        # Extract key insights
        insights = response.get("insights", [])

        # Format visualizations for display
        visualizations = []
        for viz in response.get("visualizations", []):
            viz_type = viz.get("type", "")
            viz_data = viz.get("data", "")
            file_path = viz.get("file_path", "")

            if viz_type and file_path:
                viz_display = {
                    "type": viz_type,
                    "title": self._get_visualization_title(viz_type),
                    "data": file_path  # <- for Streamlit to load with `st.image(path)`
                }
                visualizations.append(viz_display)

        # Format data points
        data_points = []
        for point in response.get("data_points", []):
            if point.get("type") == "source":
                source_data = point.get("value", {})

                if source_data:
                    data_points.append({
                        "type": "source",
                        "name": source_data.get("source", "Unknown"),
                        "url": source_data.get("url", "")
                    })

        # Format confidence
        confidence = response.get("confidence", 0.0)
        confidence_level = "High" if confidence >= 0.7 else "Medium" if confidence >= 0.4 else "Low"

        return {
            "query": response.get("query", ""),
            "summary": summary,
            "insights": insights,
            "visualizations": visualizations,
            "data_points": data_points,
            "confidence": {
                "value": confidence,
                "level": confidence_level
            }
        }

    def _get_visualization_title(self, viz_type: str) -> str:
        """Get a display title for a visualization type."""
        titles = {
            "create_price_chart": "Price Chart",
            "create_sentiment_chart": "Sentiment Analysis",
            "create_technical_dashboard": "Technical Analysis Dashboard"
        }
        return titles.get(viz_type, "Visualization")

    def extract_ticker_from_query(self, query: str) -> Optional[str]:
        """
        Extract a potential ticker symbol from a query.

        Args:
            query: User query

        Returns:
            Extracted ticker or None
        """
        import re

        # Look for ticker patterns
        ticker_match = re.search(r'\b[A-Z]{1,5}-?USD?\b', query)
        if ticker_match:
            return ticker_match.group(0)

        # Look for common company names
        company_map = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'tesla': 'TSLA',
            'facebook': 'META',
            'meta': 'META',
            'nvidia': 'NVDA',
            'netflix': 'NFLX',
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'jpmorgan': 'JPM',
            'goldman sachs': 'GS',
            'coca cola': 'KO',
            'coca-cola': 'KO',
            'disney': 'DIS',
            'boeing': 'BA',
            'ibm': 'IBM',
            'intel': 'INTC',
            'walmart': 'WMT',
            'exxon': 'XOM',
            'johnson & johnson': 'JNJ',
            'verizon': 'VZ',
            'at&t': 'T',
            'ford': 'F',
            'general motors': 'GM',
            'gm': 'GM'
        }

        query_lower = query.lower()
        for company, ticker in company_map.items():
            if company in query_lower:
                return ticker

        return None