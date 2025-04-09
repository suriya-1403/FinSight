"""
ToolManager manages the available tools that the agent can use.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
import inspect

logger = logging.getLogger(__name__)


class ToolManager:
    """Class for managing and executing tools for the agentic RAG system."""

    def __init__(self, pipeline_manager=None):
        """
        Initialize the tool manager.

        Args:
            pipeline_manager: Reference to the PipelineManager instance
        """
        self.pipeline_manager = pipeline_manager
        self.tools = {}
        self.tool_descriptions = {}

        # Register default tools if pipeline_manager is provided
        if pipeline_manager:
            self._register_default_tools()

    def register_tool(self, name: str, function: Callable, description: str):
        """
        Register a new tool.

        Args:
            name: Tool name
            function: Tool function
            description: Tool description
        """
        self.tools[name] = function
        self.tool_descriptions[name] = description
        logger.info(f"Registered tool: {name}")

    def execute_tool(self, name: str, parameters: Dict) -> Dict:
        """
        Execute a tool with the given parameters.

        Args:
            name: Tool name
            parameters: Tool parameters

        Returns:
            Tool execution results
        """
        if name not in self.tools:
            error_msg = f"Tool not found: {name}"
            logger.error(error_msg)
            return {"error": error_msg}

        try:
            # Get the tool function
            tool_function = self.tools[name]

            # Get valid parameters for the function
            sig = inspect.signature(tool_function)
            valid_params = {}

            # Filter parameters to only include those expected by the function
            for param_name, param in sig.parameters.items():
                if param_name in parameters:
                    valid_params[param_name] = parameters[param_name]
                elif param.default is param.empty:
                    # Required parameter is missing
                    error_msg = f"Missing required parameter for {name}: {param_name}"
                    logger.error(error_msg)
                    return {"error": error_msg}

            # Execute the tool function with the valid parameters
            logger.info(f"Executing tool: {name} with parameters: {valid_params}")
            result = tool_function(**valid_params)

            # Process the result for consistent output
            processed_result = self._process_tool_result(result)

            return {
                "tool": name,
                "parameters": parameters,
                "result": processed_result
            }

        except Exception as e:
            error_msg = f"Error executing tool {name}: {str(e)}"
            logger.error(error_msg)
            return {
                "tool": name,
                "parameters": parameters,
                "error": error_msg
            }

    def get_available_tools(self) -> Dict[str, str]:
        """
        Get available tools with descriptions.

        Returns:
            Dictionary of tool names mapped to descriptions
        """
        return self.tool_descriptions

    # Add this to your tool manager class

    def _register_default_tools(self):
        """Register default tools from the pipeline manager."""
        pm = self.pipeline_manager

        # News and market data retrieval tools
        self.register_tool(
            "query_financial_insights",
            pm.query_financial_insights,
            "Retrieve relevant financial news articles based on a query"
        )

        self.register_tool(
            "collect_market_data",
            pm.market_collector.collect_market_data,
            "Collect market data for specific tickers over a time period"
        )

        # Analysis tools - Use wrapper functions with proper error handling
        self.register_tool(
            "news_sentiment_agent",
            lambda query, news_articles=None: self._analyze_sentiment(query, news_articles),
            "Analyze sentiment in news articles"
        )

        self.register_tool(
            "technical_analysis_agent",
            lambda query, market_data=None: self._analyze_technical(query, market_data),
            "Perform technical analysis on market data"
        )

        self.register_tool(
            "forecast_prices",
            lambda ticker, price_data=None, days=7: self._forecast_prices(ticker, price_data, days),
            "Forecast future prices for a ticker"
        )

        # Vector database tools
        self.register_tool(
            "hybrid_search",
            lambda query, n_results=10, filter_criteria=None:
            pm.enhanced_retriever.hybrid_search(
                query=query,
                n_results=n_results,
                filter_criteria=filter_criteria
            ),
            "Perform hybrid search combining vector similarity and BM25"
        )

        # Visualization tools - Use wrapper functions with proper error handling
        self.register_tool(
            "create_price_chart",
            lambda price_data=None, ticker=None, forecast=None:
            self._create_price_chart(price_data, ticker, forecast),
            "Create a price chart with optional forecast"
        )

        self.register_tool(
            "create_sentiment_chart",
            lambda sentiment_data=None:
            self._create_sentiment_chart(sentiment_data),
            "Create a sentiment analysis visualization"
        )

        self.register_tool(
            "create_technical_dashboard",
            lambda price_data=None, ticker=None, technical_analysis=None:
            self._create_technical_dashboard(price_data, ticker, technical_analysis),
            "Create a technical analysis dashboard"
        )

        # Insight generation
        self.register_tool(
            "generate_insight",
            lambda query, relevant_articles=None:
            self._generate_insight(query, relevant_articles),
            "Generate financial insights based on retrieved articles"
        )

    def _create_price_chart(self, price_data=None, ticker=None, forecast=None):
        """Wrapper for price chart creation with error handling."""
        pm = self.pipeline_manager

        # If price_data not provided, retrieve it
        if price_data is None and ticker is not None:
            try:
                market_data = pm.market_collector.collect_market_data(
                    tickers=[ticker],
                    period="60d"
                )

                if ticker in market_data:
                    price_data = market_data[ticker]
                    logger.info(f"Retrieved price data for {ticker}")
                else:
                    return {"error": f"No price data available for {ticker}"}
            except Exception as e:
                logger.error(f"Error retrieving price data: {str(e)}")
                return {"error": str(e)}
        elif price_data is None and ticker is None:
            return {"error": "Both price_data and ticker are missing"}

        # Create price chart
        try:
            base64_img, path = pm.visualizer.create_price_chart(
                price_data=price_data,
                ticker=ticker,
                forecast=forecast
            )
            return {
                "base64": base64_img,
                "path": path
            }
        except Exception as e:
            logger.error(f"Error creating price chart: {str(e)}")
            return {"error": str(e)}

    def _create_sentiment_chart(self, sentiment_data=None):
        """Wrapper for sentiment chart creation with error handling."""
        pm = self.pipeline_manager

        # If sentiment_data not provided, create default data
        if sentiment_data is None:
            sentiment_data = {
                "positive_count": 1,
                "negative_count": 1,
                "neutral_count": 1,
                "key_factors": {
                    "positive": ["No positive factors identified"],
                    "negative": ["No negative factors identified"]
                }
            }

        # Create sentiment chart
        try:
            base64_img, path = pm.visualizer.create_sentiment_chart(sentiment_data)
            return {
    "base64": base64_img,
    "path": path
}
        except Exception as e:
            logger.error(f"Error creating sentiment chart: {str(e)}")
            return {"error": str(e)}

    def _create_technical_dashboard(self, price_data=None, ticker=None, technical_analysis=None):
        """Wrapper for technical dashboard creation with error handling."""
        pm = self.pipeline_manager

        # If price_data not provided, retrieve it
        if price_data is None and ticker is not None:
            try:
                market_data = pm.market_collector.collect_market_data(
                    tickers=[ticker],
                    period="60d"
                )

                if ticker in market_data:
                    price_data = market_data[ticker]
                    logger.info(f"Retrieved price data for {ticker} with {len(price_data)} records")
                else:
                    return {"error": f"No price data available for {ticker}"}
            except Exception as e:
                logger.error(f"Error retrieving price data: {str(e)}")
                return {"error": str(e)}
        elif price_data is None and ticker is None:
            # Try to extract ticker from context and get data
            ticker = self._extract_ticker_from_query("Tesla") or "TSLA"
            try:
                market_data = pm.market_collector.collect_market_data(
                    tickers=[ticker],
                    period="60d"
                )
                price_data = market_data[ticker]
                logger.info(f"Retrieved fallback price data for {ticker}")
            except Exception as e:
                logger.error(f"Error retrieving fallback price data: {str(e)}")
                return {"error": "Missing price data and ticker for technical dashboard"}

        # If technical_analysis not provided, create default data
        if technical_analysis is None:
            technical_analysis = {
                "trend_direction": "sideways",
                "recommendation": "hold",
                "recommendation_timeframe": "medium",
                "support_levels": [],
                "resistance_levels": []
            }

        # Create technical dashboard
        try:
            base64_img, path = pm.visualizer.create_technical_dashboard(
                price_data=price_data,
                ticker=ticker,
                technical_analysis=technical_analysis
            )
            return  {
                "base64": base64_img,
                "path": path
            }
        except Exception as e:
            logger.error(f"Error creating technical dashboard: {str(e)}")
            return {"error": str(e)}

    def _generate_insight(self, query, relevant_articles=None):
        """Wrapper for insight generation with error handling."""
        pm = self.pipeline_manager

        # If relevant_articles not provided, retrieve them
        if relevant_articles is None:
            try:
                relevant_df = pm.query_financial_insights(
                    query=query,
                    n_results=5
                )
                logger.info(f"Retrieved {len(relevant_df)} articles for insight generation")
                relevant_articles = relevant_df
            except Exception as e:
                logger.error(f"Error retrieving articles for insights: {str(e)}")
                return {
                    "query": query,
                    "insight": f"Unable to generate insights due to an error: {str(e)}",
                    "sources": []
                }

        # Generate insights
        try:
            return pm.insight_generator.generate_insight(query, relevant_articles)
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                "query": query,
                "insight": f"Unable to generate insights due to an error: {str(e)}",
                "sources": []
            }

    # Add these helper methods to handle parameter requirements and errors gracefully
    def _analyze_sentiment(self, query, news_articles=None):
        """Wrapper for sentiment analysis to handle missing parameters."""
        pm = self.pipeline_manager

        # If news_articles not provided, retrieve them
        if news_articles is None:
            try:
                news_df = pm.query_financial_insights(
                    query=query,
                    n_results=10
                )

                # Convert to format expected by sentiment agent
                news_articles = []
                for _, row in news_df.iterrows():
                    news_dict = {
                        "document": row.get("document", ""),
                        "metadata": row.get("metadata", {})
                    }
                    news_articles.append(news_dict)

                logger.info(f"Retrieved {len(news_articles)} news articles for sentiment analysis")
            except Exception as e:
                logger.error(f"Error retrieving news articles: {str(e)}")
                news_articles = []

        # Get sentiment agent using helper function
        agent = self._get_agent('sentiment')

        # Now analyze sentiment
        try:
            if agent:
                result = agent.analyze({
                    "query": query,
                    "articles": news_articles
                })
                return result
            else:
                logger.error("No sentiment agent available")
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.0,
                    "error": "No sentiment agent available",
                    "key_factors": {
                        "positive": ["Unable to analyze"],
                        "negative": ["Unable to analyze"]
                    }
                }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "error": str(e),
                "key_factors": {
                    "positive": ["Unable to analyze"],
                    "negative": ["Unable to analyze"]
                }
            }

    def _analyze_technical(self, query, market_data=None):
        """Wrapper for technical analysis to handle missing parameters."""
        pm = self.pipeline_manager

        # If market_data not provided, retrieve it based on ticker in query
        if market_data is None:
            try:
                # Try to extract ticker from query
                ticker = self._extract_ticker_from_query(query)
                if ticker:
                    market_data = pm.market_collector.collect_market_data(
                        tickers=[ticker],
                        period="30d"
                    )
                    logger.info(f"Retrieved market data for {ticker}")
                else:
                    return {"error": "No ticker identified in query"}
            except Exception as e:
                logger.error(f"Error retrieving market data: {str(e)}")
                return {"error": str(e)}

        # Get technical agent using helper function
        agent = self._get_agent('technical')

        # Now perform technical analysis
        try:
            if agent:
                result = agent.analyze({
                    "query": query,
                    "market_data": market_data
                })
                return result
            else:
                logger.error("No technical agent available")
                return {
                    "trend_direction": "sideways",
                    "confidence": 0.5,
                    "recommendation": "hold",
                    "recommendation_timeframe": "medium",
                    "error": "No technical agent available"
                }
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return {
                "trend_direction": "sideways",
                "confidence": 0.5,
                "recommendation": "hold",
                "recommendation_timeframe": "medium",
                "error": str(e)
            }

    def _forecast_prices(self, ticker, price_data=None, days=7):
        """Wrapper for price forecasting to handle missing parameters."""
        pm = self.pipeline_manager

        # If price_data not provided, retrieve it
        if price_data is None:
            try:
                market_data = pm.market_collector.collect_market_data(
                    tickers=[ticker],
                    period="60d"  # Get enough data for forecasting
                )

                if ticker in market_data:
                    price_data = market_data[ticker]
                    logger.info(f"Retrieved price data for {ticker} with {len(price_data)} records")
                else:
                    return {"error": f"No price data available for {ticker}"}
            except Exception as e:
                logger.error(f"Error retrieving price data: {str(e)}")
                return {"error": str(e)}

        # Now generate forecast
        try:
            return pm.forecasting_engine.forecast_prices(
                ticker=ticker,
                price_data=price_data,
                forecast_days=days
            )
        except Exception as e:
            logger.error(f"Error in price forecasting: {str(e)}")
            return {"error": str(e)}

    def _extract_ticker_from_query(self, query):
        """Extract a ticker symbol from the query."""
        import re

        # Look for ticker patterns
        ticker_match = re.search(r'\b[A-Z]{1,5}-?USD?\b', query)
        if ticker_match:
            return ticker_match.group(0)

        # Common company name mappings
        company_map = {
            'tesla': 'TSLA',
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'google': 'GOOGL',
            'facebook': 'META',
            'meta': 'META',
            'netflix': 'NFLX',
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD'
        }

        query_lower = query.lower()
        for company, ticker in company_map.items():
            if company in query_lower:
                return ticker

        # Default to a common ticker if nothing found
        if 'stock' in query_lower:
            return 'SPY'  # S&P 500 ETF
        elif 'crypto' in query_lower or 'bitcoin' in query_lower:
            return 'BTC-USD'

        return None

    def _process_tool_result(self, result: Any) -> Any:
        """
        Process tool results for consistent output format.

        Args:
            result: Raw tool result

        Returns:
            Processed result
        """
        # Handle pandas DataFrames
        if hasattr(result, 'to_dict'):
            try:
                # For DataFrames, return a sample and metadata instead of full data
                if hasattr(result, 'shape'):
                    # This is a DataFrame
                    return {
                        "type": "dataframe",
                        "shape": result.shape,
                        "columns": list(result.columns),
                        "sample": result.head(3).to_dict('records'),
                        "summary": f"DataFrame with {result.shape[0]} rows and {result.shape[1]} columns"
                    }
            except Exception as e:
                logger.warning(f"Error processing DataFrame: {str(e)}")
                return {"type": "dataframe", "error": str(e)}

        # Handle other non-serializable objects
        if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
            try:
                # Try to convert to string representation
                return {
                    "type": type(result).__name__,
                    "string_representation": str(result),
                    "summary": f"Non-serializable object of type {type(result).__name__}"
                }
            except Exception as e:
                logger.warning(f"Error processing non-serializable object: {str(e)}")
                return {"type": type(result).__name__, "error": str(e)}

        # Return serializable objects as is
        return result

    def _get_agent(self, agent_type):
        """
        Helper function to get agent reference regardless of structure.

        Args:
            agent_type: Type of agent to get ('sentiment', 'technical', etc.)

        Returns:
            Agent instance or None
        """
        pm = self.pipeline_manager

        if agent_type == 'sentiment':
            # Option 1: Direct access if the agent exists on pipeline_manager
            if hasattr(pm, "news_sentiment_agent"):
                return pm.news_sentiment_agent

            # Option 2: Access through the agent_system
            if hasattr(pm, "agent_system"):
                # Check multiple possible attribute names
                for attr_name in ["news_sentiment_agent", "sentiment_agent", "NewsSentimentAgent"]:
                    if hasattr(pm.agent_system, attr_name):
                        return getattr(pm.agent_system, attr_name)

                # If not found directly, try master if available
                if hasattr(pm, "master") and hasattr(pm.master, "sentiment_agent"):
                    return pm.master.sentiment_agent

            # Option 3: Create a new instance
            from finsight.agents.base_agents import NewsSentimentAgent
            logger.info("Creating new sentiment agent instance")
            return NewsSentimentAgent(name="SentimentAgent", model="mistral")

        elif agent_type == 'technical':
            # Option 1: Direct access if the agent exists on pipeline_manager
            if hasattr(pm, "technical_agent"):
                return pm.technical_agent

            # Option 2: Access through the agent_system
            if hasattr(pm, "agent_system"):
                # Check multiple possible attribute names
                for attr_name in ["technical_agent", "TechnicalAnalysisAgent"]:
                    if hasattr(pm.agent_system, attr_name):
                        return getattr(pm.agent_system, attr_name)

                # If not found directly, try master if available
                if hasattr(pm, "master") and hasattr(pm.master, "technical_agent"):
                    return pm.master.technical_agent

            # Option 3: Create a new instance
            from finsight.agents.base_agents import TechnicalAnalysisAgent
            logger.info("Creating new technical agent instance")
            return TechnicalAnalysisAgent(name="TechnicalAgent", model="mistral")

        else:
            logger.warning(f"Unknown agent type: {agent_type}")
            return None