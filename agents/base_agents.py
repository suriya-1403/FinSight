"""
Multi-agent system for financial analysis.
"""

import logging
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


def parse_llm_response(response: Any, default_structure: Dict, logger=None) -> Dict:
    """
    Safely parse an LLM response to extract JSON, with robust error handling.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Extract text from response
        if isinstance(response, dict) and "text" in response:
            response_text = response["text"]
        else:
            response_text = str(response)

        # Try to find JSON in markdown code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.info(f"Found JSON in code block: {json_str[:100]}...")
        else:
            # If no code block, try to find JSON-like structure using regex
            json_match = re.search(r"(\{[\s\S]*?\})", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                logger.info(f"Found JSON-like structure: {json_str[:100]}...")
            else:
                # Use the whole text as last resort
                json_str = response_text.strip()
                logger.info("Using full response text for parsing")

        # Clean up common issues
        # Remove any leading/trailing backticks
        json_str = re.sub(r'^`+|`+$', '', json_str)

        # Remove comments
        json_str = re.sub(r'//.*?(\n|$)', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        # Fix trailing commas before closing brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Try to parse the JSON
        try:
            result = json.loads(json_str)

            # Ensure all required keys from default structure exist
            for key in default_structure:
                if key not in result:
                    result[key] = default_structure[key]
                    logger.warning(f"Added missing key '{key}' with default value")

            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}. Text: {json_str[:100]}...")

            # If all else fails, return the default structure
            return default_structure

    except Exception as e:
        logger.error(f"Unexpected error parsing LLM response: {str(e)}")
        return default_structure

class BaseAgent(ABC):
    """Abstract base class for financial analysis agents."""

    def __init__(
            self,
            name: str,
            model: str = "mistral",
            base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent name.
            model: LLM model to use.
            base_url: LLM API base URL.
        """
        self.name = name
        self.model_name = model
        self.base_url = base_url

        # Initialize LLM
        try:
            self.llm = OllamaLLM(model=model, base_url=base_url)
            logger.info(f"Initialized agent {name} with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing LLM for agent {name}: {str(e)}")
            self.llm = None

        # Initialize prompt template
        self.prompt_template = self._get_prompt_template()

        # Create LLM chain
        if self.llm:
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        else:
            self.chain = None



    @abstractmethod
    def _get_prompt_template(self) -> PromptTemplate:
        """Get the prompt template for the agent."""
        pass

    @abstractmethod
    def analyze(self, data: Any) -> Dict:
        """Analyze data and return results."""
        pass


class NewsSentimentAgent(BaseAgent):
    """Agent for analyzing news sentiment and relevance to the query."""

    def __init__(
            self,
            name: str,
            model: str = "mistral",
            base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the base agent with LangSmith tracing.
        """
        import os
        self.name = name
        self.model_name = model
        self.base_url = base_url

        # Set up LangSmith tracing
        from langchain.callbacks.tracers import LangChainTracer
        from langchain.callbacks.manager import CallbackManager

        tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        callbacks = []

        if tracing_enabled:
            project_name = os.getenv("LANGCHAIN_PROJECT", "default")
            try:
                tracer = LangChainTracer(project_name=project_name)
                callbacks.append(tracer)
                logger.info(f"Enabled LangSmith tracing for project: {project_name}")
            except Exception as e:
                logger.error(f"Failed to set up LangSmith tracing: {str(e)}")

        # Initialize LLM with callbacks
        try:
            self.llm = OllamaLLM(
                model=model,
                base_url=base_url,
                callbacks=callbacks if callbacks else None
            )
            logger.info(f"Initialized agent {name} with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing LLM for agent {name}: {str(e)}")
            self.llm = None

        # Initialize prompt template
        self.prompt_template = self._get_prompt_template()

        # Create LLM chain with callbacks
        if self.llm:
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
                callbacks=callbacks if callbacks else None
            )
        else:
            self.chain = None

    def _get_prompt_template(self) -> PromptTemplate:
        """Get the prompt template for the agent."""
        return PromptTemplate(
            input_variables=["query", "articles"],
            template="""You are a financial news sentiment analyst. Analyze the sentiment and relevance of these news articles related to the query.

USER QUERY: {query}

NEWS ARTICLES:
{articles}

Provide a detailed sentiment analysis with the following information:
1. Overall Sentiment: Analyze the collective sentiment (positive, negative, or neutral) across all articles.
2. Sentiment Breakdown: Identify key positive and negative factors mentioned.
3. Sentiment Timeline: Detect any sentiment shifts over time in the articles.
4. Key Entities: Identify important companies, people, or assets mentioned and their sentiment context.
5. Relevance Analysis: Rate how relevant each article is to the query.

Format your analysis as JSON with the following structure:
{{
    "overall_sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "key_factors": {{
        "positive": ["factor1", "factor2", ...],
        "negative": ["factor1", "factor2", ...]
    }},
    "sentiment_timeline": [
        {{"date": "date1", "sentiment": "sentiment1", "key_point": "summary1"}},
        ...
    ],
    "key_entities": [
        {{"entity": "entity1", "sentiment": "sentiment1", "mention_count": count1, "context": "context1"}},
        ...
    ],
    "article_relevance": [
        {{"source": "source1", "relevance_score": 0.0-1.0, "key_relevance_factor": "factor1"}},
        ...
    ]
}}

Ensure your analysis is fact-based and derived only from the provided articles.
"""
        )

    def analyze(self, data: Dict) -> Dict:
        """
        Analyze news articles and return sentiment analysis.
        """
        try:
            query = data.get("query", "")
            articles = data.get("articles", [])

            # Format articles for the prompt
            articles_text = ""
            for i, article in enumerate(articles, 1):
                source = article.get("metadata", {}).get("source", "Unknown")
                date = article.get("metadata", {}).get("publishedAt", "Unknown date")
                content = article.get("document", "")
                articles_text += f"ARTICLE {i} - SOURCE: {source} (DATE: {date})\n{content}\n\n"

            # Generate analysis using LLM
            if self.chain:
                logger.info(f"Sending request to LLM with query: '{query}' and {len(articles)} articles")
                response = self.chain.invoke({"query": query, "articles": articles_text})

                # Log the raw response
                if isinstance(response, dict) and "text" in response:
                    response_text = response["text"]
                    # Log first 500 characters for debugging
                    logger.info(f"Raw LLM response (first 500 chars): {response_text[:500]}")
                else:
                    response_text = str(response)
                    logger.info(f"Raw LLM response (str) (first 500 chars): {response_text[:500]}")

                # Parse JSON from response - try multiple methods
                try:
                    # Method 1: Look for code blocks
                    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        logger.info(f"Found JSON in code block: {json_str[:200]}...")
                        try:
                            analysis = json.loads(json_str)
                            logger.info("Successfully parsed JSON from code block")
                            return analysis
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON from code block: {e}")
                            # Continue to the next method

                    # Method 2: Look for JSON-like structure
                    json_match = re.search(r"\{[\s\S]*\}", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0).strip()
                        logger.info(f"Found JSON-like structure: {json_str[:200]}...")
                        try:
                            analysis = json.loads(json_str)
                            logger.info("Successfully parsed JSON from structure")
                            return analysis
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON from structure: {e}")
                            # Continue to the next method

                    # Method 3: Try to clean up and parse
                    clean_text = re.sub(r'[\n\r\t]', ' ', response_text)
                    clean_text = re.sub(r'```json|```', '', clean_text)
                    json_match = re.search(r"\{[\s\S]*\}", clean_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0).strip()
                        logger.info(f"Found JSON after cleaning: {json_str[:200]}...")
                        try:
                            analysis = json.loads(json_str)
                            logger.info("Successfully parsed JSON after cleaning")
                            return analysis
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing cleaned JSON: {e}")
                            # Fall back to default

                    logger.error("All JSON parsing methods failed, using fallback")
                    return self._generate_fallback_analysis(query, articles)

                except Exception as e:
                    logger.error(f"Unexpected error during JSON parsing: {str(e)}")
                    return self._generate_fallback_analysis(query, articles)
            else:
                logger.warning("LLM chain not available, using fallback analysis")
                return self._generate_fallback_analysis(query, articles)

        except Exception as e:
            logger.error(f"Error in NewsSentimentAgent.analyze: {str(e)}")
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "error": str(e),
            }

    def _generate_fallback_analysis(self, query: str, articles: List[Dict]) -> Dict:
        """
        Generate a fallback analysis when LLM is not available.

        Args:
            query: User query.
            articles: List of articles.

        Returns:
            Dictionary with fallback analysis.
        """
        # Count sentiments from metadata
        sentiments = [a.get("metadata", {}).get("sentiment_label", "neutral") for a in articles]
        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        neutral = sentiments.count("neutral")

        overall_sentiment = "neutral"
        if positive > negative and positive > neutral:
            overall_sentiment = "positive"
        elif negative > positive and negative > neutral:
            overall_sentiment = "negative"

        # Calculate confidence
        total = len(sentiments)
        if total > 0:
            if overall_sentiment == "positive":
                confidence = positive / total
            elif overall_sentiment == "negative":
                confidence = negative / total
            else:
                confidence = neutral / total
        else:
            confidence = 0.0

        return {
            "overall_sentiment": overall_sentiment,
            "confidence": confidence,
            "key_factors": {
                "positive": ["Data not available - fallback mode"],
                "negative": ["Data not available - fallback mode"]
            },
            "sentiment_timeline": [],
            "key_entities": [],
            "article_relevance": [
                {"source": a.get("metadata", {}).get("source", "Unknown"), "relevance_score": 0.5, "key_relevance_factor": "Fallback mode"}
                for a in articles
            ]
        }

class TechnicalAnalysisAgent(BaseAgent):
    """Agent for technical market analysis."""

    def _get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["query", "market_data"],
            template="""
You are a technical analysis expert. Analyze the market data below for insights.

QUERY: {query}

MARKET DATA:
{market_data}

Provide a detailed technical analysis focusing on:
1. Primary trend direction (bullish, bearish, or sideways)
2. Key support and resistance levels
3. Notable technical patterns (e.g., head and shoulders, double top)
4. Volume analysis
5. Technical indicators (RSI, MACD, Moving Averages)
6. Trading recommendation

Format your response as valid JSON:

```json
{
    "trend_direction": "bullish|bearish|sideways",
    "confidence": 0.7,
    "support_levels": [42000, 40500, 38200],
    "resistance_levels": [45000, 47500, 50000],
    "patterns_identified": [
        {"pattern": "bullish flag", "strength": 0.8, "timeframe": "daily"}
    ],
    "volume_analysis": "increasing on upward moves",
    "technical_indicators": {
        "rsi": {"value": 65, "interpretation": "approaching overbought"},
        "macd": {"value": "positive", "interpretation": "bullish momentum"}
    },
    "recommendation": "buy|sell|hold",
    "recommendation_timeframe": "short|medium|long"
}```
Your analysis must be based solely on the provided market data.
"""
)

    def analyze(self, data: Dict) -> Dict:
        query = data.get("query", "")
        market_data = data.get("market_data", {})

        # Format market data as text
        formatted_data = self._format_market_data(market_data)

        default_result = {
            "trend_direction": "sideways",
            "confidence": 0.5,
            "support_levels": [],
            "resistance_levels": [],
            "patterns_identified": [],
            "volume_analysis": "Insufficient data",
            "technical_indicators": {},
            "recommendation": "hold",
            "recommendation_timeframe": "medium"
        }

        if not self.chain:
            return default_result

        # Call LLM
        try:
            logger.info(f"Analyzing technical data for query: '{query}'")
            response = self.chain.invoke({"query": query, "market_data": formatted_data})

            # Parse the response
            return parse_llm_response(response, default_result, logger)

        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return default_result

    def _format_market_data(self, market_data: Dict) -> str:
        """Format market data for the prompt."""
        formatted_text = ""

        for ticker, df in market_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            formatted_text += f"TICKER: {ticker}\n"

            # Calculate basic metrics
            if 'Close' in df.columns:
                current_price = df['Close'].iloc[-1]
                price_change = ((current_price / df['Close'].iloc[0]) - 1) * 100

                # Calculate moving averages
                if len(df) >= 50:
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['MA50'] = df['Close'].rolling(window=50).mean()

                    ma20 = df['MA20'].iloc[-1]
                    ma50 = df['MA50'].iloc[-1]

                    formatted_text += f"Current Price: {current_price:.2f}\n"
                    formatted_text += f"Price Change: {price_change:.2f}%\n"
                    formatted_text += f"20-day MA: {ma20:.2f}\n"
                    formatted_text += f"50-day MA: {ma50:.2f}\n"

                    # Add MA relationship
                    if ma20 > ma50:
                        formatted_text += "MA Relationship: 20-day above 50-day (bullish)\n"
                    else:
                        formatted_text += "MA Relationship: 50-day above 20-day (bearish)\n"

                # Calculate RSI if we have enough data
                if len(df) >= 14:
                    delta = df['Close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)

                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()

                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                    current_rsi = rsi.iloc[-1]
                    # Convert to scalar if it's a Series or DataFrame element
                    if hasattr(current_rsi, 'item'):
                        current_rsi = current_rsi.item()
                    formatted_text += f"RSI (14): {current_rsi:.2f}\n"

                    if current_rsi > 70:
                        formatted_text += "RSI indicates: Overbought\n"
                    elif current_rsi < 30:
                        formatted_text += "RSI indicates: Oversold\n"
                    else:
                        formatted_text += "RSI indicates: Neutral\n"

            # Add recent price data
            formatted_text += "\nRecent price data (last 7 days):\n"
            formatted_text += df.tail(7).to_string() + "\n\n"

        return formatted_text


class AgentController:
    """Class for orchestrating multi-agent interactions."""

    def __init__(self, llm_model: str = "llama3.2", llm_base_url: str = "http://localhost:11434"):
        """
        Initialize the agent controller.

        Args:
            llm_model: LLM model to use.
            llm_base_url: LLM API base URL.
        """
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url

        # Initialize only the news sentiment agent
        self.news_sentiment_agent = NewsSentimentAgent(
            name="NewsSentimentAgent",
            model=llm_model,
            base_url=llm_base_url
        )

    def process_query(
            self,
            query: str,
            news_articles: List[Dict],
            market_data: Dict = None,
            financial_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Process a query using the news sentiment agent.

        Args:
            query: User query.
            news_articles: List of news articles.
            market_data: Market data (ignored in this simplified version).
            financial_data: Financial data (ignored in this simplified version).

        Returns:
            Dictionary with sentiment analysis.
        """
        logger.info(f"Processing query with simplified agent system: '{query}'")
        logger.info(f"News articles count: {len(news_articles)}")

        # Process with news sentiment agent only
        logger.info("Starting news sentiment analysis...")
        sentiment_analysis = self.news_sentiment_agent.analyze({
            "query": query,
            "articles": news_articles
        })
        logger.info(f"News sentiment analysis complete. Result type: {type(sentiment_analysis)}")

        # Return a simplified analysis with just the sentiment
        comprehensive_analysis = {
            "query": query,
            "timestamp": pd.Timestamp.now().isoformat(),
            "news_sentiment": sentiment_analysis
        }

        logger.info("Analysis complete")
        return comprehensive_analysis

    def get_summary(self, comprehensive_analysis: Dict) -> str:
        """
        Generate a human-readable summary of the sentiment analysis.

        Args:
            comprehensive_analysis: Dictionary with analysis.

        Returns:
            String with human-readable summary.
        """
        query = comprehensive_analysis.get("query", "")
        timestamp = comprehensive_analysis.get("timestamp", "")
        sentiment = comprehensive_analysis.get("news_sentiment", {})

        summary = f"FINANCIAL INSIGHTS SUMMARY\n"
        summary += f"Query: {query}\n"
        summary += f"Generated: {timestamp}\n\n"

        # News Sentiment
        summary += "NEWS SENTIMENT\n"
        summary += f"Overall: {sentiment.get('overall_sentiment', 'Unknown')}"
        summary += f" (Confidence: {sentiment.get('confidence', 0.0):.2f})\n"

        if 'key_factors' in sentiment:
            pos_factors = sentiment['key_factors'].get('positive', [])
            neg_factors = sentiment['key_factors'].get('negative', [])

            summary += "\nKEY FACTORS\n"
            if pos_factors:
                summary += "Positive factors: " + ", ".join(pos_factors[:3]) + "\n"
            if neg_factors:
                summary += "Negative factors: " + ", ".join(neg_factors[:3]) + "\n"

        if 'key_entities' in sentiment and sentiment['key_entities']:
            summary += "\nKEY ENTITIES\n"
            entities = sentiment['key_entities'][:3]
            for entity in entities:
                entity_name = entity.get('entity', 'Unknown')
                entity_sentiment = entity.get('sentiment', 'neutral')
                context = entity.get('context', '')
                summary += f"- {entity_name} ({entity_sentiment}): {context}\n"

        return summary