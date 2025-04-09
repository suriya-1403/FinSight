"""
FinSight master integration class.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from finsight.agents.base_agents import NewsSentimentAgent, TechnicalAnalysisAgent
from finsight.forecasting.price_forecasting import PriceForecastingEngine
from finsight.visualization.financial_visualizer import FinancialVisualizer

logger = logging.getLogger(__name__)

class FinSightMaster:
    """Master integration class for FinSight system."""

    def __init__(self, model="mistral", base_url="http://localhost:11434"):
        """
        Initialize FinSight master class.

        Args:
            model: LLM model to use
            base_url: Base URL for LLM API
        """
        logger.info("Initializing FinSight Master system")

        # Initialize agents
        self.sentiment_agent = NewsSentimentAgent(
            name="SentimentAgent",
            model=model,
            base_url=base_url
        )

        self.technical_agent = TechnicalAnalysisAgent(
            name="TechnicalAgent",
            model=model,
            base_url=base_url
        )

        # Initialize supporting components
        self.forecasting_engine = PriceForecastingEngine()
        self.visualizer = FinancialVisualizer()

    def generate_comprehensive_analysis(
            self,
            query: str,
            ticker: str,
            news_articles: List[Dict] = None,
            market_data: Dict = None,
            forecast_days: int = 7
    ) -> Dict:
        """
        Generate comprehensive financial analysis.

        Args:
            query: User query
            ticker: Ticker symbol
            news_articles: List of news articles
            market_data: Market data for tickers
            forecast_days: Number of days to forecast

        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Generating comprehensive analysis for {ticker}: '{query}'")

        result = {
            "query": query,
            "ticker": ticker,
            "analysis_time": datetime.now().isoformat(),
            "components": {}
        }

        # 1. Run sentiment analysis if news articles are provided
        if news_articles:
            logger.info(f"Running sentiment analysis on {len(news_articles)} news articles")
            sentiment_result = self.sentiment_agent.analyze({
                "query": query,
                "articles": news_articles
            })

            result["components"]["sentiment_analysis"] = sentiment_result

            # Generate sentiment visualization
            if 'key_factors' in sentiment_result:
                pos_count = len(sentiment_result['key_factors'].get('positive', []))
                neg_count = len(sentiment_result['key_factors'].get('negative', []))
                neutral_count = len(news_articles) - pos_count - neg_count

                sentiment_data = {
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "neutral_count": neutral_count,
                    "key_factors": sentiment_result.get('key_factors', {})
                }

                sentiment_chart = self.visualizer.create_sentiment_chart(sentiment_data)
                result["components"]["sentiment_chart"] = sentiment_chart

        # 2. Run technical analysis if market data is provided
        if market_data and ticker in market_data:
            price_data = market_data[ticker]

            if not isinstance(price_data, pd.DataFrame) or price_data.empty:
                logger.warning(f"Invalid or empty price data for {ticker}")
            else:
                logger.info(f"Running technical analysis for {ticker}")

                technical_result = self.technical_agent.analyze({
                    "query": query,
                    "market_data": {ticker: price_data}
                })

                result["components"]["technical_analysis"] = technical_result

                # Create technical dashboard
                technical_chart = self.visualizer.create_technical_dashboard(
                    price_data,
                    ticker,
                    technical_result
                )
                result["components"]["technical_chart"] = technical_chart

                # 3. Generate price forecast
                try:
                    logger.info(f"Generating price forecast for {ticker}, {forecast_days} days")
                    forecast = self.forecasting_engine.forecast_prices(
                        ticker,
                        price_data,
                        forecast_days
                    )

                    result["components"]["price_forecast"] = forecast

                    # Create price chart with forecast
                    price_chart = self.visualizer.create_price_chart(
                        price_data,
                        ticker,
                        forecast
                    )
                    result["components"]["price_chart"] = price_chart

                except Exception as e:
                    logger.error(f"Error generating forecast: {str(e)}")
                    # Run diagnostics when forecasting fails
                    try:
                        logger.info("Running forecast diagnostics...")
                        diagnostics = self.forecasting_engine.diagnose_forecasting_process(ticker, price_data)

                        # Log diagnostic results
                        if diagnostics and "errors" in diagnostics and diagnostics["errors"]:
                            logger.error(f"Forecast diagnostic errors: {diagnostics['errors']}")

                        # Store diagnostics in result for future reference
                        result["components"]["forecast_diagnostics"] = {
                            "errors": diagnostics.get("errors", []),
                            "data_stats": diagnostics.get("data_stats", {}),
                        }
                    except Exception as diag_error:
                        logger.error(f"Error running diagnostics: {str(diag_error)}")

        # 4. Generate summary text
        result["summary"] = self._generate_summary(result)


        # Save visualizations to disk
        try:
            import base64
            import os

            # Create a directory for visualizations if it doesn't exist
            viz_dir = "visualizations"
            os.makedirs(viz_dir, exist_ok=True)

            # Helper function to save base64 image
            def save_base64_image(b64_string, filename):
                try:
                    image_data = base64.b64decode(b64_string)
                    file_path = os.path.join(viz_dir, filename)
                    with open(file_path, "wb") as f:
                        f.write(image_data)
                    logger.info(f"Saved visualization to: {file_path}")
                    return file_path
                except Exception as e:
                    logger.error(f"Error saving visualization: {str(e)}")
                    return None

            # Save each visualization
            if "sentiment_chart" in result["components"]:
                sentiment_path = save_base64_image(
                    result["components"]["sentiment_chart"],
                    f"{ticker}_sentiment_chart.png"
                )

            if "technical_chart" in result["components"]:
                technical_path = save_base64_image(
                    result["components"]["technical_chart"],
                    f"{ticker}_technical_chart.png"
                )

            if "price_chart" in result["components"]:
                price_path = save_base64_image(
                    result["components"]["price_chart"],
                    f"{ticker}_price_chart.png"
                )

        except Exception as viz_error:
            logger.error(f"Error saving visualizations: {str(viz_error)}")
        return result

    def _generate_summary(self, analysis_result: Dict) -> str:
        """Generate a human-readable summary of analysis results."""
        components = analysis_result.get("components", {})
        ticker = analysis_result.get("ticker", "Unknown")

        summary = f"FINANCIAL ANALYSIS SUMMARY FOR {ticker}\n\n"

        # Add sentiment summary
        if "sentiment_analysis" in components:
            sentiment = components["sentiment_analysis"]
            overall = sentiment.get("overall_sentiment", "neutral")
            confidence = sentiment.get("confidence", 0.0)

            summary += f"SENTIMENT: {overall.upper()} (Confidence: {confidence:.2f})\n"

            if "key_factors" in sentiment:
                pos_factors = sentiment["key_factors"].get("positive", [])
                neg_factors = sentiment["key_factors"].get("negative", [])

                if pos_factors:
                    summary += "Positive factors: " + ", ".join(pos_factors[:3]) + "\n"
                if neg_factors:
                    summary += "Negative factors: " + ", ".join(neg_factors[:3]) + "\n"

        # Add technical analysis
        if "technical_analysis" in components:
            tech = components["technical_analysis"]
            trend = tech.get("trend_direction", "sideways")
            recommendation = tech.get("recommendation", "hold").upper()
            timeframe = tech.get("recommendation_timeframe", "medium")

            summary += f"\nTECHNICAL ANALYSIS:\n"
            summary += f"Trend: {trend.upper()}\n"
            summary += f"Recommendation: {recommendation} ({timeframe} term)\n"

            if "support_levels" in tech and tech["support_levels"]:
                summary += "Support levels: " + ", ".join([str(level) for level in tech["support_levels"][:2]]) + "\n"

            if "resistance_levels" in tech and tech["resistance_levels"]:
                summary += "Resistance levels: " + ", ".join([str(level) for level in tech["resistance_levels"][:2]]) + "\n"

        # Add forecast summary
        if "price_forecast" in components:
            forecast = components["price_forecast"]
            last_price = forecast.get("last_known_price", 0)
            predicted_prices = forecast.get("forecast", [])

            if predicted_prices:
                last_predicted = predicted_prices[-1]
                change_pct = ((last_predicted / last_price) - 1) * 100

                summary += f"\nPRICE FORECAST ({len(predicted_prices)} days):\n"
                summary += f"Current price: ${last_price:.2f}\n"
                summary += f"Forecasted price: ${last_predicted:.2f} ({change_pct:+.2f}%)\n"

        return summary