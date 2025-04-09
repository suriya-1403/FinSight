"""
Main module for running the FinSight AI system with Agentic RAG capabilities.
"""

import argparse
import logging
import sys
from datetime import datetime

from finsight.pipeline import PipelineManager
from finsight.config import DEFAULT_NEWS_QUERIES, DEFAULT_TICKERS, DEFAULT_DAYS_BACK
from finsight.master import FinSightMaster

# Import Agentic RAG components when available
try:
    from finsight.agent.controller import AgenticRAGController

    agentic_rag_available = True
except ImportError:
    agentic_rag_available = False

# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FinSight AI - Financial insights system")

    parser.add_argument(
        "--use_agents",
        action="store_true",
        help="Use multi-agent system for analysis",
    )

    parser.add_argument(
        "--use_agentic_rag",
        action="store_true",
        help="Use the new Agentic RAG system for autonomous analysis",
    )

    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Generate comprehensive analysis with visualizations and forecasts",
    )

    parser.add_argument(
        "--mode",
        choices=["collect", "analyze", "full"],
        default="full",
        help="Pipeline mode to run",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Query for financial insights",
    )

    parser.add_argument(
        "--analyze_with_llm",
        action="store_true",
        help="Use LLM to analyze and generate insights",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help=f"Number of days to look back (default: {DEFAULT_DAYS_BACK})",
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to collect data for",
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_NEWS_QUERIES,
        help="News query terms",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Add arguments for Agentic RAG
    if agentic_rag_available:
        parser.add_argument(
            "--model",
            type=str,
            default="llama3.2",
            help="LLM model to use for Agentic RAG"
        )

        parser.add_argument(
            "--base_url",
            type=str,
            default="http://localhost:11434",
            help="LLM API base URL"
        )

        parser.add_argument(
            "--max_iterations",
            type=int,
            default=10,
            help="Maximum iterations for Agentic RAG"
        )

        parser.add_argument(
            "--debug",
            action="store_true",
            help="Include debug info in Agentic RAG output"
        )

    return parser.parse_args()


def extract_ticker_from_query(query):
    """Extract potential ticker symbol from query."""
    import re

    # Look for ticker patterns
    ticker_match = re.search(r'\b[A-Z]{1,5}-?USD?\b', query)
    if ticker_match:
        return ticker_match.group(0)

    # Look for common crypto keywords
    crypto_map = {
        'bitcoin': 'BTC-USD',
        'btc': 'BTC-USD',
        'ethereum': 'ETH-USD',
        'eth': 'ETH-USD',
        'ripple': 'XRP-USD',
        'xrp': 'XRP-USD',
        'dogecoin': 'DOGE-USD',
        'doge': 'DOGE-USD',
        'solana': 'SOL-USD',
        'sol': 'SOL-USD',
        'cardano': 'ADA-USD',
        'ada': 'ADA-USD',
        'binance coin': 'BNB-USD',
        'bnb': 'BNB-USD',
        'tether': 'USDT-USD',
        'usdt': 'USDT-USD',
        'usd coin': 'USDC-USD',
        'usdc': 'USDC-USD',
        'litecoin': 'LTC-USD',
        'ltc': 'LTC-USD',
        'polkadot': 'DOT-USD',
        'dot': 'DOT-USD',
        'chainlink': 'LINK-USD',
        'link': 'LINK-USD',
        'stellar': 'XLM-USD',
        'xlm': 'XLM-USD',
        'monero': 'XMR-USD',
        'xmr': 'XMR-USD',
        'eos': 'EOS-USD',
        'tron': 'TRX-USD',
        'trx': 'TRX-USD',
        'iota': 'MIOTA-USD',
        'neo': 'NEO-USD',
        'dash': 'DASH-USD',
        'zcash': 'ZEC-USD',
        'cosmos': 'ATOM-USD',
        'atom': 'ATOM-USD',
        'tezos': 'XTZ-USD',
        'xtz': 'XTZ-USD',
        'algorand': 'ALGO-USD',
        'algo': 'ALGO-USD',
        'terra': 'LUNA-USD',
        'luna': 'LUNA-USD',
        'avalanche': 'AVAX-USD',
        'avax': 'AVAX-USD',
        'polygon': 'MATIC-USD',
        'matic': 'MATIC-USD',
        'uniswap': 'UNI-USD',
        'uni': 'UNI-USD',
        'aave': 'AAVE-USD',
        'filecoin': 'FIL-USD',
        'fil': 'FIL-USD',
    }

    query_lower = query.lower()
    for keyword, ticker in crypto_map.items():
        if keyword in query_lower:
            return ticker

    return None


def run_agentic_rag(pipeline_manager, query, args):
    """
    Run analysis using the new Agentic RAG system.

    Args:
        pipeline_manager: PipelineManager instance
        query: User query string
        args: Command line arguments

    Returns:
        Dictionary with analysis results
    """
    if not agentic_rag_available:
        logger.error("Agentic RAG components not available")
        return {"error": "Agentic RAG not available - make sure agent module is installed"}

    logger.info(f"Running Agentic RAG analysis for query: {query}")

    # Initialize the Agentic RAG controller
    agentic_rag = AgenticRAGController(
        pipeline_manager=pipeline_manager,
        model=args.model,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        max_execution_time=300  # 5 minutes default
    )

    # Process the query
    result = agentic_rag.process_query(
        query=query,
        context="",  # Can add context here if available
        return_debug_info=args.debug
    )

    # Format for display
    formatted_result = agentic_rag.format_response_for_display(result)

    # Print to console
    print("\n" + "=" * 80)
    print("FINSIGHT AGENTIC RAG ANALYSIS")
    print("=" * 80)
    print(f"\nQuery: {query}")
    print(f"\nSummary:\n{formatted_result['summary']}")

    # Show insights if available
    if formatted_result.get('insights'):
        print("\nKey Insights:")
        for insight in formatted_result['insights']:
            if isinstance(insight, dict):
                insight_type = insight.get('type', '')
                if insight_type == 'sentiment':
                    print(f"- Sentiment: {insight.get('value', 'unknown').upper()}")
                elif insight_type == 'recommendation':
                    print(
                        f"- Recommendation: {insight.get('value', 'hold').upper()} ({insight.get('timeframe', 'medium')} term)")
                elif insight_type == 'generated_insight':
                    print(f"- {insight.get('value', '')}")
                else:
                    print(f"- {insight}")

    print("=" * 80 + "\n")

    return formatted_result


def main():
    """Run the FinSight AI system."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    start_time = datetime.now()
    logger.info(f"Starting FinSight AI at {start_time}")
    logger.info(f"Mode: {args.mode}")

    # Initialize pipeline
    pipeline = PipelineManager()

    # Run specified pipeline mode
    if args.mode == "collect":
        result = pipeline.run_data_collection_pipeline(
            news_queries=args.queries,
            tickers=args.tickers,
            days_back=args.days,
        )
        logger.info(f"Collected {len(result['news_data'])} news articles")
        logger.info(f"Collected market data for {len(result['market_data'])} tickers")

    elif args.mode == "analyze":
        logger.info("Running analysis-only mode...")
        analyzed_news = pipeline.run_analysis_pipeline(
            news_df=None,  # Will fetch from database
            store_results=True,
            days_back=args.days
        )

        if not analyzed_news.empty:
            logger.info(f"Analyzed {len(analyzed_news)} news articles")
        else:
            logger.warning("No news data was found or processed")

    elif args.mode == "full":
        result = pipeline.run_full_pipeline(
            news_queries=args.queries,
            tickers=args.tickers,
            days_back=args.days,
        )
        logger.info(f"Processed {len(result['news_data'])} news articles")
        logger.info(f"Processed market data for {len(result['market_data'])} tickers")

    if args.query:
        logger.info(f"Querying for insights: {args.query}")

        # Check which analysis method to use
        if args.use_agentic_rag and agentic_rag_available:
            # Use the new Agentic RAG system
            result = run_agentic_rag(pipeline, args.query, args)

        elif args.comprehensive:
            # Use the master integration for comprehensive analysis
            master = FinSightMaster(model="mistral")

            # Get relevant ticker from query or use default
            ticker = extract_ticker_from_query(args.query) or "BTC-USD"

            # Get news and market data
            news_df = pipeline.query_financial_insights(
                query=args.query,
                n_results=10
            )

            # Convert news DataFrame to list of dictionaries
            news_list = []
            for _, row in news_df.iterrows():
                news_dict = {
                    "document": row.get("content", ""),
                    "metadata": {
                        "source": row.get("source", {}).get("name", "Unknown") if isinstance(row.get("source"),
                                                                                             dict) else "Unknown",
                        "publishedAt": row.get("publishedAt", ""),
                        "url": row.get("url", ""),
                        "sentiment_label": row.get("sentiment_label", "neutral"),
                        "sentiment_compound": row.get("sentiment_compound", 0.0)
                    }
                }
                news_list.append(news_dict)

            # Get market data for the ticker
            market_data = {}
            try:
                # Use the market data collector from pipeline_manager
                market_data = pipeline.market_collector.collect_market_data(
                    tickers=[ticker],
                    period=f"{args.days}d"
                )
                logger.info(f"Retrieved market data for ticker: {ticker}")
            except Exception as e:
                logger.error(f"Error retrieving market data: {str(e)}")

            # Generate comprehensive analysis
            result = master.generate_comprehensive_analysis(
                query=args.query,
                ticker=ticker,
                news_articles=news_list,
                market_data=market_data,
                forecast_days=7
            )

            # Print summary
            print("\nComprehensive Financial Analysis:")
            print(result["summary"])

            # Print location of saved visualizations
            print("\nVisualizations saved to:")
            for name, _ in result["components"].items():
                if name.endswith("_chart"):
                    print(f" - {name}")

        elif args.use_agents:
            # Use multi-agent system
            insights = pipeline.generate_multi_agent_insights(
                query=args.query,
                n_results=10,
                days_back=args.days
            )

            # Print insights
            print("\nFinancial Insights (Multi-Agent Analysis):")
            print(f"Query: {args.query}")
            print("\n" + insights.get("summary", "No summary available"))

            if "error" in insights and insights["error"]:
                print(f"\nNote: {insights['error']}")

        elif args.analyze_with_llm:
            # Generate insights with LLM
            insights = pipeline.generate_financial_insights(
                query=args.query,
                n_results=5
            )

            # Print insights
            print("\nFinancial Insights:")
            print(f"Query: {insights['query']}")
            print("\n" + insights['insight'])

            print("\nSources:")
            for i, source in enumerate(insights['sources'], 1):
                print(f"{i}. {source['source']} - {source['url']}")
        else:
            # Standard query without LLM
            results = pipeline.query_financial_insights(
                query=args.query,
                n_results=5
            )

            if len(results) > 0:
                print("\nTop insights:")
                for i, (_, row) in enumerate(results.iterrows(), 1):
                    print(f"{i}. {row['metadata']['source']} ({row['metadata']['publishedAt']})")
                    print(
                        f"   Sentiment: {row['metadata']['sentiment_label']} ({row['metadata']['sentiment_compound']:.2f})")
                    print(f"   Relevance: {1 - row['distance']:.2f}")
                    print(f"   URL: {row['metadata']['url']}")
                    print(f"   Excerpt: {row['document'][:150]}...")
                    print()
            else:
                print("\nNo relevant insights found.")
                print(
                    "Try rephrasing your query or running without the --query parameter first to build the vector database.")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"FinSight AI completed in {duration} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running FinSight AI: {str(e)}", exc_info=True)
        sys.exit(1)