#!/usr/bin/env python
"""
Example script showing how to use FinSight's Agentic RAG capabilities.
"""

import logging
import json
from datetime import datetime
from finsight import PipelineManager
from finsight import AgenticRAGController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_results(result, filename):
    """Save results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {filename}")


def run_agentic_rag_example():
    """Example of using FinSight's Agentic RAG system."""
    logger.info("Starting FinSight Agentic RAG example")

    # 1. Initialize pipeline manager
    pipeline_manager = PipelineManager()

    # 2. Initialize Agentic RAG controller
    agentic_rag = AgenticRAGController(
        pipeline_manager=pipeline_manager,
        model="llama3.2",  # Using default model
        base_url="http://localhost:11434"  # Ollama API URL
    )

    # 3. Define the query
    query = "What is the current sentiment around Bitcoin and how might it affect price in the next week?"

    # 4. Process the query with Agentic RAG
    logger.info(f"Processing query: {query}")
    start_time = datetime.now()

    result = agentic_rag.process_query(
        query=query,
        context="",  # Optional context
        return_debug_info=True  # Include debugging info
    )

    # 5. Format the result for display
    formatted_result = agentic_rag.format_response_for_display(result)

    # 6. Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Query processed in {execution_time:.2f} seconds")

    # 7. Display the result
    print("\n" + "=" * 80)
    print("FINANCIAL ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nQuery: {query}")
    print(f"\nSummary:\n{formatted_result['summary']}")

    # 8. Save the results
    save_results(formatted_result, "agentic_rag_result.json")

    # 9. Return the result
    return formatted_result


def extract_and_display_insights(result):
    """Extract and display insights from the result."""
    insights = result.get('insights', [])

    print("\nKey Insights:")
    for insight in insights:
        if isinstance(insight, dict):
            insight_type = insight.get('type', '')

            if insight_type == 'sentiment':
                print(f"- Overall sentiment: {insight.get('value', 'unknown').upper()}")
            elif insight_type == 'positive_factor':
                print(f"- Positive factor: {insight.get('value', '')}")
            elif insight_type == 'negative_factor':
                print(f"- Negative factor: {insight.get('value', '')}")
            elif insight_type == 'trend':
                print(f"- Market trend: {insight.get('value', 'unknown').upper()}")
            elif insight_type == 'recommendation':
                print(f"- Trading recommendation: {insight.get('value', 'hold').upper()} "
                      f"({insight.get('timeframe', 'medium')} term)")
            elif insight_type == 'price_forecast':
                forecast = insight.get('value', {})
                if isinstance(forecast, dict):
                    current = forecast.get('current_price', 0)
                    forecasted = forecast.get('forecasted_price', 0)
                    change = forecast.get('change_percentage', 0)
                    print(f"- Price forecast: ${current:.2f} → ${forecasted:.2f} ({change:+.2f}%)")
            elif insight_type == 'generated_insight':
                print(f"- Analysis: {insight.get('value', '')}")
            else:
                print(f"- {insight}")


def run_multiple_queries():
    """Run multiple example queries."""
    # Initialize components
    pipeline_manager = PipelineManager()
    agentic_rag = AgenticRAGController(pipeline_manager=pipeline_manager)

    # Define queries
    queries = [
        "What's the current sentiment around Tesla and how will it affect stock price?",
        "Analyze the recent performance of Bitcoin and provide a price forecast",
        "What are the key factors affecting the tech sector this week?",
        "How is inflation affecting the market sentiment for gold as a safe haven?",
        "What technical indicators suggest about Apple's stock movement in the near term?"
    ]

    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\nProcessing query {i}/{len(queries)}: {query}")

        try:
            # Process with minimal debug output
            result = agentic_rag.process_query(query, return_debug_info=False)
            formatted = agentic_rag.format_response_for_display(result)

            # Display summary
            print(f"\nSummary: {formatted['summary'][:200]}...")

            # Save result
            save_results(formatted, f"query_{i}_result.json")

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")

    print("\nAll queries processed. Results saved to JSON files.")


if __name__ == "__main__":
    try:
        # Run the Agentic RAG example
        result = run_agentic_rag_example()

        # Display more detailed insights
        extract_and_display_insights(result)

        # Uncomment to run multiple example queries
        # run_multiple_queries()

    except Exception as e:
        logger.error(f"Error running example: {str(e)}", exc_info=True)