"""
Module for generating financial insights using Ollama with LangChain.
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Setup logging
logger = logging.getLogger(__name__)

class InsightGenerator:
    """Class for generating financial insights using Ollama with LangChain."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize the insight generator.

        Args:
            model: Ollama model name to use.
            base_url: Ollama API base URL.
        """
        self.model_name = model
        self.base_url = base_url

        # Initialize Ollama LLM
        try:
            self.llm = OllamaLLM(model=model, base_url=base_url)
            logger.info(f"Initialized Ollama with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {str(e)}")
            self.llm = None

        # Create prompt template
        # Enhanced prompt template for the InsightGenerator class
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a sophisticated financial analyst with expertise in markets, stocks, and cryptocurrencies. Your task is to provide nuanced, actionable financial insights based on news data.

        USER QUERY: {query}

        RELEVANT NEWS ARTICLES:
        {context}

        Based on these articles, provide a comprehensive financial insight addressing the user's query. Your analysis should be structured with the following sections:

        1. Executive Summary (1-2 sentences capturing the core insight)

        2. Market Analysis:
           - Key trends and patterns identified from the articles
           - Correlation between news events and market movements
           - Sentiment analysis interpretation (positive/negative/neutral)
           - Potential market signals or indicators

        3. Strategic Implications:
           - Short-term outlook (1-7 days)
           - Medium-term projections (1-4 weeks)
           - Potential risks and opportunities
           - Market segments or assets most likely to be affected

        4. Actionable Recommendations:
           - Specific, actionable steps for investors to consider
           - Risk management strategies
           - Alternative perspectives to consider
           - Information gaps that require additional research

        Keep your analysis FACTUAL and based ONLY on the provided articles. Use a professional, balanced tone. Acknowledge uncertainty where appropriate and avoid overly speculative claims. Include relevant numerical data when available.

        Your audience consists of experienced investors who require sophisticated analysis rather than basic explanations.
        """
        )

        # Create LLM chain
        if self.llm:
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        else:
            self.chain = None

    def generate_insight(self, query: str, relevant_articles: pd.DataFrame) -> Dict:
        """
        Generate financial insights based on retrieved articles.

        Args:
            query: User query.
            relevant_articles: DataFrame of relevant articles.

        Returns:
            Dictionary with generated insights.
        """
        try:
            # Prepare context from relevant articles
            context = self._prepare_context(relevant_articles)

            # Generate insight using LangChain
            if self.chain:
                response = self.chain.invoke({"query": query, "context": context})
                # Extract text from response
                if isinstance(response, dict) and "text" in response:
                    insight_text = response["text"]
                else:
                    insight_text = str(response)
            else:
                # Fallback if Ollama is not available
                insight_text = self._generate_fallback_insight(query, relevant_articles)

            return {
                "query": query,
                "insight": insight_text,
                "sources": relevant_articles["metadata"].apply(
                    lambda x: {"source": x.get("source", ""), "url": x.get("url", "")}
                ).tolist()
            }

        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}")
            return {
                "query": query,
                "insight": f"Unable to generate insight due to an error: {str(e)}",
                "sources": []
            }

    def _prepare_context(self, articles: pd.DataFrame) -> str:
        """
        Prepare context from relevant articles.

        Args:
            articles: DataFrame of relevant articles.

        Returns:
            Context string.
        """
        context_parts = []

        for _, row in articles.iterrows():
            source = row["metadata"].get("source", "Unknown")
            url = row["metadata"].get("url", "")
            published = row["metadata"].get("publishedAt", "")
            sentiment = row["metadata"].get("sentiment_label", "neutral")
            text = row["document"][:500]  # Limit text length

            article_context = f"SOURCE: {source} ({published})\n"
            article_context += f"SENTIMENT: {sentiment}\n"
            article_context += f"TEXT: {text}\n"
            article_context += f"URL: {url}\n\n"

            context_parts.append(article_context)

        return "\n".join(context_parts)

    def _generate_fallback_insight(self, query: str, articles: pd.DataFrame) -> str:
        """
        Generate a fallback insight when LLM is not available.

        Args:
            query: User query.
            articles: DataFrame of relevant articles.

        Returns:
            Generated insight.
        """
        # Create a simple insight based on articles
        if len(articles) == 0:
            return "No relevant articles found to answer your query."

        summary = "Based on the retrieved articles:"

        # Aggregate sentiments
        sentiments = []
        for _, row in articles.iterrows():
            sentiment = row["metadata"].get("sentiment_label", "neutral")
            sentiments.append(sentiment)

        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        neutral = sentiments.count("neutral")

        overall_sentiment = "neutral"
        if positive > negative and positive > neutral:
            overall_sentiment = "positive"
        elif negative > positive and negative > neutral:
            overall_sentiment = "negative"

        summary += f"\n\n1. Summary:\nOverall sentiment is {overall_sentiment} with {positive} positive, {negative} negative, and {neutral} neutral articles."

        # List sources
        summary += "\n\n2. Analysis:\nInformation retrieved from sources including: "
        sources = [row["metadata"].get("source", "Unknown") for _, row in articles.iterrows()]
        unique_sources = list(set(sources))
        summary += ", ".join(unique_sources[:5])

        summary += "\n\n3. Implications:\nFor a more detailed analysis, please consider reviewing the linked articles directly."

        summary += "\n\n4. Recommendations:\nConsider consulting with a financial advisor for personalized advice based on this information."

        return summary