# FinSight AI: Intelligent Financial Insights Platform

## Project Overview

FinSight AI is an advanced, end-to-end financial analysis platform that leverages cutting-edge artificial intelligence to provide comprehensive, actionable insights into financial markets, stocks, cryptocurrencies, and news.

### Key Innovation: Agentic RAG System

At the heart of FinSight AI is a revolutionary Agentic Retrieval-Augmented Generation (RAG) system that goes beyond traditional information retrieval by:
- Dynamically planning and executing multi-step analysis
- Autonomously selecting and combining tools
- Providing contextual, nuanced financial insights

## Technical Architecture

### Core Components

1. **Data Collection Layer**
   - News Collector: Retrieves financial news from multiple sources
   - Market Data Collector: Gathers historical and real-time market data
   - Supports multiple data sources and customizable queries

2. **Preprocessing and Analysis**
   - Text Preprocessor: Cleans and normalizes text data
   - Sentiment Analyzer: Advanced sentiment analysis using NLTK
   - Embeddings Generator: Semantic text representations

3. **Intelligence Agents**
   - News Sentiment Agent: Analyzes market sentiment
   - Technical Analysis Agent: Provides technical market insights
   - Price Forecasting Agent: Uses LSTM neural networks for price predictions

4. **Vector Database**
   - Uses ChromaDB for efficient semantic search
   - Hybrid search combining vector similarity and BM25 ranking
   - Supports advanced filtering and retrieval

5. **Visualization and Reporting**
   - Generates comprehensive charts and dashboards
   - Provides actionable insights and summaries

### Unique Features

#### 1. Multi-Agent Reasoning System
- Dynamically creates and executes analysis plans
- Adapts to different query contexts
- Combines insights from multiple specialized agents

#### 2. Advanced Forecasting
- LSTM-based price prediction
- Handles various financial instruments
- Provides confidence intervals and trend analysis

#### 3. Hybrid Information Retrieval
- Combines vector similarity with lexical matching
- Reranks results for maximum relevance
- Handles complex, nuanced financial queries

## Technical Stack

- **Languages**: Python
- **Machine Learning**: 
  - TensorFlow
  - SentenceTransformers
  - scikit-learn
- **NLP**: 
  - NLTK
  - SpaCy
- **Databases**: 
  - MongoDB
  - ChromaDB
- **LLM Integration**: Ollama with LangChain

## Potential Hackathon Impact

### Problem Solved
Traditional financial analysis tools are:
- Expensive
- Lack contextual understanding

FinSight AI democratizes sophisticated financial analysis through:
- Accessible AI-powered insights
- Real-time processing
- Comprehensive, nuanced reporting

### Scalability and Future Work
- Multi-language support
- Integration with trading platforms
- Enhanced machine learning models
- Real-time market monitoring

## Technical Challenges Overcome

1. Semantic Understanding of Financial Texts
2. Multi-Agent Coordination
3. Scalable Information Retrieval
4. Accurate Price Forecasting

## Sample Workflow

```
User Query: "Analyze Tesla's market sentiment"
↓
Data Collection
  ├─ Retrieve Recent News
  ├─ Fetch Market Data
↓
Preprocessing
  ├─ Text Cleaning
  ├─ Sentiment Analysis
↓
Agent Reasoning
  ├─ Plan Execution
  ├─ Multi-Agent Insights
↓
Visualization & Reporting
  ├─ Sentiment Chart
  ├─ Price Forecast
  ├─ Comprehensive Summary
```

## Getting Started

### Prerequisites
- Python 3.10.15
- MongoDB
- Ollama
- Required dependencies in `requirements.txt`

### Installation
```bash
git clone https://github.com/suriya-1403/FinSight.git
cd FinSight
pip install -r requirements.txt
```
### CLI Usage

You can also run the system from the command line:

```bash
python -m finsight.main --query "What are the risks for Tesla in Q2?" --use_agents
```
### CLI Options

| Flag                | Description                                                |
|---------------------|------------------------------------------------------------|
| `--query`           | Natural language financial question                        |
| `--use_agents`      | Use multi-agent system                                     |
| `--use_agentic_rag` | Use autonomous Agentic RAG                                 |
| `--comprehensive`   | Run full news + market + visualization pipeline            |
| `--mode`            | `collect`, `analyze`, or `full` (default: `full`)          |
| `--tickers`         | Ticker symbols to collect market data for                  |
| `--queries`         | News search query terms                                    |
| `--days`            | Number of days to look back (default: from config)         |
| `--verbose`         | Enable debug logging                                       |

## Contribution and Future

FinSight AI is an open-source project welcoming contributions in:
- UI Enhancement
- Machine Learning Improvements
- New Agent Development
- Visualization Enhancements
- Platform Integrations

---
