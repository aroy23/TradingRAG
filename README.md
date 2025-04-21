# TradingRAG: Market Chart Data Collection and Analysis

## Overview

TradingRAG is a system that combines market chart data analysis with natural language processing to provide insights into stock patterns. It uses semantic matching to find relevant chart patterns based on natural language queries.

## System Architecture

The system consists of two main components:

1. **Chart Data Collection**
   - Collects 31 days of historical data for stocks and indices
   - Uses Yahoo Finance API for data retrieval
   - Stores data in CSV format for analysis
   - Currently tracks:
     - Stocks: AAPL, MSFT, GOOGL, AMZN, META
     - Indices: S&P 500 (^GSPC), NASDAQ (^IXIC), Dow Jones (^DJI)

2. **RAG Query Pipeline**
   - Query Encoding: Converts natural language queries to embeddings using text-embedding-3-large
   - Pattern Description: Converts chart data into natural language descriptions including:
     - Overall trend and recent price action
     - Total return and price range
     - Volatility and momentum indicators
     - Volume patterns and RSI conditions
   - Semantic Matching: Finds similar patterns using cosine similarity
   - Re-ranking: Combines multiple factors (similarity, temporal relevance, metadata)
   - Analysis Generation: Optional GPT-4o analysis of results

## Requirements

- Python 3.10+
- OpenAI API key
- Required packages in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aroy23/TradingRAG.git
cd TradingRAG
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure
```
DemoRAG/
├── tradingrag/
│   ├── data_collect/
│   │   └── collect_charts.py      # Script for collecting chart data
│   └── retrieval/
│       └── rag_query.py           # Script for RAG-based queries
├── rag_results.json               # Results after querying
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Usage

1. Collect Chart Data:
```bash
python3 -m tradingrag.data_collect.collect_charts
```

2. Run RAG Query:
```bash
python3 -m tradingrag.retrieval.rag_query --query "Show me stocks with downward momentum" --generate --top_k 2
```

3. Display Results:
```bash
# Terminal display
python3 -m tradingrag.retrieval.display_results

# Web interface
python3 -m tradingrag.retrieval.web_display
```
Then, navigate to the link in the terminal to view the results in a web interface.

Results are saved in JSON format to the specified output file (default: rag_results.json).