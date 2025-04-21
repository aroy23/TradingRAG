import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class RAGQuery:
    """RAG query system for market analysis."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the RAG query system.
        
        Args:
            data_dir: Directory containing chart data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.client = OpenAI()
        
    def encode_query(self, query: str) -> np.ndarray:
        """Convert user query into embedding vector using text-embedding-3-large."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=query
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            return np.array([])
    
    def _calculate_similarity_score(self, query_embedding: np.ndarray, chart_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between query and chart embeddings."""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        chart_norm = chart_embedding / np.linalg.norm(chart_embedding)
        similarity = np.dot(query_norm, chart_norm)
        return (similarity + 1) / 2
    
    def _describe_chart_pattern(self, df: pd.DataFrame) -> str:
        """Convert chart technical indicators into natural language description."""
        # Calculate basic statistics
        total_return = (df['Close'].iloc[-1] / df['Open'].iloc[0] - 1) * 100
        volatility = df['Close'].pct_change().std() * 100
        
        # Calculate trend using multiple methods
        # 1. Overall linear trend
        x = np.arange(len(df))
        y = df['Close'].values
        slope = np.polyfit(x, y, 1)[0]
        overall_trend = "upward" if slope > 0 else "downward"
        
        # 2. Recent trend (last 5 days)
        recent_prices = df['Close'].iloc[-5:]
        recent_slope = np.polyfit(np.arange(len(recent_prices)), recent_prices, 1)[0]
        recent_trend = "upward" if recent_slope > 0 else "downward"
        
        # 3. Moving average crossover
        ma_5 = df['Close'].rolling(window=5).mean()
        ma_10 = df['Close'].rolling(window=10).mean()
        ma_trend = "upward" if ma_5.iloc[-1] > ma_10.iloc[-1] else "downward"
        
        # Determine final trend based on majority
        trend_signals = [overall_trend, recent_trend, ma_trend]
        trend = "upward" if trend_signals.count("upward") >= 2 else "downward"
        
        # Calculate momentum
        momentum_5 = df['Close'].pct_change(5).iloc[-1] * 100
        momentum_10 = df['Close'].pct_change(10).iloc[-1] * 100
        
        # Calculate volume pattern
        recent_volume = df['Volume'].iloc[-5:].mean()
        avg_volume = df['Volume'].mean()
        volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]
        
        # Calculate price range
        price_range = ((df['High'].max() - df['Low'].min()) / df['Close'].mean()) * 100
        
        # Create natural language description
        description = (
            f"This chart shows a {trend} trend with {total_return:.1f}% total return. "
            f"Recent price action is {recent_trend}. "
            f"Price range is {price_range:.1f}% of the average price. "
            f"Volatility is {volatility:.1f}%. "
            f"Recent momentum is {momentum_5:.1f}% over 5 days and {momentum_10:.1f}% over 10 days. "
            f"Volume is {volume_trend} compared to the average. "
            f"RSI is at {rsi_value:.1f}, indicating {'overbought' if rsi_value > 70 else 'oversold' if rsi_value < 30 else 'neutral'} conditions."
        )
        
        return description

    def retrieve_similar_charts(self, query_embedding: np.ndarray, top_k: int = 2) -> List[Dict]:
        """Search for similar chart patterns based on semantic matching."""
        if query_embedding.size == 0:
            return []
            
        results = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_chart.csv'):
                symbol = filename.replace('_chart.csv', '')
                filepath = os.path.join(self.data_dir, filename)
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Convert chart to natural language description
                chart_description = self._describe_chart_pattern(df)
                
                # Get embedding for chart description
                chart_embedding = self.encode_query(chart_description)
                
                # Calculate similarity score
                similarity_score = self._calculate_similarity_score(query_embedding, chart_embedding)
                
                # Calculate return using the same method as in description
                total_return = (df['Close'].iloc[-1] / df['Open'].iloc[0] - 1) * 100
                
                stats = {
                    "symbol": symbol,
                    "start_date": df.index[0].strftime("%Y-%m-%d"),
                    "end_date": df.index[-1].strftime("%Y-%m-%d"),
                    "high": df["High"].max(),
                    "low": df["Low"].min(),
                    "open": df["Open"].iloc[0],
                    "close": df["Close"].iloc[-1],
                    "volume": df["Volume"].mean(),
                    "return": total_return,
                    "similarity_score": similarity_score,
                    "description": chart_description
                }
                
                results.append(stats)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
    
    def rerank_results(self, results: List[Dict], query: str, chart_similarity_weight: float = 0.6,
                      temporal_weight: float = 0.2, metadata_weight: float = 0.2) -> List[Dict]:
        """Re-rank results based on multiple factors."""
        if not results:
            return []
            
        for result in results:
            chart_score = result.get("similarity_score", 0)
            temporal_score = self._calculate_temporal_score(result)
            metadata_score = self._calculate_metadata_score(result, query)
            
            combined_score = (
                chart_score * chart_similarity_weight +
                temporal_score * temporal_weight +
                metadata_score * metadata_weight
            )
            result["combined_score"] = combined_score
        
        return sorted(results, key=lambda x: x["combined_score"], reverse=True)
    
    def generate_analysis(self, results: List[Dict], query: str) -> str:
        """Generate analysis of retrieved results using GPT-4."""
        try:
            context = self._prepare_generation_context(results)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing insights on market patterns."},
                    {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\nProvide a detailed analysis."}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return "Error generating analysis"
    
    def _get_chart_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for a symbol."""
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_temporal_score(self, result: Dict) -> float:
        """Calculate temporal relevance score for a result."""
        end_date = datetime.strptime(result["end_date"], "%Y-%m-%d")
        days_old = (datetime.now() - end_date).days
        return 1 / (1 + days_old)
    
    def _calculate_metadata_score(self, result: Dict, query: str) -> float:
        """Calculate metadata matching score for a result."""
        query_terms = set(query.lower().split())
        symbol_terms = set(result["symbol"].lower().split())
        return len(query_terms.intersection(symbol_terms)) / len(query_terms)
    
    def _prepare_generation_context(self, results: List[Dict]) -> str:
        """Prepare context for analysis generation."""
        context = []
        for result in results:
            context.append(f"Symbol: {result['symbol']}")
            context.append(f"Period: {result['start_date']} to {result['end_date']}")
            context.append(f"Return: {result['return']:.2f}%")
            context.append(f"Similarity Score: {result['similarity_score']:.2f}")
            context.append(f"Description: {result['description']}")
            context.append("---")
        return "\n".join(context)

def main():
    """Run the RAG query pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--generate", action="store_true", help="Generate analysis")
    parser.add_argument("--output", type=str, default="rag_results.json", help="Output file")
    args = parser.parse_args()
    
    load_dotenv()
    query = RAGQuery()
    
    query_embedding = query.encode_query(args.query)
    results = query.retrieve_similar_charts(query_embedding, args.top_k)
    reranked_results = query.rerank_results(results, args.query)
    
    if args.generate:
        analysis = query.generate_analysis(reranked_results, args.query)
        reranked_results.append({"analysis": analysis})
    
    with open(args.output, 'w') as f:
        json.dump(reranked_results, f, indent=2)

if __name__ == "__main__":
    main() 