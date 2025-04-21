import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ChartCollector:
    """Collects and processes market chart data using Yahoo Finance."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get stock price data for a given symbol."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_index_data(self, index: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get index data (e.g., S&P 500, NASDAQ)."""
        return self.get_stock_data(index, start_date, end_date, interval)
    
    def save_chart_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save chart data to CSV file."""
        if not df.empty:
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            logger.info(f"Saved chart data to {filepath}")
        else:
            logger.warning("No data to save")

def main():
    """Collect chart data for stocks and indices."""
    load_dotenv()
    collector = ChartCollector()
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=31)).strftime("%Y-%m-%d")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    for symbol in symbols:
        df = collector.get_stock_data(symbol, start_date, end_date)
        collector.save_chart_data(df, f"{symbol}_chart.csv")
    
    indices = ["^GSPC", "^IXIC", "^DJI"]
    for index in indices:
        df = collector.get_index_data(index, start_date, end_date)
        collector.save_chart_data(df, f"{index}_chart.csv")

if __name__ == "__main__":
    main() 