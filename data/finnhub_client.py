import os
import finnhub
import pandas as pd
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

class FinnhubClient:
    """Adapter for Finnhub API to fetch quotes, news, and sentiment."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            print("Warning: FINNHUB_API_KEY not found in .env file. Real-time data and news will be unavailable.")
            self.client = None
        else:
            self.client = finnhub.Client(api_key=self.api_key)

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get real-time quote data.
        Returns dict with keys: c (current), d (change), dp (percent change), h (high), l (low), o (open), pc (prev close)
        """
        if not self.client:
            return {}
        try:
            return self.client.quote(ticker)
        except Exception as e:
            print(f"Error fetching quote for {ticker}: {e}")
            return {}

    def get_company_news(self, ticker: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get company news for the last N days."""
        if not self.client:
            return []
            
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        try:
            return self.client.company_news(ticker, _from=start_date, to=end_date)
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []

    def get_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get news sentiment (if available in free tier, else mock or use news headlines)."""
        if not self.client:
            return {}
        
        try:
            # Finnhub 'news-sentiment' endpoint might be premium. 
            # We can check basic sentiment or just return raw news for the AI refiner to process.
            # For now, let's try to fetch it, handle error if premium-only.
            return self.client.news_sentiment(ticker)
        except Exception as e:
            # print(f"Sentiment endpoint might be restricted: {e}")
            return {}

if __name__ == "__main__":
    # Test
    client = FinnhubClient()
    if client.client:
        print("Quote:", client.get_quote("AAPL"))
        print("News:", len(client.get_company_news("AAPL")))
