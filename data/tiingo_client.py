"""
Tiingo API Client
Free tier: 1000 requests/day, 50 unique symbols/day
Covers: News Sentiment, EOD Prices
"""
import os
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


class TiingoClient:
    """Adapter for Tiingo API — News sentiment and EOD prices."""

    BASE_URL = "https://api.tiingo.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            print("Warning: TIINGO_API_KEY not found. Tiingo data will be unavailable.")

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}"
        }

    def _get(self, endpoint: str, params: dict = None) -> Any:
        """Generic GET request."""
        if not self.api_key:
            return None
        try:
            resp = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                headers=self._headers(),
                params=params or {},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Tiingo API error ({endpoint}): {e}")
            return None

    # --- News with Sentiment ---
    def get_news_sentiment(self, tickers: str = None, limit: int = 20,
                           source: str = None) -> List[Dict[str, Any]]:
        """
        Fetch news articles with sentiment scores.
        tickers: comma-separated ticker list (e.g., 'AAPL,MSFT')
        """
        params = {"limit": limit}
        if tickers:
            params["tickers"] = tickers
        if source:
            params["source"] = source

        data = self._get("tiingo/news", params)
        if not data:
            return []

        results = []
        for article in data:
            results.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "publishedDate": article.get("publishedDate", ""),
                "tickers": article.get("tickers", []),
                "tags": article.get("tags", []),
                "sentiment": self._extract_sentiment(article)
            })
        return results

    @staticmethod
    def _extract_sentiment(article: dict) -> str:
        """Extract sentiment from article tags or title keywords."""
        tags = [t.lower() for t in article.get("tags", [])]
        title = article.get("title", "").lower()

        bullish_words = ["surge", "rally", "beat", "upgrade", "growth", "record", "soar"]
        bearish_words = ["drop", "fall", "miss", "downgrade", "decline", "crash", "plunge"]

        bull = sum(1 for w in bullish_words if w in title or w in " ".join(tags))
        bear = sum(1 for w in bearish_words if w in title or w in " ".join(tags))

        if bull > bear:
            return "bullish"
        elif bear > bull:
            return "bearish"
        return "neutral"

    # --- EOD Prices ---
    def get_daily_prices(self, ticker: str, start_date: str = None,
                         end_date: str = None) -> pd.DataFrame:
        """
        Fetch daily EOD prices.
        Dates in 'YYYY-MM-DD' format.
        """
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = self._get(f"tiingo/daily/{ticker}/prices", params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={
            "date": "Date", "adjOpen": "Open", "adjHigh": "High",
            "adjLow": "Low", "adjClose": "Close", "adjVolume": "Volume"
        })
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols] if cols else df

    # --- Ticker Metadata ---
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch metadata for a ticker."""
        data = self._get(f"tiingo/daily/{ticker}")
        if not data:
            return {}
        return data


if __name__ == "__main__":
    client = TiingoClient()
    if client.api_key:
        news = client.get_news_sentiment("AAPL", limit=5)
        for n in news:
            print(f"[{n['sentiment']}] {n['title'][:80]}")
        print(f"\n{len(news)} articles fetched")
