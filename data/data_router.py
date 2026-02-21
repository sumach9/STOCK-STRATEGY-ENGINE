"""
Unified Data Router
Provides a single interface for all data categories with automatic
primary → fallback provider switching.
"""
import pandas as pd
from typing import Dict, Any, List, Optional

from data.yfinance_client import YFinanceClient
from data.finnhub_client import FinnhubClient
from data.fmp_client import FMPClient
from data.alphavantage_client import AlphaVantageClient
from data.tiingo_client import TiingoClient
from data.yahooquery_client import YahooQueryClient


class DataRouter:
    """
    Unified entry point for all market data.
    Automatically falls back to secondary providers on failure.

    Provider Map:
        EOD/Historical  → yfinance (primary) → FMP (backup)
        Intraday        → AlphaVantage (primary) → yfinance (backup)
        Sentiment       → Finnhub (primary) → Tiingo (backup)
        Short Interest  → Yahooquery
        Fundamentals    → FMP (primary) → Yahooquery (backup)
        Sector Data     → AlphaVantage (primary) → FMP (backup)
    """

    def __init__(self):
        self.yfinance = YFinanceClient()
        self.finnhub = FinnhubClient()
        self.fmp = FMPClient()
        self.alphavantage = AlphaVantageClient()
        self.tiingo = TiingoClient()
        self.yahooquery = YahooQueryClient()

    # ─── EOD / HISTORICAL ────────────────────────────────────────────
    def get_historical_data(self, ticker: str, period: str = "1y",
                            interval: str = "1d") -> pd.DataFrame:
        """
        Fetch EOD OHLCV. Primary: yfinance, Backup: FMP.
        """
        # Primary: yfinance
        df = YFinanceClient.get_historical_data(ticker, period=period, interval=interval)
        if not df.empty:
            return df

        # Backup: FMP
        print(f"  [DataRouter] yfinance failed for {ticker}, trying FMP...")
        df = self.fmp.get_historical_data(ticker, period=period)
        if not df.empty:
            return df

        print(f"  [DataRouter] All EOD providers failed for {ticker}")
        return pd.DataFrame()

    # ─── INTRADAY ────────────────────────────────────────────────────
    def get_intraday(self, ticker: str, interval: str = "5min",
                     days_back: int = 1) -> pd.DataFrame:
        """
        Fetch intraday OHLCV.
        Primary: AlphaVantage, Backup: yfinance (5d/5m).
        """
        # Primary: AlphaVantage
        if self.alphavantage.api_key:
            df = self.alphavantage.get_intraday(ticker, interval=interval)
            if not df.empty:
                return df

        # Backup: yfinance
        print(f"  [DataRouter] Falling back to yfinance intraday for {ticker}")
        yf_interval = interval.replace("min", "m") if "min" in interval else interval
        df = YFinanceClient.get_historical_data(ticker, period="5d", interval=yf_interval)
        return df

    # ─── SENTIMENT ───────────────────────────────────────────────────
    def get_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch sentiment data.
        Primary: Finnhub, Backup: Tiingo.
        """
        # Primary: Finnhub
        if self.finnhub.client:
            data = self.finnhub.get_sentiment(ticker)
            if data:
                return {"provider": "finnhub", "data": data}

        # Backup: Tiingo news sentiment
        if self.tiingo.api_key:
            news = self.tiingo.get_news_sentiment(ticker, limit=10)
            if news:
                sentiments = [n["sentiment"] for n in news]
                bull = sentiments.count("bullish")
                bear = sentiments.count("bearish")
                total = len(sentiments)
                score = (bull - bear) / total if total > 0 else 0
                return {
                    "provider": "tiingo",
                    "data": {
                        "bullish": bull, "bearish": bear,
                        "neutral": total - bull - bear,
                        "score": round(score, 2),
                        "articles": len(news)
                    }
                }

        return {"provider": "none", "data": {}}

    # ─── NEWS ────────────────────────────────────────────────────────
    def get_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """
        Fetch company news.
        Primary: Finnhub, Backup: Tiingo.
        """
        # Primary: Finnhub
        if self.finnhub.client:
            news = self.finnhub.get_company_news(ticker)
            if news:
                return [{"provider": "finnhub", **n} for n in news[:limit]]

        # Backup: Tiingo
        if self.tiingo.api_key:
            news = self.tiingo.get_news_sentiment(ticker, limit=limit)
            if news:
                return [{"provider": "tiingo", **n} for n in news]

        return []

    # ─── SHORT INTEREST ──────────────────────────────────────────────
    def get_short_interest(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch short interest data.
        Primary: Finnhub, Backup: Yahooquery.
        """
        # Primary: Finnhub (if they expose short interest)
        # Finnhub free tier doesn't have short interest, go straight to yahooquery
        data = self.yahooquery.get_short_interest(ticker)
        if data:
            return {"provider": "yahooquery", "data": data}

        return {"provider": "none", "data": {}}

    # ─── FUNDAMENTALS ────────────────────────────────────────────────
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamentals.
        Primary: FMP, Backup: Yahooquery.
        """
        # Primary: FMP
        if self.fmp.api_key:
            data = self.fmp.get_fundamentals(ticker)
            if data:
                return {"provider": "fmp", "data": data}

        # Backup: Yahooquery
        data = self.yahooquery.get_fundamentals(ticker)
        if data:
            return {"provider": "yahooquery", "data": data}

        return {"provider": "none", "data": {}}

    # ─── SECTOR PERFORMANCE ──────────────────────────────────────────
    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Fetch sector performance.
        Primary: AlphaVantage, Backup: FMP.
        """
        # Primary: AlphaVantage
        if self.alphavantage.api_key:
            data = self.alphavantage.get_sector_performance()
            if data:
                return {"provider": "alphavantage", "data": data}

        # Backup: FMP
        if self.fmp.api_key:
            data = self.fmp.get_sector_performance()
            if data:
                return {"provider": "fmp", "data": data}

        return {"provider": "none", "data": {}}

    # ─── REAL-TIME QUOTE ─────────────────────────────────────────────
    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch real-time quote.
        Primary: Finnhub, Backup: yfinance Ticker.info.
        """
        if self.finnhub.client:
            data = self.finnhub.get_quote(ticker)
            if data:
                return {"provider": "finnhub", "data": data}

        # Fallback: yfinance
        info = YFinanceClient.get_ticker_info(ticker)
        if info:
            return {"provider": "yfinance", "data": {
                "c": info.get("currentPrice") or info.get("regularMarketPrice"),
                "pc": info.get("previousClose"),
                "h": info.get("dayHigh"),
                "l": info.get("dayLow"),
                "o": info.get("open"),
            }}

        return {"provider": "none", "data": {}}

    # ─── STATUS / DIAGNOSTICS ────────────────────────────────────────
    def status(self) -> Dict[str, str]:
        """Report which providers are configured and available."""
        return {
            "yfinance": "✅ active",
            "finnhub": "✅ active" if self.finnhub.client else "❌ no API key",
            "fmp": "✅ active" if self.fmp.api_key else "❌ no API key",
            "alphavantage": "✅ active" if self.alphavantage.api_key else "❌ no API key",
            "tiingo": "✅ active" if self.tiingo.api_key else "❌ no API key",
            "yahooquery": "✅ active",
        }


if __name__ == "__main__":
    router = DataRouter()
    print("=== Data Provider Status ===")
    for name, status in router.status().items():
        print(f"  {name:15s} {status}")

    print("\n=== Test: AAPL EOD ===")
    df = router.get_historical_data("AAPL", "1mo")
    print(f"  Got {len(df)} rows")

    print("\n=== Test: Short Interest GME ===")
    short = router.get_short_interest("GME")
    print(f"  Provider: {short['provider']}, Data: {short['data']}")
