"""
AlphaVantage API Client
Free tier: 25 requests/day (standard), 75/min (premium)
Covers: Intraday data, Sector Performance
"""
import os
import requests
import pandas as pd
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class AlphaVantageClient:
    """Adapter for Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            print("Warning: ALPHAVANTAGE_API_KEY not found. AlphaVantage data will be unavailable.")

    def _get(self, params: dict) -> Any:
        """Generic GET request with API key."""
        if not self.api_key:
            return None
        params["apikey"] = self.api_key
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # AlphaVantage returns error messages in JSON
            if "Error Message" in data or "Note" in data:
                msg = data.get("Error Message") or data.get("Note", "Rate limit hit")
                print(f"AlphaVantage: {msg}")
                return None
            return data
        except Exception as e:
            print(f"AlphaVantage API error: {e}")
            return None

    # --- Intraday ---
    def get_intraday(self, ticker: str, interval: str = "5min",
                     outputsize: str = "compact") -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.
        interval: '1min', '5min', '15min', '30min', '60min'
        outputsize: 'compact' (last 100 points) or 'full' (full history)
        """
        data = self._get({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": interval,
            "outputsize": outputsize,
        })

        if not data:
            return pd.DataFrame()

        ts_key = f"Time Series ({interval})"
        if ts_key not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[ts_key], orient="index")
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High",
            "3. low": "Low", "4. close": "Close", "5. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Convert to numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # --- Daily ---
    def get_daily(self, ticker: str, outputsize: str = "compact") -> pd.DataFrame:
        """Fetch daily OHLCV data."""
        data = self._get({
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": outputsize,
        })

        if not data or "Time Series (Daily)" not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High",
            "3. low": "Low", "4. close": "Close", "5. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df = df.sort_index()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # --- Sector Performance ---
    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Fetch real-time and historical sector performance.
        Returns dict with keys like 'Rank A: Real-Time Performance', etc.
        """
        data = self._get({"function": "SECTOR"})
        if not data:
            return {}

        result = {}
        # Extract real-time performance
        rt_key = "Rank A: Real-Time Performance"
        if rt_key in data:
            result["realtime"] = {
                k: float(v.strip("%"))
                for k, v in data[rt_key].items()
            }

        # 1-day, 5-day, 1-month, etc.
        for rank_key in data:
            if rank_key.startswith("Rank") and rank_key != rt_key:
                label = rank_key.split(": ", 1)[-1] if ": " in rank_key else rank_key
                result[label] = {
                    k: float(v.strip("%"))
                    for k, v in data[rank_key].items()
                }

        return result

    # --- Technical Indicators ---
    def get_rsi(self, ticker: str, interval: str = "daily",
                time_period: int = 14) -> pd.DataFrame:
        """Fetch RSI indicator."""
        data = self._get({
            "function": "RSI",
            "symbol": ticker,
            "interval": interval,
            "time_period": time_period,
            "series_type": "close",
        })

        if not data or "Technical Analysis: RSI" not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["Technical Analysis: RSI"], orient="index")
        df.index = pd.to_datetime(df.index)
        df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")
        return df.sort_index()

    def get_macd(self, ticker: str, interval: str = "daily") -> pd.DataFrame:
        """Fetch MACD indicator."""
        data = self._get({
            "function": "MACD",
            "symbol": ticker,
            "interval": interval,
            "series_type": "close",
        })

        if not data or "Technical Analysis: MACD" not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["Technical Analysis: MACD"], orient="index")
        df.index = pd.to_datetime(df.index)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_index()


if __name__ == "__main__":
    client = AlphaVantageClient()
    if client.api_key:
        df = client.get_intraday("AAPL", "5min")
        print("Intraday:", len(df), "rows")
        print(df.tail())
        print("\nSector:", client.get_sector_performance().get("realtime", {}))
