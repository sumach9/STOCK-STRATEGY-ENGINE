"""
Financial Modeling Prep (FMP) API Client
Free tier: 250 requests/day
Covers: EOD/Historical data, Fundamentals, Sector Performance
"""
import os
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


class FMPClient:
    """Adapter for Financial Modeling Prep API."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            print("Warning: FMP_API_KEY not found. FMP data will be unavailable.")

    def _get(self, endpoint: str, params: dict = None) -> Any:
        """Generic GET request with API key."""
        if not self.api_key:
            return None
        if params is None:
            params = {}
        params["apikey"] = self.api_key
        try:
            resp = requests.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"FMP API error ({endpoint}): {e}")
            return None

    # --- EOD / Historical ---
    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch daily OHLCV data.
        period: '1mo', '3mo', '6mo', '1y', '5y'
        """
        # Map period to FMP's timeseries param (number of trading days)
        period_map = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "5y": 1260}
        limit = period_map.get(period, 252)

        data = self._get(f"historical-price-full/{ticker}", {"serietype": "line"})
        if not data or "historical" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["historical"][:limit])
        if df.empty:
            return df

        # Rename to match yfinance convention
        df = df.rename(columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]

    # --- Fundamentals ---
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch key financial ratios and metrics."""
        profile = self._get(f"profile/{ticker}")
        ratios = self._get(f"ratios-ttm/{ticker}")

        result = {}
        if profile and isinstance(profile, list) and len(profile) > 0:
            p = profile[0]
            result["profile"] = {
                "sector": p.get("sector"),
                "industry": p.get("industry"),
                "marketCap": p.get("mktCap"),
                "price": p.get("price"),
                "beta": p.get("beta"),
                "description": p.get("description", "")[:200],
            }

        if ratios and isinstance(ratios, list) and len(ratios) > 0:
            r = ratios[0]
            result["ratios"] = {
                "pe": r.get("peRatioTTM"),
                "pb": r.get("priceToBookRatioTTM"),
                "debtToEquity": r.get("debtEquityRatioTTM"),
                "roe": r.get("returnOnEquityTTM"),
                "dividendYield": r.get("dividendYielTTM"),
                "eps": r.get("netIncomePerShareTTM"),
            }

        return result

    # --- Income Statement ---
    def get_income_statement(self, ticker: str, period: str = "annual", limit: int = 4) -> List[Dict]:
        """Fetch income statements."""
        data = self._get(f"income-statement/{ticker}", {"period": period, "limit": limit})
        return data if data else []

    # --- Balance Sheet ---
    def get_balance_sheet(self, ticker: str, period: str = "annual", limit: int = 4) -> List[Dict]:
        """Fetch balance sheet."""
        data = self._get(f"balance-sheet-statement/{ticker}", {"period": period, "limit": limit})
        return data if data else []

    # --- Sector Performance ---
    def get_sector_performance(self) -> Dict[str, float]:
        """Fetch real-time sector performance."""
        data = self._get("sector-performance")
        if not data:
            return {}
        return {item["sector"]: float(item["changesPercentage"].strip("%"))
                for item in data if "sector" in item and "changesPercentage" in item}

    # --- Stock Screener ---
    def get_stock_screener(self, market_cap_min: int = 0, sector: str = None, limit: int = 50) -> List[Dict]:
        """Simple screener."""
        params = {"limit": limit}
        if market_cap_min:
            params["marketCapMoreThan"] = market_cap_min
        if sector:
            params["sector"] = sector
        data = self._get("stock-screener", params)
        return data if data else []


if __name__ == "__main__":
    client = FMPClient()
    if client.api_key:
        df = client.get_historical_data("AAPL", "1mo")
        print("Historical:", len(df), "rows")
        print(df.head())
        print("\nSector Perf:", client.get_sector_performance())
