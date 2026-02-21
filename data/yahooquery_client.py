"""
Yahooquery Client (no API key needed)
Covers: Short Interest, Fundamentals, Financial Statements
"""
import pandas as pd
from typing import Dict, Any, Optional

try:
    from yahooquery import Ticker
    YAHOOQUERY_AVAILABLE = True
except ImportError:
    YAHOOQUERY_AVAILABLE = False
    print("Warning: yahooquery not installed. Run: pip install yahooquery")


class YahooQueryClient:
    """Adapter for yahooquery — no API key required."""

    def __init__(self):
        if not YAHOOQUERY_AVAILABLE:
            print("YahooQueryClient: yahooquery package not available.")

    @staticmethod
    def _get_ticker(symbol: str):
        """Create a Ticker object."""
        if not YAHOOQUERY_AVAILABLE:
            return None
        try:
            return Ticker(symbol)
        except Exception as e:
            print(f"Yahooquery error creating ticker {symbol}: {e}")
            return None

    # --- Short Interest ---
    def get_short_interest(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch short interest data.
        Returns: short_pct_float, short_ratio (days to cover), shares_short
        """
        t = self._get_ticker(ticker)
        if not t:
            return {}

        try:
            stats = t.key_stats
            if isinstance(stats, dict) and ticker in stats:
                s = stats[ticker]
                if isinstance(s, str):  # Error message
                    return {}
                return {
                    "short_pct_float": s.get("shortPercentOfFloat", 0),
                    "short_ratio": s.get("shortRatio", 0),
                    "shares_short": s.get("sharesShort", 0),
                    "shares_short_prior": s.get("sharesShortPriorMonth", 0),
                    "short_pct_shares_out": s.get("shortPercentOfFloat", 0),
                }
            return {}
        except Exception as e:
            print(f"Yahooquery short interest error for {ticker}: {e}")
            return {}

    # --- Fundamentals ---
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch key financial data: valuation, profitability, balance sheet metrics."""
        t = self._get_ticker(ticker)
        if not t:
            return {}

        result = {}
        try:
            # Summary detail (PE, dividend yield, market cap)
            summary = t.summary_detail
            if isinstance(summary, dict) and ticker in summary:
                s = summary[ticker]
                if not isinstance(s, str):
                    result["valuation"] = {
                        "pe_trailing": s.get("trailingPE"),
                        "pe_forward": s.get("forwardPE"),
                        "dividend_yield": s.get("dividendYield"),
                        "market_cap": s.get("marketCap"),
                        "beta": s.get("beta"),
                        "52w_high": s.get("fiftyTwoWeekHigh"),
                        "52w_low": s.get("fiftyTwoWeekLow"),
                    }

            # Financial data (revenue, margins, etc)
            fin_data = t.financial_data
            if isinstance(fin_data, dict) and ticker in fin_data:
                f = fin_data[ticker]
                if not isinstance(f, str):
                    result["financials"] = {
                        "revenue": f.get("totalRevenue"),
                        "revenue_growth": f.get("revenueGrowth"),
                        "gross_margin": f.get("grossMargins"),
                        "operating_margin": f.get("operatingMargins"),
                        "profit_margin": f.get("profitMargins"),
                        "roe": f.get("returnOnEquity"),
                        "debt_to_equity": f.get("debtToEquity"),
                        "current_ratio": f.get("currentRatio"),
                        "free_cash_flow": f.get("freeCashflow"),
                    }

        except Exception as e:
            print(f"Yahooquery fundamentals error for {ticker}: {e}")

        return result

    # --- Income Statement ---
    def get_income_statement(self, ticker: str, frequency: str = "a") -> pd.DataFrame:
        """
        Fetch income statements.
        frequency: 'a' (annual) or 'q' (quarterly)
        """
        t = self._get_ticker(ticker)
        if not t:
            return pd.DataFrame()
        try:
            df = t.income_statement(frequency=frequency)
            if isinstance(df, pd.DataFrame):
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Yahooquery income statement error: {e}")
            return pd.DataFrame()

    # --- Balance Sheet ---
    def get_balance_sheet(self, ticker: str, frequency: str = "a") -> pd.DataFrame:
        """Fetch balance sheet data."""
        t = self._get_ticker(ticker)
        if not t:
            return pd.DataFrame()
        try:
            df = t.balance_sheet(frequency=frequency)
            if isinstance(df, pd.DataFrame):
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Yahooquery balance sheet error: {e}")
            return pd.DataFrame()

    # --- Cash Flow ---
    def get_cash_flow(self, ticker: str, frequency: str = "a") -> pd.DataFrame:
        """Fetch cash flow statement."""
        t = self._get_ticker(ticker)
        if not t:
            return pd.DataFrame()
        try:
            df = t.cash_flow(frequency=frequency)
            if isinstance(df, pd.DataFrame):
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Yahooquery cash flow error: {e}")
            return pd.DataFrame()

    # --- Analyst Recommendations ---
    def get_recommendations(self, ticker: str) -> pd.DataFrame:
        """Fetch analyst recommendations."""
        t = self._get_ticker(ticker)
        if not t:
            return pd.DataFrame()
        try:
            rec = t.recommendation_trend
            if isinstance(rec, pd.DataFrame):
                return rec
            return pd.DataFrame()
        except Exception as e:
            print(f"Yahooquery recommendations error: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    client = YahooQueryClient()
    if YAHOOQUERY_AVAILABLE:
        short = client.get_short_interest("GME")
        print("Short Interest:", short)
        fund = client.get_fundamentals("AAPL")
        print("Fundamentals:", fund)
