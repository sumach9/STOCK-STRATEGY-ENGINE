import yfinance as yf
import pandas as pd
from typing import Optional
from datetime import datetime

class YFinanceClient:
    """Adapter for yfinance to fetch historical stock data and options."""
    
    @staticmethod
    def get_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given ticker.
        
        Args:
            ticker: The stock symbol (e.g., 'AAPL').
            period: The data period (e.g., '1y', 'max').
            interval: The data interval (e.g., '1d', '1h').
            
        Returns:
            pd.DataFrame: Historical data with Open, High, Low, Close, Volume.
        """
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                print(f"Warning: No data found for {ticker}")
                return pd.DataFrame() # Return empty if no data found
            
            # Flatten MultiIndex columns if present (fix for yfinance headers)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    def get_options_chain(ticker: str, expiration_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch options chain for a given ticker.
        If expiration_date is None, uses the nearest expiration.
        """
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if not expirations:
                print(f"No options found for {ticker}")
                return pd.DataFrame()
                
            expiry = expiration_date if expiration_date in expirations else expirations[0]
            if expiration_date and expiration_date not in expirations:
                print(f"Warning: {expiration_date} not found. Using {expiry}")
                
            opt = stock.option_chain(expiry)
            calls = opt.calls
            puts = opt.puts
            calls['type'] = 'call'
            puts['type'] = 'put'
            
            # Standardization for scoring engine compatibility
            start_date = datetime.strptime(expiry, '%Y-%m-%d')
            calls['expiration'] = start_date
            puts['expiration'] = start_date
            
            # Combine
            chain = pd.concat([calls, puts])
            return chain
            
        except Exception as e:
            print(f"Error fetching options for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_ticker_info(ticker: str) -> dict:
        """
        Fetch metadata for a ticker (market cap, sector, etc).
        """
        try:
            t = yf.Ticker(ticker)
            return t.info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {}

if __name__ == "__main__":
    # Quick test
    client = YFinanceClient()
    df = client.get_historical_data("AAPL", period="1mo")
    print(df.head())
