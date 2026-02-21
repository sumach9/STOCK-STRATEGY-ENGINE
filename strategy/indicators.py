import pandas as pd
import pandas_ta as ta

class IndicatorLibrary:
    """Library for calculating technical indicators using pandas_ta."""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA, RSI, MACD, and Bollinger Bands to the dataframe."""
        if df.empty:
            return df
            
        # Ensure price columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # EMA
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)

        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)

        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)
            # Find the BB columns dynamically to avoid KeyError
            upper = [c for c in bbands.columns if c.startswith('BBU')][0]
            mid = [c for c in bbands.columns if c.startswith('BBM')][0]
            lower = [c for c in bbands.columns if c.startswith('BBL')][0]
            
            # BB width for squeeze detection
            df['BBW'] = (df[upper] - df[lower]) / df[mid]

        return df

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.yfinance_client import YFinanceClient
    
    client = YFinanceClient()
    df = client.get_historical_data("AAPL", period="1y")
    df = IndicatorLibrary.add_all_indicators(df)
    print(df.tail())
