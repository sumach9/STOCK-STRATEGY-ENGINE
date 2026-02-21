import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockDataClient:
    """Mock data client for testing purposes."""
    
    @staticmethod
    def get_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Generate mock OHLCV data."""
        np.random.seed(42) # Deterministic for testing
        
        # Decide number of points based on period
        days = 365
        if "y" in period: days = int(period.replace("y", "")) * 365
        elif "mo" in period: days = int(period.replace("mo", "")) * 30
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random walk price
        price = 100.0
        prices = []
        for _ in range(len(dates)):
            price *= (1 + np.random.normal(0, 0.02))
            prices.append(price)
            
        prices = np.array(prices)
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        return df

if __name__ == "__main__":
    client = MockDataClient()
    df = client.get_historical_data("MOCK")
    print(df.head())
