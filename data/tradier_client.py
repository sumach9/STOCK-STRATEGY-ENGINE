import os
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, Dict

load_dotenv()

class TradierClient:
    """Adapter for Tradier API to fetch options chain data."""
    
    BASE_URL = "https://api.tradier.com/v1/markets"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TRADIER_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }

    def get_options_chain(self, ticker: str, expiration: str) -> pd.DataFrame:
        """
        Fetch the options chain for a given ticker and expiration date.
        
        Args:
            ticker: The stock symbol (e.g., 'AAPL').
            expiration: The expiration date (YYYY-MM-DD).
            
        Returns:
            pd.DataFrame: Options chain data.
        """
        if not self.token:
            print("Error: Tradier API token not provided.")
            return pd.DataFrame()

        url = f"{self.BASE_URL}/options/chains"
        params = {
            "symbol": ticker,
            "expiration": expiration,
            "greeks": "true"
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if "options" in data and data["options"] and "option" in data["options"]:
                options_list = data["options"]["option"]
                # Tradier returns a dict if there's only one option, list otherwise
                if isinstance(options_list, dict):
                    options_list = [options_list]
                return pd.DataFrame(options_list)
            else:
                print(f"No options found for {ticker} on {expiration}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching options for {ticker}: {e}")
            return pd.DataFrame()

    def get_expirations(self, ticker: str) -> list:
        """Fetch available expiration dates for a ticker."""
        if not self.token:
            return []
            
        url = f"{self.BASE_URL}/options/expirations"
        params = {"symbol": ticker, "includeAllRoots": "true", "strikes": "false"}
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if "expirations" in data and data["expirations"]:
                return data["expirations"]["date"]
            return []
        except Exception as e:
            print(f"Error fetching expirations for {ticker}: {e}")
            return []

if __name__ == "__main__":
    # Example usage (requires TRADIER_TOKEN in .env)
    client = TradierClient()
    expirations = client.get_expirations("AAPL")
    if expirations:
        chain = client.get_options_chain("AAPL", expirations[0])
        print(chain.head())
