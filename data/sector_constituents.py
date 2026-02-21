import pandas as pd
import requests
from io import StringIO

# Mapping of major stock sectors to their lead SPDR ETFs
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Consumer Staples": "XLP"
}

def get_sector_constituents():
    """
    Fetch S&P 500 components from Wikipedia and group by GICS Sector.
    Returns a dictionary: { 'Sector Name': ['TICKER1', 'TICKER2', ...] }
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        # Renaissance mapping of Wikipedia sector names to our project usage
        # (Wikipedia uses "Information Technology", we used "Technology")
        sector_mapping = {
            "Information Technology": "Technology",
            "Consumer Discretionary": "Consumer Discretionary",
            "Communication Services": "Communication Services",
            "Financials": "Financials",
            "Health Care": "Healthcare",
            "Industrials": "Industrials",
            "Energy": "Energy",
            "Materials": "Materials",
            "Real Estate": "Real Estate",
            "Utilities": "Utilities",
            "Consumer Staples": "Consumer Staples"
        }
        
        # Clean Tickers (SP500 uses dots, e.g. BF.B -> BF-B for yfinance/finnhub)
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        
        constituents = {}
        for wiki_sector, our_sector in sector_mapping.items():
            tickers = df[df['GICS Sector'] == wiki_sector]['Symbol'].tolist()
            if tickers:
                constituents[our_sector] = tickers
                
        return constituents
        
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        # Fallback to a small static list if offline/fails
        return {
            "Technology": ["AAPL", "MSFT", "NVDA"],
            "Financials": ["JPM", "V", "MA"],
            "Healthcare": ["LLY", "UNH", "JNJ"]
        }

# For backward compatibility if imported directly as a dict (though function is preferred)
SECTOR_CONSTITUENTS = get_sector_constituents()
