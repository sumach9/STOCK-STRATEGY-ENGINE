import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OptionsScanner:
    def __init__(self):
        self.alerts = []

    def get_option_chain(self, ticker):
        """Fetch the nearest monthly expiration chain."""
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                return None
            
            # Find closest expiration at least 1 week out (swing trading focus)
            target_date = None
            today = datetime.now().date()
            
            for d_str in exps:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                if (d - today).days > 7:
                    target_date = d_str
                    break
            
            if not target_date:
                target_date = exps[0] # Fallback to nearest
                
            chain = tk.option_chain(target_date)
            return chain
        except Exception as e:
            print(f"Error fetching options for {ticker}: {e}")
            return None

    def analyze_ticker(self, ticker):
        """Analyze options for a specific ticker."""
        chain = self.get_option_chain(ticker)
        if not chain:
            return

        calls = chain.calls
        puts = chain.puts
        
        # 1. Unusual Volume (Vol > OI) on Calls
        # We look for liquid options (Vol > 500) where Vol > OI * 1.5
        unusual_calls = calls[
            (calls['volume'] > 500) & 
            (calls['volume'] > calls['openInterest'] * 1.5)
        ]
        
        for _, row in unusual_calls.iterrows():
            self.alerts.append({
                "ticker": ticker,
                "type": "Unusual Call Volume",
                "strike": row['strike'],
                "expiration": "Next Monthly", # Simplified
                "volume": int(row['volume']),
                "oi": int(row['openInterest']),
                "iv": row['impliedVolatility'],
                "details": f"Vol {int(row['volume'])} > OI {int(row['openInterest'])} (IV {row['impliedVolatility']:.2f})"
            })
            
        # 2. Unusual Volume on Puts
        unusual_puts = puts[
            (puts['volume'] > 500) & 
            (puts['volume'] > puts['openInterest'] * 1.5)
        ]
        
        for _, row in unusual_puts.iterrows():
            self.alerts.append({
                "ticker": ticker,
                "type": "Unusual Put Volume",
                "strike": row['strike'],
                "expiration": "Next Monthly",
                "volume": int(row['volume']),
                "oi": int(row['openInterest']),
                "iv": row['impliedVolatility'],
                "details": f"Vol {int(row['volume'])} > OI {int(row['openInterest'])} (IV {row['impliedVolatility']:.2f})"
            })
            
        # 3. Put/Call Ratio (Volume based)
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        
        if total_call_vol > 0:
            pcr = total_put_vol / total_call_vol
            
            # Extreme Bullish Sentiment (PCR < 0.5)
            if pcr < 0.5 and total_call_vol > 5000:
                 self.alerts.append({
                    "ticker": ticker,
                    "type": "Bullish Sentiment (PCR)",
                    "strike": "N/A",
                    "expiration": "N/A",
                    "volume": int(total_call_vol),
                    "oi": 0,
                    "iv": 0,
                    "details": f"Put/Call Ratio is extremely low ({pcr:.2f}). heavy call buying."
                })
            
            # Extreme Bearish Sentiment (PCR > 2.0)
            elif pcr > 2.0 and total_put_vol > 5000:
                 self.alerts.append({
                    "ticker": ticker,
                    "type": "Bearish Sentiment (PCR)",
                    "strike": "N/A",
                    "expiration": "N/A",
                    "volume": int(total_put_vol),
                    "oi": 0,
                    "iv": 0,
                    "details": f"Put/Call Ratio is extremely high ({pcr:.2f}). heavy put buying."
                })

    def get_alerts(self):
        return self.alerts

if __name__ == "__main__":
    scanner = OptionsScanner()
    # Test with a few liquid tickers
    for t in ["AAPL", "AMD", "TSLA", "NVDA", "SPY"]:
        print(f"Scanning {t}...")
        scanner.analyze_ticker(t)
        
    print(scanner.get_alerts())
