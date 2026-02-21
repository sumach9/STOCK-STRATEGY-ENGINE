import yfinance as yf
import pandas as pd
import json
import time
from data.sector_constituents import get_sector_constituents
from strategy.indicators import IndicatorLibrary
from strategy.scoring import StrategyScorer

def run_opportunity_scan():
    print("🚀 Starting Opportunity Scan...")
    
    # 1. Get all tickers
    sector_map = get_sector_constituents()
    all_tickers = []
    for tickers in sector_map.values():
        all_tickers.extend(tickers)
    
    # Remove duplicates and clean
    all_tickers = sorted(list(set(all_tickers)))
    print(f"📋 Scanning {len(all_tickers)} tickers from S&P 500...")
    
    # 2. Batch Download Data (Faster than loop)
    # limit to top 50 for speed in this demo/dev phase, or full list?
    # Let's do full list but in chunks if needed. yf.download handles lists well.
    # We need about 1 year of data for 200 EMA.
    
    print("⏳ Downloading historical data (this may take a moment)...")
    try:
        # Downloading all at once might timeout or be messy with multi-index
        # Let's do it in chunks of 50 to be safe and show progress
        chunk_size = 50
        opportunities = []
        scorer = StrategyScorer()
        
        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i+chunk_size]
            print(f"   Processing {i}/{len(all_tickers)}...", end="\r")
            
            data = yf.download(chunk, period="1y", group_by='ticker', progress=False, threads=True)
            
            for ticker in chunk:
                try:
                    # Handle MultiIndex data structure from batch download
                    # Handle MultiIndex data structure from batch download
                    if isinstance(data.columns, pd.MultiIndex):
                        try:
                            # Use cross-section if group_by='ticker' (ticker is level 0)
                            if ticker in data.columns.levels[0]:
                                df = data[ticker].copy()
                            else:
                                df = data.xs(ticker, level=1, axis=1)
                        except KeyError:
                            continue
                    else:
                        df = data.copy()
                        
                    if df.empty: continue
                    
                    # Cleanup: Drop rows with all NaNs
                    df.dropna(how='all', inplace=True)
                    if len(df) < 200: continue # Need enough data for EMA200
                    
                    # 3. Add Indicators
                    df = IndicatorLibrary.add_all_indicators(df)
                    
                    # 4. Detect Setup
                    setup = scorer.detect_setup(df)
                    
                    if setup:
                        # Calculate a score too, for sorting
                        tech_scores = {
                            "trend": scorer.score_trend(df),
                            "momentum": scorer.score_momentum(df),
                            "squeeze": scorer.score_squeeze(df),
                            "breakout": scorer.score_breakout(df)
                        }
                        total_score = scorer.calculate_total_score(tech_scores)
                        
                        opportunities.append({
                            "Ticker": ticker,
                            "Setup": setup,
                            "Score": round(total_score, 1),
                            "Close": round(df['Close'].iloc[-1], 2),
                            "Volume": int(df['Volume'].iloc[-1])
                        })
                        
                except Exception as e:
                    continue # Skip bad tickers
                    
        print(f"\n✅ Scan Complete. Found {len(opportunities)} opportunities.")
        
        # 5. Save Results
        with open("opportunities.json", "w") as f:
            json.dump(opportunities, f, indent=4)
        print("💾 Saved to opportunities.json")
        
    except Exception as e:
        print(f"\n❌ critical error during scan: {e}")

if __name__ == "__main__":
    run_opportunity_scan()
