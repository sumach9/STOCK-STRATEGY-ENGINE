import pandas as pd
import yfinance as yf
import json
import time
import os
from data.yfinance_client import YFinanceClient
from data.sector_constituents import SECTOR_ETFS
from data.sector_constituents import SECTOR_CONSTITUENTS as FALLBACK_CONSTITUENTS
from strategy.scoring import StrategyScorer
from strategy.indicators import IndicatorLibrary

def get_sector_constituents_safe():
    from data.sector_constituents import get_sector_constituents
    try:
        return get_sector_constituents()
    except:
        return FALLBACK_CONSTITUENTS

def calculate_performance(df):
    if df.empty:
        return {}
    
    close = df['Close']
    current_price = close.iloc[-1]
    
    perf = {}
    periods = {
        '1D': 1,
        '1W': 5,
        '1M': 21,
        '3M': 63,
        '6M': 126,
        'YTD': 0 # Special case
    }
    
    # Calculate YTD index
    current_year = df.index[-1].year
    ytd_start_idx = df.index.searchsorted(pd.Timestamp(f"{current_year}-01-01"))
    
    for label, days in periods.items():
        if label == 'YTD':
            if ytd_start_idx < len(df):
                start_price = close.iloc[ytd_start_idx]
                perf[label] = ((current_price - start_price) / start_price) * 100
            else:
                perf[label] = 0.0
        else:
            if len(close) > days:
                start_price = close.iloc[-days-1]
                perf[label] = float(((current_price - start_price) / start_price) * 100)
            else:
                perf[label] = 0.0

    return perf

def run_detailed_sector_report():
    print("Starting Detailed Sector Report Generation...")
    yf_client = YFinanceClient()
    scorer = StrategyScorer()
    
    # 1. Sector ETF Performance
    print("\n--- Phase 1: Sector ETF Performance ---")
    sector_performance = []
    
    for sector_name, etf_ticker in SECTOR_ETFS.items():
        print(f"Processing {sector_name} ({etf_ticker})...")
        df = yf_client.get_historical_data(etf_ticker, period="2y") # 2y to safe for YTD/1y
        if df.empty:
            continue
            
        perf = calculate_performance(df)
        
        # Add Volume/Momentum trend for sector itself
        df = IndicatorLibrary.add_all_indicators(df)
        trend_score = float(scorer.score_trend(df))
        momentum_score = float(scorer.score_momentum(df))
        
        # Volume Spike check
        last_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        vol_spike = float((last_vol / avg_vol) if avg_vol > 0 else 1.0)
        
        sector_performance.append({
            "sector": sector_name,
            "etf": etf_ticker,
            "performance": perf,
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "volume_spike": vol_spike,
            "close": float(df['Close'].iloc[-1])
        })
        
    # 2. Constituent Analysis
    print("\n--- Phase 2: Constituent Analysis (Top Holdings) ---")
    constituents = get_sector_constituents_safe()
    detailed_holdings = {}
    
    all_tickers_flat = []
    for s, tickers in constituents.items():
        # Limit to top 15 tickers per sector to save time/api calls for this demo
        # In prod, we'd run this asynchronously or batch it better
        top_tickers = tickers[:15] 
        detailed_holdings[s] = []
        all_tickers_flat.extend(top_tickers)

    # Batch download data for constituents
    # We need to fetch info one by one for Market Cap :( 
    # yfinance batch download gives price history, but not market cap easily without .info
    # Optimization: Use .info only for market cap.
    
    # Let's verify batch download works for price history first
    print(f"Fetching price history for {len(all_tickers_flat)} tickers...")
    
    # Chunking to avoid URL too long errors
    chunk_size = 50
    ticker_chunks = [all_tickers_flat[i:i + chunk_size] for i in range(0, len(all_tickers_flat), chunk_size)]
    
    full_price_data = {}
    
    for chunk in ticker_chunks:
        try:
            data = yf.download(chunk, period="1y", group_by='ticker', progress=False, threads=True)
            # Reformat data to dict of dfs
            for t in chunk:
                try:
                    if len(chunk) == 1:
                        tdf = data
                    else:
                        tdf = data[t]
                    
                    if not tdf.empty:
                        # Drop nans
                        tdf = tdf.dropna(how='all')
                        if not tdf.empty:
                            full_price_data[t] = tdf
                except KeyError:
                    pass
        except Exception as e:
            print(f"Batch download error: {e}")
            
    print("Price history fetched. Analyzing constituents...")
    
    # ... (inside run_detailed_sector_report)
    from analysis.watchlist import WatchlistBuilder
    watchlist_builder = WatchlistBuilder()

    # ... (existing code) ...

    # 3. Constituent Analysis Loop
    for sector, tickers in constituents.items():
        print(f"Analyzing {sector} constituents...")
        sector_holdings_data = []
        
        for t in tickers[:15]: 
            if t not in full_price_data:
                continue
                
            df = full_price_data[t]
            
            # Calculate metrics
            perf = calculate_performance(df)
            df = IndicatorLibrary.add_all_indicators(df)
            
            # --- NEW: Enhanced Scoring ---
            tech_scores = {
                "trend": scorer.score_trend(df),
                "breakout": scorer.score_breakout(df),
                "momentum": scorer.score_momentum(df),
                "squeeze": scorer.score_squeeze(df),
                "volatility": scorer.score_volatility(df),
                "volume_surge": scorer.score_volume_surge(df),
                "price_roc": scorer.score_price_momentum(df)
            }
            total_score = scorer.calculate_total_score(tech_scores)
            
            # Market Cap
            try:
                info = yf.Ticker(t).info
                mkt_cap = info.get('marketCap', 0)
            except:
                mkt_cap = 0
            
            # Classify Cap
            if mkt_cap > 10 * 10**9: cap_cat = "Large-Cap"
            elif mkt_cap > 2 * 10**9: cap_cat = "Mid-Cap"
            else: cap_cat = "Small-Cap"

            # --- NEW: Scan for Watchlists ---
            watchlist_builder.scan_ticker(t, df, cap_cat, total_score)

            sector_holdings_data.append({
                "ticker": t,
                "market_cap": int(mkt_cap) if mkt_cap else 0,
                "market_cap_category": cap_cat,
                "performance": perf,
                "scores": tech_scores,
                "total_score": float(total_score),
                "volume": int(df['Volume'].iloc[-1]),
                "close": float(df['Close'].iloc[-1])
            })
            
        detailed_holdings[sector] = sector_holdings_data

    # Save Reports
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sector_overview": sector_performance,
        "constituents": detailed_holdings
    }
    
    with open("detailed_sector_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Save Watchlists
    with open("watchlists.json", "w") as f:
        json.dump(watchlist_builder.get_watchlists(), f, indent=4)
        
    print("\nReports saved: detailed_sector_report.json, watchlists.json")

from alerts.sector_alerts import SectorAlertSystem
from analysis.watchlist import WatchlistBuilder

if __name__ == "__main__":
    # We need to capture the data as we generate it to build watchlists efficiently
    # The current run_detailed_sector_report function saves to JSON but doesn't return the full objects we need for scanning easily
    # without re-loading. 
    # Ideally, we refactor run_detailed_sector_report to call WatchlistBuilder INSIDE the loop.
    
    # 1. Run Data Gen
    run_detailed_sector_report()
    
    # 2. Load Data for Alerts & Watchlists (Simulating "Post-Process" phase)
    if os.path.exists("detailed_sector_report.json"):
        with open("detailed_sector_report.json", "r") as f:
            data = json.load(f)
            
        # --- ALERTS ---
        print("\n--- Running Alert System ---")
        alert_system = SectorAlertSystem(data)
        alerts = alert_system.generate_alerts()
        
        with open("alerts.json", "w") as f:
            json.dump(alerts, f, indent=4)
        print(f"Generated {len(alerts['alerts'])} alerts.")
        
    # --- WATCHLISTS ---
    # Watchlists are already generated and saved in the main function loop above.
    # No need to re-load them here unless we were doing further processing.
    if os.path.exists("watchlists.json"):
        print("Watchlists available.")

    # --- OPTIONS SCANNING (New Phase 3) ---
    print("\n--- Running Options Scanner ---")
    from analysis.options_scanner import OptionsScanner
    opt_scanner = OptionsScanner()
    
    # scan top 3 holdings of each sector + major ETFs
    # This is to avoid hitting API limits with too many requests
    scan_list = set()
    for s in SECTOR_ETFS.values():
        scan_list.add(s) # Sector ETF itself
        
    # Also add top holdings from our detailed report
    if 'constituents' in data:
        for sector, stocks in data['constituents'].items():
            # Add top 2 by momentum/score
            top_stocks = sorted(stocks, key=lambda x: x.get('total_score', 0), reverse=True)[:2]
            for s in top_stocks:
                scan_list.add(s['ticker'])
                
    print(f"Scanning options for {len(scan_list)} tickers...")
    for t in list(scan_list)[:20]: # Limit to 20 for speed in this MVP run
        opt_scanner.analyze_ticker(t)
        
    opt_alerts = opt_scanner.get_alerts()
    with open("options_activity.json", "w") as f:
        json.dump(opt_alerts, f, indent=4)
        
    print(f"Found {len(opt_alerts)} option alerts. Saved to options_activity.json.")

