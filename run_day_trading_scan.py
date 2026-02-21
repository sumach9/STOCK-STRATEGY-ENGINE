"""
Day Trading Scanner Pipeline
Fetches 5-minute intraday data for sector constituents,
runs the DayTradingScanner, and saves results to day_trading_signals.json.
"""
import json
import time
import yfinance as yf
import pandas as pd
from data.sector_constituents import SECTOR_ETFS
from data.sector_constituents import SECTOR_CONSTITUENTS as FALLBACK_CONSTITUENTS
from analysis.day_trading import DayTradingScanner


def get_sector_constituents_safe():
    from data.sector_constituents import get_sector_constituents
    try:
        return get_sector_constituents()
    except Exception:
        return FALLBACK_CONSTITUENTS


def run_day_trading_scan():
    print("=== Day Trading Scanner ===")
    print("Fetching sector constituents...")
    constituents = get_sector_constituents_safe()

    # Build unique ticker list (top 10 per sector + ETFs)
    tickers = set(SECTOR_ETFS.values())
    for sector, holdings in constituents.items():
        for t in holdings[:10]:
            tickers.add(t)
    tickers = sorted(tickers)
    print(f"Scanning {len(tickers)} tickers...")

    results = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ")
        try:
            # Fetch 5-minute data (last 5 days max for 5m interval)
            df_intra = yf.download(ticker, period="5d", interval="5m", progress=False)
            if isinstance(df_intra.columns, pd.MultiIndex):
                df_intra.columns = df_intra.columns.get_level_values(0)

            # Fetch daily data for Gap & Go and avg vol
            df_daily = yf.download(ticker, period="1mo", interval="1d", progress=False)
            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = df_daily.columns.get_level_values(0)

            daily_avg_vol = float(df_daily['Volume'].tail(10).mean()) if not df_daily.empty else 0

            if df_intra.empty or len(df_intra) < 12:
                print("skip (no data)")
                continue

            result = DayTradingScanner.scan_ticker(
                ticker, df_intra, df_daily, daily_avg_vol
            )

            if result:
                results.append(result)
                print(f"✓ {result['num_signals']} signals, conf={result['composite_confidence']}")
            else:
                print("no signals")

        except Exception as e:
            print(f"error: {e}")
            continue

        time.sleep(0.2)  # Rate limit

    # Sort by composite confidence descending
    results.sort(key=lambda x: x['composite_confidence'], reverse=True)

    # Save
    with open("day_trading_signals.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Saved {len(results)} day trading signals to day_trading_signals.json")
    return results


if __name__ == "__main__":
    run_day_trading_scan()
