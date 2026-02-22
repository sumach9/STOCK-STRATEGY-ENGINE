import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
import pandas_ta as ta
from datetime import datetime, timedelta

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.rotation import SectorRotationStrategy

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super(SafeEncoder, self).default(obj)

def run_rotation_backtest():
    print("🚀 Starting Refined Sector Rotation Backtest...")
    
    strategy = SectorRotationStrategy()
    universe = strategy.UNIVERSE + [strategy.BENCHMARK, strategy.CASH_ETF]
    
    # 1. Download Data
    print(f"⏳ Downloading data for {len(universe)} symbols...")
    data = yf.download(universe, period="2y", group_by='ticker', progress=False)
    
    prices = {}
    ma50_dfs = {}
    for ticker in universe:
        if isinstance(data.columns, pd.MultiIndex):
            df = data[ticker].dropna().copy()
        else:
            df = data.dropna().copy()
        prices[ticker] = df
        if 'Close' in df.columns and len(df) > 50:
            ma50_dfs[ticker] = ta.sma(df['Close'], length=50)

    spy = prices[strategy.BENCHMARK]
    rebalance_dates = spy.resample('ME').last().index
    
    # 2. Simulation State
    cash = 1000.0
    shares = {} # ticker: share_count
    
    all_dates = spy.index
    history = []
    plan = {}
    
    print("🏃 Running simulation with Daily Hard Exit logic...")
    
    for date in all_dates:
        # --- A. DAILY HARD EXIT CHECK ---
        # "Sell a sector immediately if Price drops below 50-day MA"
        liquidated_proceeds = 0.0
        to_delete = []
        for ticker, s in shares.items():
            if ticker in [strategy.CASH_ETF, strategy.BENCHMARK]: continue
            
            if ticker in ma50_dfs and date in ma50_dfs[ticker].index:
                curr_price = prices[ticker].loc[date, 'Close']
                curr_ma50 = ma50_dfs[ticker].loc[date]
                
                if curr_price < curr_ma50:
                    # HARD EXIT triggered
                    liquidated_proceeds += s * curr_price
                    to_delete.append(ticker)
        
        for t in to_delete:
            del shares[t]
        cash += liquidated_proceeds

        # --- B. MONTHLY REBALANCE ---
        if date in rebalance_dates:
            # We use data available UP TO this date
            obs_data = {t: prices[t][:date] for t in prices if not prices[t][:date].empty}
            spy_obs = spy[:date]
            
            # Analyze using refined strategy
            plan = strategy.analyze_universe(obs_data, spy_obs)
            
            # Liquidation of whole portfolio to rethink allocation
            total_value = cash
            for t, s in shares.items():
                p = prices[t].loc[date, 'Close'] if date in prices[t].index else prices[t][:date]['Close'].iloc[-1]
                total_value += s * p
            
            # Reset and re-allocate
            shares = {}
            cash = total_value
            new_weights = plan["weights"]
            
            for ticker, weight in new_weights.items():
                if weight <= 0: continue
                p_df = prices[ticker][:date]
                if not p_df.empty:
                    p = p_df['Close'].iloc[-1]
                    shares[ticker] = (total_value * weight) / p
                    cash -= (shares[ticker] * p)

        # --- C. DAILY LOGGING ---
        day_equity = cash
        for t, s in shares.items():
            current_p_df = prices[t][:date]
            if not current_p_df.empty:
                day_equity += s * current_p_df['Close'].iloc[-1]
            
        history.append({
            "date": date.strftime('%Y-%m-%d'),
            "equity": day_equity,
            "spy_price": spy.loc[date, 'Close']
        })

    # 3. Performance Metrics
    df_hist = pd.DataFrame(history)
    df_hist['equity_ret'] = df_hist['equity'].pct_change()
    df_hist['strategy_cum'] = df_hist['equity'] / df_hist['equity'].iloc[0]
    df_hist['spy_cum'] = df_hist['spy_price'] / df_hist['spy_price'].iloc[0]
    
    n_days = len(df_hist)
    cagr = (df_hist['strategy_cum'].iloc[-1] ** (252/n_days) - 1) * 100
    spy_cagr = (df_hist['spy_cum'].iloc[-1] ** (252/n_days) - 1) * 100
    vol = df_hist['equity_ret'].std() * np.sqrt(252) * 100
    sharpe = (cagr - 3.0) / vol if vol > 0 else 0
    roll_max = df_hist['strategy_cum'].cummax()
    max_dd = ((df_hist['strategy_cum'] - roll_max) / roll_max).min() * 100
    
    results = {
        "metrics": {
            "cagr": float(round(cagr, 1)),
            "spy_cagr": float(round(spy_cagr, 1)),
            "volatility": float(round(vol, 1)),
            "sharpe": float(round(sharpe, 2)),
            "max_drawdown": float(round(max_dd, 1))
        },
        "equity_curve": df_hist[['date', 'strategy_cum', 'spy_cum']].to_dict(orient='records'),
        "last_rebalance": plan
    }
    
    with open("rotation_results.json", "w") as f:
        json.dump(results, f, indent=4, cls=SafeEncoder)
        
    print(f"✅ Refined Backtest Complete. CAGR: {cagr:.1f}% | Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    run_rotation_backtest()
