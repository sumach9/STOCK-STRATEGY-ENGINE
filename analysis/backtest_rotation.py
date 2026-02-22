import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.rotation import SectorRotationStrategy
from data.yfinance_client import YFinanceClient

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super(SafeEncoder, self).default(obj)

def run_rotation_backtest():
    print("🚀 Starting Sector Rotation Backtest...")
    
    strategy = SectorRotationStrategy()
    
    # 1. Download Data
    universe = strategy.UNIVERSE + [strategy.BENCHMARK, strategy.CASH_ETF]
    print(f"⏳ Downloading data for {len(universe)} symbols...")
    
    data = yf.download(universe, period="2y", group_by='ticker', progress=False)
    
    # 2. Extract price dataframes
    prices = {}
    for ticker in universe:
        if isinstance(data.columns, pd.MultiIndex):
            prices[ticker] = data[ticker].dropna().copy()
        else:
            prices[ticker] = data.dropna().copy()
            
    # 3. Monthly Rebalance Dates
    spy = prices[strategy.BENCHMARK]
    rebalance_dates = spy.resample('ME').last().index
    
    # 4. Simulation Loop
    cash = 1000.0
    shares = {}
    
    all_dates = spy.index
    history = []
    plan = {}
    
    print("🏃 Running simulation...")
    
    for date in all_dates:
        # Check if today is a rebalance day (monthly)
        if date in rebalance_dates:
            # We use data available UP TO this date
            obs_data = {t: prices[t][:date] for t in prices if not prices[t][:date].empty}
            spy_obs = spy[:date]
            
            # Analyze
            plan = strategy.analyze_universe(obs_data, spy_obs)
            current_weights = plan["weights"]
            
            # Rebalance Logic
            total_value = cash
            for t, s in shares.items():
                price = prices[t].loc[date, 'Close'] if date in prices[t].index else prices[t].iloc[prices[t].index.get_indexer([date], method='pad')[0]]['Close']
                total_value += s * price
            
            cash = total_value
            shares = {}
            for ticker, weight in current_weights.items():
                available_price_df = prices[ticker][:date]
                if not available_price_df.empty:
                    price = available_price_df['Close'].iloc[-1]
                    shares[ticker] = (total_value * weight) / price
                    cash -= (shares[ticker] * price)
                
        # Daily Update
        day_value = cash
        for t, s in shares.items():
            current_price_df = prices[t][:date]
            if not current_price_df.empty:
                price = current_price_df['Close'].iloc[-1]
                day_value += s * price
            
        history.append({
            "date": date.strftime('%Y-%m-%d'),
            "equity": day_value,
            "spy_price": spy.loc[date, 'Close']
        })

    # 5. Calculate Metrics
    df_hist = pd.DataFrame(history)
    df_hist['equity_ret'] = df_hist['equity'].pct_change()
    
    # Normalize comparison
    df_hist['strategy_cum'] = df_hist['equity'] / df_hist['equity'].iloc[0]
    df_hist['spy_cum'] = df_hist['spy_price'] / df_hist['spy_price'].iloc[0]
    
    cagr = (df_hist['strategy_cum'].iloc[-1] ** (252/len(df_hist)) - 1) * 100
    spy_cagr = (df_hist['spy_cum'].iloc[-1] ** (252/len(df_hist)) - 1) * 100
    
    vol = df_hist['equity_ret'].std() * np.sqrt(252) * 100
    sharpe = (cagr - 3.0) / vol if vol > 0 else 0
    
    roll_max = df_hist['strategy_cum'].cummax()
    dd = (df_hist['strategy_cum'] - roll_max) / roll_max
    max_dd = dd.min() * 100
    
    results = {
        "metrics": {
            "cagr": float(round(cagr, 2)),
            "spy_cagr": float(round(spy_cagr, 2)),
            "volatility": float(round(vol, 2)),
            "sharpe": float(round(sharpe, 2)),
            "max_drawdown": float(round(max_dd, 2))
        },
        "equity_curve": df_hist[['date', 'strategy_cum', 'spy_cum']].to_dict(orient='records'),
        "last_rebalance": plan
    }
    
    with open("rotation_results.json", "w") as f:
        json.dump(results, f, indent=4, cls=SafeEncoder)
        
    print(f"✅ Backtest Complete. CAGR: {cagr:.1f}% | Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    run_rotation_backtest()
