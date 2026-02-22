import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Any, Optional

class SectorRotationStrategy:
    """
    Refined Top-3 Sector Rotation Strategy.
    1. Trend Score: 0.4*Slope30 + 0.4*Slope90 + 0.2*(Price > MA50)
    2. RS Score: 0.5*(30D Return vs SPY) + 0.5*(90D Return vs SPY)
    3. Final Score: 0.5*Trend + 0.5*RS
    4. Selection: Top 3 passing filters (Price > MA50)
    """
    
    UNIVERSE = ["XLE", "XLK", "XLV", "XLF", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU"]
    BENCHMARK = "SPY"
    CASH_ETF = "SHV"

    @staticmethod
    def calculate_slope(prices: pd.Series, length: int) -> float:
        """Calculate linear regression slope of normalized prices over length."""
        if len(prices) < length:
            return 0.0
        y = prices.tail(length).values
        x = np.arange(len(y))
        # Normalize y to start at 1.0
        if y[0] == 0: return 0.0
        y_norm = y / y[0]
        slope, _ = np.polyfit(x, y_norm, 1)
        return float(slope)

    def analyze_universe(self, data_map: Dict[str, pd.DataFrame], spy_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the universe based on the refined composite scoring.
        data_map: { 'TICKER': DataFrame }
        spy_df: Optional SPY DataFrame
        """
        if self.BENCHMARK not in data_map and spy_df is None:
            return {"error": "SPY data required for RS calculation"}
            
        spy_df = spy_df if spy_df is not None else data_map[self.BENCHMARK]
        spy_close = spy_df['Close']
        
        raw_results = []
        for ticker in self.UNIVERSE:
            if ticker not in data_map:
                continue
                
            df = data_map[ticker]
            if len(df) < 90:
                continue
                
            close = df['Close']
            
            # Trend Components
            slope30 = self.calculate_slope(close, 30)
            slope90 = self.calculate_slope(close, 90)
            ma50 = ta.sma(close, length=50).iloc[-1]
            price_gt_ma50 = bool(close.iloc[-1] > ma50)
            
            # RS Components (Relative to SPY)
            ret30 = (close.iloc[-1] / close.iloc[-30]) - 1 if len(close) >= 30 else 0
            ret90 = (close.iloc[-1] / close.iloc[-90]) - 1 if len(close) >= 90 else 0
            
            spy_ret30 = (spy_close.iloc[-1] / spy_close.iloc[-30]) - 1 if len(spy_close) >= 30 else 0
            spy_ret90 = (spy_close.iloc[-1] / spy_close.iloc[-90]) - 1 if len(spy_close) >= 90 else 0
            
            rs30 = ret30 - spy_ret30
            rs90 = ret90 - spy_ret90
            
            raw_results.append({
                "ticker": ticker,
                "slope30": slope30,
                "slope90": slope90,
                "price_gt_ma50": price_gt_ma50,
                "rs30": rs30,
                "rs90": rs90,
                "ret30": ret30,
                "close": close.iloc[-1]
            })
            
        if not raw_results:
            return {"picks": [], "weights": {self.CASH_ETF: 1.0}, "all_results": []}

        # Cross-universe normalization (0 to 1)
        def normalize_series(key):
            vals = [r[key] for r in raw_results]
            v_min, v_max = min(vals), max(vals)
            if v_max == v_min:
                return {r['ticker']: 0.5 for r in raw_results}
            return {r['ticker']: (r[key] - v_min) / (v_max - v_min) for r in raw_results}

        n_slope30 = normalize_series("slope30")
        n_slope90 = normalize_series("slope90")
        n_rs30 = normalize_series("rs30")
        n_rs90 = normalize_series("rs90")

        final_results = []
        for r in raw_results:
            t = r['ticker']
            # Trend Score (0-1)
            trend_score = (0.4 * n_slope30[t]) + (0.4 * n_slope90[t]) + (0.2 * (1.0 if r['price_gt_ma50'] else 0.0))
            
            # RS Score (0-1)
            rs_score = (0.5 * n_rs30[t]) + (0.5 * n_rs90[t])
            
            # Final Composite Score (0-1)
            final_score = (0.5 * trend_score) + (0.5 * rs_score)
            
            final_results.append({
                "ticker": t,
                "trend_score": float(trend_score),
                "rs_score": float(rs_score),
                "momentum_score": float(final_score * 100), # For UI Labeling
                "final_score": float(final_score),
                "hard_exit": bool(not r['price_gt_ma50']),
                "weakness": bool(r['ret30'] < 0),
                "trend_pass": bool(r['price_gt_ma50']),
                "close": float(r['close'])
            })

        # Selection: Rank by Final Score, but must pass Hard Exit filter (MA50)
        qualified = [f for f in final_results if not f['hard_exit']]
        qualified.sort(key=lambda x: x['final_score'], reverse=True)
        
        top_3 = qualified[:3]
        
        # Allocate weights
        weights = {}
        if not top_3:
            weights[self.CASH_ETF] = 1.0
        else:
            w = 1.0 / len(top_3)
            for p in top_3:
                weights[p['ticker']] = w
                
        return {
            "picks": top_3,
            "weights": weights,
            "all_results": final_results
        }
