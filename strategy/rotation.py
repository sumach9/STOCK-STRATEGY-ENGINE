import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Any, Optional

class SectorRotationStrategy:
    """
    Implements the Top-3 Sector Rotation Strategy.
    1. Momentum Score: 0.4*1M + 0.4*3M + 0.2*6M
    2. Trend Filters: Price > MA50, Price > MA200, EMA10 > EMA20
    3. Selection: Top 3 passing filters
    4. Safety: SPY > MA200
    """
    
    UNIVERSE = ["XLE", "XLV", "XLK", "XLY", "XLF", "XLU", "XLI", "XLB", "XLP", "XLRE", "XLC"]
    BENCHMARK = "SPY"
    CASH_ETF = "SHV" # Short-term Treasury ETF

    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> float:
        """
        Momentum Score = 0.40 * 1M + 0.40 * 3M + 0.20 * 6M
        Returns represent percentage returns over the periods.
        """
        if len(df) < 126: # Approx 6 months
            return -999.0
            
        close = df['Close']
        ret_1m = close.pct_change(21).iloc[-1]
        ret_3m = close.pct_change(63).iloc[-1]
        ret_6m = close.pct_change(126).iloc[-1]
        
        score = (0.40 * ret_1m) + (0.40 * ret_3m) + (0.20 * ret_6m)
        return float(score * 100) # In percentage terms for easier reading

    @staticmethod
    def check_trend_filters(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Trend Filters:
        - price > MA50
        - price > MA200
        - EMA10 > EMA20
        """
        if len(df) < 200:
            return {"pass_all": False, "details": "Insufficient data"}
            
        close = df['Close']
        ma50 = ta.sma(close, length=50).iloc[-1]
        ma200 = ta.sma(close, length=200).iloc[-1]
        ema10 = ta.ema(close, length=10).iloc[-1]
        ema20 = ta.ema(close, length=20).iloc[-1]
        
        current_price = close.iloc[-1]
        
        filters = {
            "price_gt_ma50": bool(current_price > ma50),
            "price_gt_ma200": bool(current_price > ma200),
            "ema10_gt_ema20": bool(ema10 > ema20)
        }
        filters["pass_all"] = all(filters.values())
        return filters

    @staticmethod
    def check_market_safety(spy_df: pd.DataFrame) -> bool:
        """Market Safety: SPY > SPY_200MA"""
        if len(spy_df) < 200:
            return False
        close = spy_df['Close']
        ma200 = ta.sma(close, length=200).iloc[-1]
        return close.iloc[-1] > ma200

    def analyze_universe(self, data_map: Dict[str, pd.DataFrame], spy_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the entire sector universe and returns the Top-3 picks.
        data_map: { 'TICKER': DataFrame }
        """
        results = []
        market_safe = self.check_market_safety(spy_df)
        
        if not market_safe:
            return {
                "safe": False,
                "picks": [],
                "weights": {self.CASH_ETF: 1.0},
                "all_results": []
            }

        for ticker in self.UNIVERSE:
            if ticker not in data_map:
                continue
                
            df = data_map[ticker]
            mom_score = self.calculate_momentum_score(df)
            trend = self.check_trend_filters(df)
            
            results.append({
                "ticker": ticker,
                "momentum_score": mom_score,
                "trend_pass": trend["pass_all"],
                "trend_details": trend
            })
            
        # Filter and Sort
        qualified = [r for r in results if r["trend_pass"]]
        qualified.sort(key=lambda x: x["momentum_score"], reverse=True)
        
        top_3 = qualified[:3]
        
        # Determine Weights
        weights = {}
        num_qualified = len(top_3)
        
        if num_qualified == 3:
            for r in top_3: weights[r["ticker"]] = 1/3
        elif num_qualified == 2:
            for r in top_3: weights[r["ticker"]] = 0.5
        elif num_qualified == 1:
            weights[top_3[0]["ticker"]] = 1.0
        else:
            weights[self.CASH_ETF] = 1.0
            
        return {
            "safe": True,
            "picks": top_3,
            "weights": weights,
            "all_results": results
        }
