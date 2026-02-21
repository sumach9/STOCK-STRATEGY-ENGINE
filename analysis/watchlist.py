from strategy.scoring import StrategyScorer
from analysis.strategies import StrategyLibrary

class WatchlistBuilder:
    def __init__(self):
        self.scorer = StrategyScorer()
        self.watchlists = {
            "Oversold Reversal": [],
            "Breakout + Retest": [],      # Strat #1
            "RS Leadership": [],          # Strat #2
            "Gap Fill Candidates": [],    # Strat #3
            "Inst. Accumulation": [],     # Strat #5
            "Base Breakouts": [],         # Strat #6
            "EMA Trend Ride": [],         # Strat #7
            "Volatility Expansion": [],   # Strat #9
            "Earnings Drift": [],         # Strat #11
            "VWAP Hold": [],              # Strat #13
            "Trend Reversals": [],
            "High Volatility / Squeeze": []
        }

    def scan_ticker(self, ticker, df, market_cap_cat, score=0):
        """Analyze a single ticker and add to relevant watchlists."""
        if df.empty or len(df) < 50:
            return

        last = df.iloc[-1]
        
        # --- ORIGINAL WATCHLISTS ---
        # 1. Oversold Reversal (RSI < 35 + Vol Spike)
        rsi = last.get('RSI', 50)
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = (last['Volume'] / avg_vol) if avg_vol > 0 else 1.0
        
        if rsi < 35 and vol_ratio > 1.2:
            self._add(ticker, "Oversold Reversal", f"RSI {rsi:.1f}, Vol {vol_ratio:.1f}x", market_cap_cat, last['Close'], score)

        # 3. Trend Reversals (MACD Bullish Cross)
        macd = last.get('MACD_12_26_9', 0)
        signal = last.get('MACDs_12_26_9', 0)
        prev_macd = df.iloc[-2].get('MACD_12_26_9', 0)
        prev_signal = df.iloc[-2].get('MACDs_12_26_9', 0)
        
        if prev_macd < prev_signal and macd > signal:
            self._add(ticker, "Trend Reversals", "MACD Bullish Crossover", market_cap_cat, last['Close'], score)

        # 4. Squeeze
        bbw = last.get('BBW', 1.0)
        if bbw < 0.10: 
             self._add(ticker, "High Volatility / Squeeze", f"BBW Compressed ({bbw:.2f})", market_cap_cat, last['Close'], score)

        # --- NEW PRO STRATEGIES ---
        if StrategyLibrary.check_breakout_retest(df):
            self._add(ticker, "Breakout + Retest", "Holding support after breakout", market_cap_cat, last['Close'], score)
            
        if StrategyLibrary.check_rs_leadership(df):
            self._add(ticker, "RS Leadership", "Outperforming Trend", market_cap_cat, last['Close'], score)
            
        if StrategyLibrary.check_gap_fill(df):
            self._add(ticker, "Gap Fill Candidates", "Entering Gap Zone", market_cap_cat, last['Close'], score)
            
        accum_setup = StrategyLibrary.check_volume_accumulation(df)
        if accum_setup.get("accumulation"):
            self._add(ticker, "Inst. Accumulation", "Rising OBV flat Price", market_cap_cat, last['Close'], score)
            
        if StrategyLibrary.check_base_breakout(df):
            self._add(ticker, "Base Breakouts", "Vol Contraction -> Expansion", market_cap_cat, last['Close'], score)
            
        if StrategyLibrary.check_ema_trend_ride(df):
            self._add(ticker, "EMA Trend Ride", "10>20>50 EMA Stacked", market_cap_cat, last['Close'], score)
            
        if StrategyLibrary.check_atr_expansion(df):
            self._add(ticker, "Volatility Expansion", "ATR Expanding", market_cap_cat, last['Close'], score)
            
        if StrategyLibrary.check_earnings_drift(df):
            self._add(ticker, "Earnings Drift", "Post-Event Drift Up", market_cap_cat, last['Close'], score)
            
        if StrategyLibrary.check_vwap_hold(df):
            self._add(ticker, "VWAP Hold", "Holding above 5D VWAP", market_cap_cat, last['Close'], score)

    def _add(self, ticker, list_name, reason, cat, price, score):
        self.watchlists[list_name].append({
            "ticker": ticker,
            "reason": reason,
            "category": cat,
            "price": float(price),
            "score": float(score)
        })

    def get_watchlists(self):
        return self.watchlists
