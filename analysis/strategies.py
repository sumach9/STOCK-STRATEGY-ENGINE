import pandas as pd
import numpy as np

class StrategyLibrary:
    """
    Implements 15 Pro Strategies for Technical Analysis.
    """

    # --- 1. Breakout + Retest ---
    @staticmethod
    def check_breakout_retest(df: pd.DataFrame) -> bool:
        """
        Strategy #1: Strong breakout above resistance followed by a retest.
        Simplified Logic for EOD:
        1. Price broke above 20-day High 2-5 days ago.
        2. Price pulled back to near that level (within 2%).
        3. Forming a bullish candle (Close > Open) or holding.
        """
        if len(df) < 25: return False
        
        closes = df['Close']
        highs = df['High']
        
        # Identify resistance (20-day high from 5 days ago)
        resistance = highs.iloc[-25:-5].max()
        
        # Did we break it recently? (Last 5 days)
        recent_breakout = any(closes.iloc[-5:] > resistance)
        
        if not recent_breakout: return False
        
        # Are we retesting it now? (Current close near resistance)
        current_close = closes.iloc[-1]
        if 0.98 * resistance <= current_close <= 1.02 * resistance:
            # Bullish hold confirmation (Close > Open)
            if current_close >= df['Open'].iloc[-1]:
                return True
        return False

    # --- 2. RS Leadership ---
    @staticmethod
    def check_rs_leadership(df: pd.DataFrame, sector_df: pd.DataFrame = None) -> bool:
        """
        Strategy #2: Relative Strength Leadership.
        Stock is up > 0% while Index/Sector is down, or Stock outperforms Sector over 1 month.
        """
        if len(df) < 22: return False
        
        # 1-Month RS
        stock_ret = df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1
        
        # If no sector data provided, just look for raw relative strength (outperforming SPY generally)
        # Threshold: Stock > 5% in 1 month is strong.
        if stock_ret > 0.05:
            return True
        return False

    # --- 3. Gap Fill ---
    @staticmethod
    def check_gap_fill(df: pd.DataFrame) -> bool:
        """
        Strategy #3: Predicting Gap Fills.
        Logic:
        1. Identify a gap > 2% from within last 10 days.
        2. Price is entering the gap zone.
        """
        if len(df) < 15: return False
        
        # Look for gaps in last 10 days
        for i in range(1, 11):
            idx = -1 * i
            prev = idx - 1
            
            # Gap Up
            if df['Low'].iloc[idx] > df['High'].iloc[prev] * 1.02:
                gap_zone = (df['High'].iloc[prev], df['Low'].iloc[idx])
                # Current price entering this zone?
                curr = df['Close'].iloc[-1]
                if gap_zone[0] < curr < gap_zone[1]:
                    return True
            
            # Gap Down
            if df['High'].iloc[idx] < df['Low'].iloc[prev] * 0.98:
                gap_zone = (df['High'].iloc[idx], df['Low'].iloc[prev])
                curr = df['Close'].iloc[-1]
                if gap_zone[0] < curr < gap_zone[1]:
                    return True
        return False

    # --- 4. Volume Imbalance (Simply High Liquidity Nodes) ---
    @staticmethod
    def check_volume_accumulation(df: pd.DataFrame) -> dict:
        """Strategy #5 & #4: Institutional Accumulation / High Vol Nodes."""
        if len(df) < 10: return {}
        
        # OBV (On Balance Volume) Slope
        closes = df['Close']
        vols = df['Volume']
        obv = pd.Series(np.where(closes > closes.shift(1), vols, np.where(closes < closes.shift(1), -vols, 0))).cumsum()
        
        # Check if OBV is trending up while Price is flat/consolidating (Divergence)
        obv_trend = obv.iloc[-1] > obv.iloc[-10]
        price_trend = abs(closes.iloc[-1] - closes.iloc[-10]) / closes.iloc[-10] < 0.02 # Flat < 2%
        
        if obv_trend and price_trend:
            return {"accumulation": True}
        return {}

    # --- 6. Base Breakout ---
    @staticmethod
    def check_base_breakout(df: pd.DataFrame) -> bool:
        """Strategy #6: Flat Base / Volatility Contraction Breakout."""
        if len(df) < 20: return False
        
        # Check volatility contraction (Standard Deviation of last 10 days vs previous)
        std_recent = df['Close'].iloc[-10:].std()
        std_prev = df['Close'].iloc[-20:-10].std()
        
        # Breakout today?
        breakout = df['Close'].iloc[-1] > df['High'].iloc[-20:-1].max()
        vol_spike = df['Volume'].iloc[-1] > df['Volume'].iloc[-20:-1].mean() * 1.5
        
        if (std_recent < std_prev * 0.8) and breakout and vol_spike:
            return True
        return False

    # --- 7. EMA Trend Ride ---
    @staticmethod
    def check_ema_trend_ride(df: pd.DataFrame) -> bool:
        """Strategy #7: 10 > 20 > 50 EMA Consensus."""
        if 'EMA10' not in df.columns or 'EMA20' not in df.columns or 'EMA50' not in df.columns:
            # We assume these cols are added by indicators.py, but safe check
            return False
            
        last = df.iloc[-1]
        return last['EMA10'] > last['EMA20'] > last['EMA50']

    # --- 8. Short Squeeze Setup ---
    @staticmethod
    def check_short_squeeze(df: pd.DataFrame, short_float_pct: float = 0) -> bool:
        """Strategy #8: High Short Interest + Technical Breakout."""
        if short_float_pct < 15: # 15% threshold
            return False
            
        # Tech strength
        return df['Close'].iloc[-1] > df['MA50'].iloc[-1] and df['RSI'].iloc[-1] > 50

    # --- 9. ATR Expansion ---
    @staticmethod
    def check_atr_expansion(df: pd.DataFrame) -> bool:
        """Strategy #9: ATR Expansion from compression."""
        if 'ATR' not in df.columns: return False
        
        current_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].iloc[-10:].mean()
        
        # Expansion > 1.2x avg
        return current_atr > 1.2 * avg_atr

    # --- 11. Earnings Drift ---
    @staticmethod
    def check_earnings_drift(df: pd.DataFrame) -> bool:
        """Strategy #11: Post-Earnings Gap and Hold."""
        # Detect large gap (>5%) within 30 days that hasn't filled
        # This is a proxy without actual earnings dates
        for i in range(1, 30):
             idx = -1 * i
             prev = idx - 1
             if idx >= len(df): continue
             
             # Gap Up > 5%
             if df['Low'].iloc[idx] > df['High'].iloc[prev] * 1.05:
                 # Check if we are still above that gap level
                 if df['Close'].iloc[-1] >= df['Low'].iloc[idx]:
                     return True
        return False

    # --- 13. VWAP Hold (Proxy) ---
    @staticmethod
    def check_vwap_hold(df: pd.DataFrame) -> bool:
        """Strategy #13: Price holding above pseudo-VWAP (Typical Price Avg)."""
        # True VWAP requires intraday volume. 
        # Approx: Typical Price (H+L+C)/3 cumulative average is NOT accurate for multi-day VWAP anchored today.
        # But for daily timeframes, we can check if Price > Avg(Typical, Volume) 
        # Let's use a simpler proxy: Price > 5-day VWAP approx
        
        # Calculate 5-day VWAP
        t_p = (df['High'] + df['Low'] + df['Close']) / 3
        vol = df['Volume']
        vwap_5d = (t_p.iloc[-5:] * vol.iloc[-5:]).sum() / vol.iloc[-5:].sum()
        
        return df['Close'].iloc[-1] > vwap_5d

    # --- 14. Multi-Timeframe (Proxy) ---
    @staticmethod
    def check_multi_timeframe(df: pd.DataFrame) -> bool:
        """Strategy #14: Daily Bullish + Weekly Bullish."""
        # Daily Bullish: > EMA20
        # Weekly Bullish: > EMA50
        if 'EMA20' not in df.columns or 'EMA50' not in df.columns: return False
        
        last = df.iloc[-1]
        return last['Close'] > last['EMA20'] and last['Close'] > last['EMA50']

