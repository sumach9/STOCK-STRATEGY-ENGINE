import pandas as pd
import numpy as np
from typing import Dict, Any

class StrategyScorer:
    """Calculates weighted scores based on various technical strategies."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "trend": 0.30,
            "breakout": 0.20,
            "momentum": 0.15,
            "squeeze": 0.10,
            "volatility": 0.10,
            "volume_surge": 0.10,
            "price_roc": 0.05
        }

    def score_trend(self, df: pd.DataFrame) -> float:
        """Trend Strategy (EMA20/50/200). Detect direction and health."""
        if df.empty or 'EMA20' not in df.columns:
            return 0.0
        
        last = df.iloc[-1]
        score = 0.0
        
        # Bullish stack: Close > EMA20 > EMA50 > EMA200
        if last['Close'] > last['EMA20']: score += 25
        if last['EMA20'] > last['EMA50']: score += 25
        if last['EMA50'] > last['EMA200']: score += 25
        if last['Close'] > last['EMA200']: score += 25
        
        return score

    def score_breakout(self, df: pd.DataFrame) -> float:
        """Breakout Strategy. New highs + Volume."""
        if df.empty:
            return 0.0
            
        last = df.iloc[-1]
        score = 0.0
        
        # Check if near 52-week high (approx 252 trading days)
        high_52w = df['High'].rolling(window=252, min_periods=1).max().iloc[-1]
        if last['Close'] >= high_52w * 0.98: # Within 2% of high
            score += 50
            
        # Volume breakout: Volume > 1.5x average volume
        avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
        if last['Volume'] > avg_vol * 1.5:
            score += 50
            
        return score

    def score_momentum(self, df: pd.DataFrame) -> float:
        """Momentum Strategy. RSI/MACD strength."""
        if df.empty or 'RSI' not in df.columns:
            return 0.0
            
        last = df.iloc[-1]
        score = 0.0
        
        # RSI health
        if 50 < last['RSI'] < 70:
            score += 50
        elif last['RSI'] >= 70: # Very strong but might be overbought
            score += 30
            
        # MACD Bullish Crossover
        # MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        macd_col = [c for c in df.columns if 'MACD_' in c][0]
        signal_col = [c for c in df.columns if 'MACDs_' in c][0]
        
        if last[macd_col] > last[signal_col]:
            score += 50
            
        return score

    def score_squeeze(self, df: pd.DataFrame) -> float:
        """Volatility Squeeze. Bollinger Band compression."""
        if df.empty or 'BBW' not in df.columns:
            return 0.0
            
        last = df.iloc[-1]
        # Squeeze if current BBW is in the bottom 20% of the last 20 days
        bbw_low = df['BBW'].rolling(window=20).min().iloc[-1]
        bbw_high = df['BBW'].rolling(window=20).max().iloc[-1]
        
        if bbw_high == bbw_low:
            return 0.0
            
        relative_bbw = (last['BBW'] - bbw_low) / (bbw_high - bbw_low)
        
        if relative_bbw < 0.2: # Compressed
            return 100.0
        elif relative_bbw < 0.5:
            return 50.0
        else:
            return 0.0

    def score_volatility(self, df: pd.DataFrame) -> float:
        """Volatility Trend. Reward contracting volatility or stable expansion."""
        if df.empty or 'ATR' not in df.columns:
            return 0.0
            
        last = df.iloc[-1]
        # Compare current ATR to 20-day average ATR
        avg_atr = df['ATR'].rolling(window=20).mean().iloc[-1]
        if avg_atr == 0: return 0.0
        
        atr_ratio = last['ATR'] / avg_atr
        
        # We like it when volatility is not exploding (unless it's a breakout)
        # Stable trend usually has ATR ratio ~ 1.0 or slightly less
        if 0.8 <= atr_ratio <= 1.2:
            return 80.0
        elif atr_ratio < 0.8: # Contracting (Squeeze setup)
            return 60.0
        else: # High volatility expansion (risky but potentially high reward)
            return 40.0

    def score_volume_surge(self, df: pd.DataFrame) -> float:
        """Volume Surge Score. Volume > Avg Volume."""
        if df.empty: return 0.0
        
        last_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_vol == 0: return 0.0
        
        ratio = last_vol / avg_vol
        
        if ratio > 2.0: return 100.0
        elif ratio > 1.5: return 80.0
        elif ratio > 1.1: return 50.0
        return 0.0

    def score_price_momentum(self, df: pd.DataFrame) -> float:
        """Price ROC (Rate of Change) Score."""
        if df.empty: return 0.0
        
        # 1-Week ROC (5 bars)
        closes = df['Close']
        if len(closes) < 6: return 0.0
        
        roc_1w = ((closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6]) * 100
        
        if roc_1w > 5.0: return 100.0
        elif roc_1w > 2.0: return 75.0
        elif roc_1w > 0: return 50.0
        return 0.0

    def score_option_liquidity(self, chain_df: pd.DataFrame) -> float:
        """Option Liquidity Filter. OI > 200, Volume > 100, IV < 120%."""
        if chain_df.empty:
            return 0.0
            
        # This function might be called on a filtered chain (e.g. near-the-money)
        # We'll score based on the average/best liquidity in the provided chain
        liquid_options = chain_df[
            (chain_df['open_interest'] > 200) & 
            (chain_df['volume'] > 100) & 
            (chain_df['greeks'].apply(lambda x: x.get('mid_iv', 0) if isinstance(x, dict) else 0) < 1.20)
        ]
        
        if len(chain_df) == 0: return 0.0
        
        liquidity_ratio = len(liquid_options) / len(chain_df)
        return min(liquidity_ratio * 200, 100) # Cap at 100

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical scores for every row in the dataframe (for backtesting)."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # 1. Trend Score (Vectorized)
        trend_score = np.zeros(len(df))
        if 'EMA20' in df.columns and 'EMA50' in df.columns and 'EMA200' in df.columns:
            trend_score += np.where(df['Close'] > df['EMA20'], 25, 0)
            trend_score += np.where(df['EMA20'] > df['EMA50'], 25, 0)
            trend_score += np.where(df['EMA50'] > df['EMA200'], 25, 0)
            trend_score += np.where(df['Close'] > df['EMA200'], 25, 0)
        df['score_trend'] = trend_score

        # 2. Breakout Score (Vectorized-ish)
        # Using rolling max for 52w high
        high_52w = df['High'].rolling(window=252, min_periods=1).max()
        avg_vol = df['Volume'].rolling(window=20).mean()
        
        breakout_score = np.zeros(len(df))
        breakout_score += np.where(df['Close'] >= high_52w * 0.98, 50, 0)
        breakout_score += np.where(df['Volume'] > avg_vol * 1.5, 50, 0)
        df['score_breakout'] = breakout_score

        # 3. Momentum Score (Vectorized)
        momentum_score = np.zeros(len(df))
        if 'RSI' in df.columns:
            momentum_score += np.where((df['RSI'] > 50) & (df['RSI'] < 70), 50, 0)
            momentum_score += np.where(df['RSI'] >= 70, 30, 0)
            
            macd_col = [c for c in df.columns if 'MACD_' in c][0]
            signal_col = [c for c in df.columns if 'MACDs_' in c][0]
            momentum_score += np.where(df[macd_col] > df[signal_col], 50, 0)
        df['score_momentum'] = momentum_score

        # 4. Squeeze Score (Vectorized)
        squeeze_score = np.zeros(len(df))
        if 'BBW' in df.columns:
            bbw_low = df['BBW'].rolling(window=20).min()
            bbw_high = df['BBW'].rolling(window=20).max()
            relative_bbw = (df['BBW'] - bbw_low) / (bbw_high - bbw_low)
            squeeze_score = np.where(relative_bbw < 0.2, 100, np.where(relative_bbw < 0.5, 50, 0))
        df['score_squeeze'] = squeeze_score

        # Calculate Total Weighted Score
        df['total_score'] = (
            df['score_trend'] * self.weights['trend'] +
            df['score_breakout'] * self.weights['breakout'] +
            df['score_momentum'] * self.weights['momentum'] +
            df['score_squeeze'] * self.weights['squeeze']
        )
        
        return df

    def calculate_total_score(self, technical_scores: Dict[str, float]) -> float:
        """Calculate weighted total score."""
        total = 0.0
        for key, weight in self.weights.items():
            total += technical_scores.get(key, 0.0) * weight
        return total

    def detect_setup(self, df: pd.DataFrame) -> str:
        """
        Detect specific trade setups based on technicals.
        Returns a description string or None.
        """
        if df.empty:
            return None
            
        trend = self.score_trend(df)
        momentum = self.score_momentum(df)
        squeeze = self.score_squeeze(df)
        
        last = df.iloc[-1]
        rsi = last.get('RSI', 50)
        
        # 1. Pullback in Uptrend (Buy the Dip)
        # Strong Trend (>75) but RSI is cooling off (<55)
        if trend >= 75 and rsi < 55 and rsi > 30:
            return "Pullback in Uptrend"
            
        # 2. Oversold in Uptrend (Aggressive Dip Buy)
        if trend >= 50 and rsi <= 30:
            return "Oversold Opportunity"
            
        # 3. Volatility Squeeze (Big Move Coming)
        if squeeze >= 100:
            return "Volatility Squeeze (Coiling)"
            
        # 4. Momentum Breakout (Continuation)
        if momentum >= 80 and trend >= 50:
            return "Momentum Breakout"
            
        return None

if __name__ == "__main__":
    # Test scoring with dummy data
    scorer = StrategyScorer()
    scores = {
        "trend": 100,
        "breakout": 50,
        "momentum": 80,
        "squeeze": 100,
        "option_liquidity": 100
    }
    print(f"Total Score: {scorer.calculate_total_score(scores)}%")
