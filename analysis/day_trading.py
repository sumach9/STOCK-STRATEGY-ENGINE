import pandas as pd
import numpy as np


class DayTradingScanner:
    """
    10 Day-Trading strategies using 5-minute intraday data from yfinance.
    Each check returns: {"signal": bool, "direction": str, "confidence": float, "reason": str}
    """

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _no_signal():
        return {"signal": False, "direction": "", "confidence": 0, "reason": ""}

    @staticmethod
    def _calc_vwap(df: pd.DataFrame) -> pd.Series:
        """Cumulative VWAP for an intraday DataFrame."""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        cum_tp_vol = (tp * df['Volume']).cumsum()
        cum_vol = df['Volume'].cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    # ── 1. Opening Range Breakout (ORB) ─────────────────────────────────
    @staticmethod
    def check_orb(df: pd.DataFrame) -> dict:
        """
        First 6 bars (≈30 min of 5-min data) define the Opening Range.
        Break above = long, break below = short, confirmed by vol spike.
        """
        if df is None or len(df) < 12:
            return DayTradingScanner._no_signal()

        opening = df.iloc[:6]
        or_high = opening['High'].max()
        or_low = opening['Low'].min()
        rest = df.iloc[6:]

        if rest.empty:
            return DayTradingScanner._no_signal()

        last = rest.iloc[-1]
        avg_vol = df['Volume'].mean()
        vol_ratio = last['Volume'] / avg_vol if avg_vol > 0 else 1

        if last['Close'] > or_high and vol_ratio > 1.3:
            conf = min(40 + vol_ratio * 10, 95)
            return {"signal": True, "direction": "long",
                    "confidence": round(conf, 1),
                    "reason": f"ORB Long — broke {or_high:.2f}, vol {vol_ratio:.1f}x"}

        if last['Close'] < or_low and vol_ratio > 1.3:
            conf = min(40 + vol_ratio * 10, 95)
            return {"signal": True, "direction": "short",
                    "confidence": round(conf, 1),
                    "reason": f"ORB Short — broke {or_low:.2f}, vol {vol_ratio:.1f}x"}

        return DayTradingScanner._no_signal()

    # ── 2. VWAP Trend Strategy ──────────────────────────────────────────
    @staticmethod
    def check_vwap_trend(df: pd.DataFrame) -> dict:
        """Price above VWAP + EMA alignment = bullish; below = bearish."""
        if df is None or len(df) < 20:
            return DayTradingScanner._no_signal()

        vwap = DayTradingScanner._calc_vwap(df)
        ema9 = DayTradingScanner._ema(df['Close'], 9)
        ema20 = DayTradingScanner._ema(df['Close'], 20)
        last = df.iloc[-1]
        v = vwap.iloc[-1]

        if pd.isna(v):
            return DayTradingScanner._no_signal()

        if last['Close'] > v and ema9.iloc[-1] > ema20.iloc[-1]:
            dist = (last['Close'] - v) / v * 100
            conf = min(50 + dist * 5, 90)
            return {"signal": True, "direction": "long",
                    "confidence": round(conf, 1),
                    "reason": f"VWAP Trend Long — {dist:.1f}% above VWAP, EMA9>EMA20"}

        if last['Close'] < v and ema9.iloc[-1] < ema20.iloc[-1]:
            dist = (v - last['Close']) / v * 100
            conf = min(50 + dist * 5, 90)
            return {"signal": True, "direction": "short",
                    "confidence": round(conf, 1),
                    "reason": f"VWAP Trend Short — {dist:.1f}% below VWAP, EMA9<EMA20"}

        return DayTradingScanner._no_signal()

    # ── 3. Intraday Momentum Ignition ───────────────────────────────────
    @staticmethod
    def check_momentum_ignition(df: pd.DataFrame) -> dict:
        """RVOL > 2 + breaking intraday high + wide-range candle."""
        if df is None or len(df) < 20:
            return DayTradingScanner._no_signal()

        last = df.iloc[-1]
        avg_vol = df['Volume'].iloc[:-1].mean()
        rvol = last['Volume'] / avg_vol if avg_vol > 0 else 1

        intraday_high = df['High'].iloc[:-1].max()
        candle_range = abs(last['Close'] - last['Open'])
        avg_range = (df['High'] - df['Low']).iloc[:-1].mean()

        if rvol > 2 and last['High'] >= intraday_high and candle_range > avg_range * 1.5:
            conf = min(45 + rvol * 8, 95)
            return {"signal": True, "direction": "long",
                    "confidence": round(conf, 1),
                    "reason": f"Momentum Ignition — RVOL {rvol:.1f}x, new intraday high"}

        return DayTradingScanner._no_signal()

    # ── 4. High Relative Volume (RVOL) Breakouts ───────────────────────
    @staticmethod
    def check_rvol_breakout(df_intraday: pd.DataFrame, daily_avg_vol: float = 0) -> dict:
        """
        RVOL = today's cumulative volume / 10-day avg daily volume.
        RVOL > 3 triggers potential breakout; add momentum filter.
        """
        if df_intraday is None or len(df_intraday) < 10:
            return DayTradingScanner._no_signal()

        today_vol = df_intraday['Volume'].sum()

        if daily_avg_vol <= 0:
            # Fallback: estimate from data
            daily_avg_vol = today_vol  # can't compare, skip
            return DayTradingScanner._no_signal()

        rvol = today_vol / daily_avg_vol

        if rvol > 3:
            last = df_intraday.iloc[-1]
            ema9 = DayTradingScanner._ema(df_intraday['Close'], 9)
            if last['Close'] > ema9.iloc[-1]:
                conf = min(50 + rvol * 5, 95)
                return {"signal": True, "direction": "long",
                        "confidence": round(conf, 1),
                        "reason": f"RVOL Breakout — RVOL {rvol:.1f}x, price > EMA9"}

        return DayTradingScanner._no_signal()

    # ── 5. Pullback to VWAP / EMA (Scalping) ───────────────────────────
    @staticmethod
    def check_pullback_scalp(df: pd.DataFrame) -> dict:
        """Uptrend: buy bounces off 9-EMA or VWAP."""
        if df is None or len(df) < 20:
            return DayTradingScanner._no_signal()

        vwap = DayTradingScanner._calc_vwap(df)
        ema9 = DayTradingScanner._ema(df['Close'], 9)
        ema20 = DayTradingScanner._ema(df['Close'], 20)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        v = vwap.iloc[-1]

        if pd.isna(v):
            return DayTradingScanner._no_signal()

        # Uptrend: EMA9 > EMA20
        if ema9.iloc[-1] > ema20.iloc[-1]:
            # Price touched VWAP or EMA9 and bounced
            touched_ema = prev['Low'] <= ema9.iloc[-2] * 1.002
            touched_vwap = prev['Low'] <= vwap.iloc[-2] * 1.002 if not pd.isna(vwap.iloc[-2]) else False
            bounce = last['Close'] > last['Open']

            if (touched_ema or touched_vwap) and bounce:
                return {"signal": True, "direction": "long",
                        "confidence": 65.0,
                        "reason": "Pullback Scalp — bounce off VWAP/EMA9 in uptrend"}

        return DayTradingScanner._no_signal()

    # ── 6. Reversal / Fade Strategy ─────────────────────────────────────
    @staticmethod
    def check_reversal_fade(df: pd.DataFrame) -> dict:
        """Price extends 2+ ATR from VWAP + exhaustion vol → fade."""
        if df is None or len(df) < 20:
            return DayTradingScanner._no_signal()

        vwap = DayTradingScanner._calc_vwap(df)
        v = vwap.iloc[-1]
        if pd.isna(v):
            return DayTradingScanner._no_signal()

        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        last = df.iloc[-1]
        dist = last['Close'] - v

        avg_vol = df['Volume'].mean()
        vol_ratio = last['Volume'] / avg_vol if avg_vol > 0 else 1

        # Overextended up → short fade
        if dist > 2 * atr and vol_ratio > 1.5:
            if last['Close'] < last['Open']:  # bearish candle
                return {"signal": True, "direction": "short",
                        "confidence": 60.0,
                        "reason": f"Reversal Fade Short — {dist/atr:.1f} ATR from VWAP, exhaustion vol"}

        # Overextended down → long fade
        if dist < -2 * atr and vol_ratio > 1.5:
            if last['Close'] > last['Open']:  # bullish candle
                return {"signal": True, "direction": "long",
                        "confidence": 60.0,
                        "reason": f"Reversal Fade Long — {abs(dist)/atr:.1f} ATR below VWAP, bounce"}

        return DayTradingScanner._no_signal()

    # ── 7. Gap & Go Strategy (Daily Proxy) ──────────────────────────────
    @staticmethod
    def check_gap_and_go(df_daily: pd.DataFrame) -> dict:
        """Gap-up > 3% with high volume, holding above gap level."""
        if df_daily is None or len(df_daily) < 5:
            return DayTradingScanner._no_signal()

        prev_close = df_daily['Close'].iloc[-2]
        today_open = df_daily['Open'].iloc[-1]
        today_close = df_daily['Close'].iloc[-1]
        gap_pct = (today_open - prev_close) / prev_close * 100

        avg_vol = df_daily['Volume'].iloc[-10:].mean()
        vol_ratio = df_daily['Volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1

        if gap_pct > 3 and vol_ratio > 1.5 and today_close >= today_open:
            conf = min(50 + gap_pct * 3, 90)
            return {"signal": True, "direction": "long",
                    "confidence": round(conf, 1),
                    "reason": f"Gap & Go — {gap_pct:.1f}% gap-up, vol {vol_ratio:.1f}x, holding"}

        if gap_pct < -3 and vol_ratio > 1.5 and today_close <= today_open:
            conf = min(50 + abs(gap_pct) * 3, 90)
            return {"signal": True, "direction": "short",
                    "confidence": round(conf, 1),
                    "reason": f"Gap & Go Short — {gap_pct:.1f}% gap-down, vol {vol_ratio:.1f}x"}

        return DayTradingScanner._no_signal()

    # ── 8. Breakout → Retest (Intraday) ─────────────────────────────────
    @staticmethod
    def check_intraday_breakout_retest(df: pd.DataFrame) -> dict:
        """Breaks intraday high, pulls back, holds on low vol."""
        if df is None or len(df) < 20:
            return DayTradingScanner._no_signal()

        # Look at last 20 bars
        recent = df.iloc[-20:]
        prior_high = recent['High'].iloc[:-5].max()

        # Did we break it in the last ~5 bars?
        broke = any(recent['High'].iloc[-5:] > prior_high)
        if not broke:
            return DayTradingScanner._no_signal()

        last = recent.iloc[-1]
        # Is current price near the breakout level (within 1%)?
        if 0.99 * prior_high <= last['Close'] <= 1.01 * prior_high:
            avg_vol = recent['Volume'].mean()
            if last['Volume'] < avg_vol * 0.8:  # Low vol retest
                return {"signal": True, "direction": "long",
                        "confidence": 70.0,
                        "reason": f"Intraday B+R — retesting {prior_high:.2f} on low vol"}

        return DayTradingScanner._no_signal()

    # ── 9. Lunch-Time Fade ──────────────────────────────────────────────
    @staticmethod
    def check_lunchtime_fade(df: pd.DataFrame) -> dict:
        """
        Afternoon bars drift back to VWAP with contracting volume.
        We detect volume declining in the last ~12 bars and price converging to VWAP.
        """
        if df is None or len(df) < 30:
            return DayTradingScanner._no_signal()

        vwap = DayTradingScanner._calc_vwap(df)
        v = vwap.iloc[-1]
        if pd.isna(v):
            return DayTradingScanner._no_signal()

        # Check if volume is declining
        recent_vol = df['Volume'].iloc[-12:]
        vol_declining = recent_vol.iloc[-3:].mean() < recent_vol.iloc[:3].mean() * 0.6

        # Price converging to VWAP
        dist_pct = abs(df['Close'].iloc[-1] - v) / v * 100
        was_far = abs(df['Close'].iloc[-12] - vwap.iloc[-12]) / vwap.iloc[-12] * 100 if not pd.isna(vwap.iloc[-12]) else 0

        if vol_declining and dist_pct < 0.3 and was_far > 0.5:
            direction = "short" if df['Close'].iloc[-12] > vwap.iloc[-12] else "long"
            return {"signal": True, "direction": direction,
                    "confidence": 55.0,
                    "reason": f"Lunch Fade — price converging to VWAP, vol declining"}

        return DayTradingScanner._no_signal()

    # ── 10. Order Flow Proxy ────────────────────────────────────────────
    @staticmethod
    def check_order_flow_proxy(df: pd.DataFrame) -> dict:
        """
        Without Level 2 data we approximate buying/selling pressure:
        Buy vol ≈ Volume on green bars, Sell vol ≈ Volume on red bars.
        Strong imbalance → directional bias.
        """
        if df is None or len(df) < 20:
            return DayTradingScanner._no_signal()

        recent = df.iloc[-20:]
        green = recent[recent['Close'] >= recent['Open']]
        red = recent[recent['Close'] < recent['Open']]

        buy_vol = green['Volume'].sum()
        sell_vol = red['Volume'].sum()
        total = buy_vol + sell_vol

        if total == 0:
            return DayTradingScanner._no_signal()

        buy_pct = buy_vol / total

        if buy_pct > 0.70:
            conf = min(50 + buy_pct * 30, 85)
            return {"signal": True, "direction": "long",
                    "confidence": round(conf, 1),
                    "reason": f"Order Flow — {buy_pct:.0%} buy pressure (last 20 bars)"}

        if buy_pct < 0.30:
            sell_pct = 1 - buy_pct
            conf = min(50 + sell_pct * 30, 85)
            return {"signal": True, "direction": "short",
                    "confidence": round(conf, 1),
                    "reason": f"Order Flow — {sell_pct:.0%} sell pressure (last 20 bars)"}

        return DayTradingScanner._no_signal()

    # ── Master Scanner ──────────────────────────────────────────────────
    @staticmethod
    def scan_ticker(ticker: str, df_intraday: pd.DataFrame,
                    df_daily: pd.DataFrame = None,
                    daily_avg_vol: float = 0) -> dict:
        """
        Run all 10 strategies on a single ticker.
        Returns consolidated result with composite confidence.
        """
        strategies = {
            "Opening Range Breakout": DayTradingScanner.check_orb(df_intraday),
            "VWAP Trend": DayTradingScanner.check_vwap_trend(df_intraday),
            "Momentum Ignition": DayTradingScanner.check_momentum_ignition(df_intraday),
            "RVOL Breakout": DayTradingScanner.check_rvol_breakout(df_intraday, daily_avg_vol),
            "Pullback Scalp": DayTradingScanner.check_pullback_scalp(df_intraday),
            "Reversal Fade": DayTradingScanner.check_reversal_fade(df_intraday),
            "Gap & Go": DayTradingScanner.check_gap_and_go(df_daily) if df_daily is not None else DayTradingScanner._no_signal(),
            "Breakout Retest": DayTradingScanner.check_intraday_breakout_retest(df_intraday),
            "Lunch-Time Fade": DayTradingScanner.check_lunchtime_fade(df_intraday),
            "Order Flow": DayTradingScanner.check_order_flow_proxy(df_intraday),
        }

        triggered = {name: res for name, res in strategies.items() if res.get("signal")}

        if not triggered:
            return None

        # Composite confidence = average of triggered confidences
        avg_conf = sum(r["confidence"] for r in triggered.values()) / len(triggered)
        # Bonus for multiple signals
        multi_bonus = min(len(triggered) * 5, 20)
        composite = min(avg_conf + multi_bonus, 99)

        price = float(df_intraday['Close'].iloc[-1]) if len(df_intraday) > 0 else 0

        return {
            "ticker": ticker,
            "composite_confidence": round(composite, 1),
            "price": round(price, 2),
            "num_signals": len(triggered),
            "strategies": {name: {
                "direction": r["direction"],
                "confidence": r["confidence"],
                "reason": r["reason"]
            } for name, r in triggered.items()}
        }
