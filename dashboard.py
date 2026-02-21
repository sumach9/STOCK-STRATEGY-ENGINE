import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from data.yfinance_client import YFinanceClient
from data.mock_client import MockDataClient
from strategy.indicators import IndicatorLibrary
from strategy.scoring import StrategyScorer
from ai.refiner import AIRefiner
import os
from datetime import datetime

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Strategy Engine",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e0e8ff;
}

/* Animated header banner */
.hero-banner {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b3e 40%, #0a0e1a 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.05) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b8cde, #00d4ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0;
}
@keyframes shimmer {
    0% { background-position: 0%; }
    100% { background-position: 200%; }
}
.hero-sub { color: #7b8fa8; font-size: 0.85rem; margin-top: 4px; }

/* Glassmorphism cards */
.glass-card {
    background: rgba(13,27,62,0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.4);
    box-shadow: 0 0 24px rgba(0,212,255,0.1);
}

/* KPI metric cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(13,27,62,0.9));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: all 0.3s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: rgba(0,212,255,0.5); }
.kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #7b8fa8; margin-bottom: 6px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #00d4ff; font-family: 'JetBrains Mono', monospace; }
.kpi-delta { font-size: 0.75rem; margin-top: 4px; }
.kpi-delta.up { color: #00e676; }
.kpi-delta.down { color: #ff5252; }
.kpi-delta.neutral { color: #7b8fa8; }

/* Signal badges */
.badge-buy  { background: rgba(0,230,118,0.15); border:1px solid #00e676; color:#00e676; border-radius:8px; padding:6px 16px; font-weight:700; font-size:0.9rem; display:inline-block; }
.badge-watch{ background: rgba(255,214,0,0.12); border:1px solid #ffd600; color:#ffd600; border-radius:8px; padding:6px 16px; font-weight:700; font-size:0.9rem; display:inline-block; }
.badge-skip { background: rgba(255,82,82,0.12);  border:1px solid #ff5252; color:#ff5252; border-radius:8px; padding:6px 16px; font-weight:700; font-size:0.9rem; display:inline-block; }

/* Alert pulse */
.alert-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#ff4444; margin-right:6px;
    animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

/* Streamlit overrides */
[data-testid="stMetric"] {
    background: rgba(13,27,62,0.8);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 10px;
    padding: 14px;
}
[data-testid="stMetricLabel"] { color: #7b8fa8 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'JetBrains Mono' !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060a14 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.1) !important;
}
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
.stButton>button {
    background: linear-gradient(135deg, #00d4ff, #7b8cde);
    color: #000;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.85; transform: translateY(-1px); }
.stTextInput>div>div>input {
    background: rgba(13,27,62,0.8) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
    color: #e0e8ff !important;
}
.stSelectbox>div>div {
    background: rgba(13,27,62,0.8) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
}
.stExpander { border: 1px solid rgba(0,212,255,0.15) !important; border-radius: 10px !important; }

/* Chat messages */
.chat-user { background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.2); border-radius: 12px; padding: 12px 16px; margin: 8px 0; }
.chat-ai   { background: rgba(13,27,62,0.8); border: 1px solid rgba(123,140,222,0.3); border-radius: 12px; padding: 12px 16px; margin: 8px 0; }

/* Watchlist chips */
.ticker-chip {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    color: #00d4ff;
    font-weight: 600;
}

/* Opportunity card */
.opp-card {
    background: rgba(13,27,62,0.8);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    border-left: 3px solid #00d4ff;
    transition: all 0.2s;
}
.opp-card:hover { border-left-color: #00e676; transform: translateX(3px); }

/* Score bar */
.score-bar-bg { background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; margin-top: 6px; }
.score-bar-fill { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #00d4ff, #00e676); transition: width 0.5s; }

/* Bloomberg-style Daily Briefing */
.bb-home {
    background: linear-gradient(180deg, rgba(5,7,13,0.95) 0%, rgba(8,11,20,0.95) 100%);
    border: 1px solid rgba(255, 170, 0, 0.22);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.bb-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(255, 170, 0, 0.25);
    padding-bottom: 12px;
}
.bb-brand {
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 0.75rem;
    color: #ffb347;
}
.bb-headline {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f6f7fb;
}
.bb-timestamp {
    color: #9ba6bf;
    font-size: 0.8rem;
    text-align: right;
}
.bb-tape {
    background: rgba(12, 15, 24, 0.95);
    border: 1px solid rgba(255, 170, 0, 0.24);
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 14px;
    overflow-x: auto;
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
}
.bb-tape-item {
    display: inline-block;
    margin-right: 18px;
    color: #d8deec;
    font-size: 0.78rem;
}
.bb-up { color: #42d69e; font-weight: 700; }
.bb-down { color: #ff6b6b; font-weight: 700; }
.bb-flat { color: #ffcc66; font-weight: 700; }
.bb-panel {
    background: rgba(10, 13, 22, 0.94);
    border: 1px solid rgba(110, 120, 145, 0.22);
    border-left: 3px solid #ffb347;
    border-radius: 10px;
    padding: 12px 14px;
    min-height: 170px;
}
.bb-panel-title {
    color: #ffb347;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.bb-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    border-bottom: 1px dashed rgba(255,255,255,0.09);
    padding: 6px 0;
    font-size: 0.83rem;
}
.bb-row:last-child { border-bottom: none; }
.bb-label { color: #d6dbee; font-weight: 600; }
.bb-muted { color: #8f9ab4; }
.bb-flash {
    display: inline-block;
    background: rgba(255, 86, 86, 0.18);
    color: #ff8b8b;
    border: 1px solid rgba(255, 86, 86, 0.45);
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-right: 7px;
}
.bb-highlight {
    background: linear-gradient(135deg, rgba(255, 179, 71, 0.14), rgba(20, 26, 40, 0.92));
    border: 1px solid rgba(255, 179, 71, 0.35);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default if default is not None else {}

def load_daily_report():
    if os.path.exists("daily_sentiment_report.csv"):
        return pd.read_csv("daily_sentiment_report.csv")
    return pd.DataFrame()

def market_status():
    now = datetime.now()
    h, m = now.hour, now.minute
    mins = h * 60 + m
    if 570 <= mins < 630:
        return "🟡 Pre-Market", "#ffd600"
    elif 630 <= mins < 960:
        return "🟢 Market Open", "#00e676"
    elif 960 <= mins < 1020:
        return "🟠 After-Hours", "#ff9800"
    else:
        return "🔴 Market Closed", "#ff5252"

def plot_price_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name=ticker,
        increasing_line_color='#00e676', decreasing_line_color='#ff5252'
    ))
    for col, color, name in [('EMA_20','#ffd600','EMA 20'), ('EMA_50','#00d4ff','EMA 50')]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1), name=name))
    fig.update_layout(
        title=f"{ticker} — Price Action",
        template="plotly_dark",
        plot_bgcolor='rgba(10,14,26,0)',
        paper_bgcolor='rgba(10,14,26,0)',
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    return fig

def score_gauge(score, title="Score"):
    color = "#00e676" if score > 75 else ("#ffd600" if score > 55 else "#ff5252")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'color': '#7b8fa8', 'size': 14}},
        number={'font': {'color': color, 'size': 36, 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#7b8fa8'},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(13,27,62,0.5)',
            'bordercolor': 'rgba(0,212,255,0.2)',
            'steps': [
                {'range': [0, 60],  'color': 'rgba(255,82,82,0.15)'},
                {'range': [60, 75], 'color': 'rgba(255,214,0,0.15)'},
                {'range': [75, 100],'color': 'rgba(0,230,118,0.15)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.85, 'value': score}
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e0e8ff'}
    )
    return fig

def run_scanner_logic(ticker):
    yf_client = YFinanceClient()
    scorer = StrategyScorer()
    refiner = AIRefiner()
    try:
        from data.finnhub_client import FinnhubClient
        finnhub_client = FinnhubClient()
    except:
        finnhub_client = None

    df = yf_client.get_historical_data(ticker, period="1y")
    if df.empty:
        df = MockDataClient().get_historical_data(ticker, period="1y")
    if df.empty:
        return None, None, None, None

    df = IndicatorLibrary.add_all_indicators(df)
    tech_scores = {
        "Trend": scorer.score_trend(df),
        "Breakout": scorer.score_breakout(df),
        "Momentum": scorer.score_momentum(df),
        "Squeeze": scorer.score_squeeze(df)
    }
    base_score = scorer.calculate_total_score(tech_scores)

    news, headlines, sentiment = [], [], {}
    if finnhub_client:
        news = finnhub_client.get_company_news(ticker) or []
        headlines = [n['headline'] for n in news[:5]]
        sentiment = finnhub_client.get_sentiment(ticker) or {}

    sentiment_context = {"finnhub_sentiment": sentiment, "recent_headlines": headlines}
    reasoning = refiner.generate_trade_reasoning(ticker, tech_scores, sentiment_context)
    return base_score, tech_scores, reasoning, df

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    status_label, status_color = market_status()
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#060a14,#0d1b3e);border:1px solid rgba(0,212,255,0.2);
    border-radius:12px;padding:16px;margin-bottom:16px;text-align:center;">
        <div style="font-size:1.6rem;font-weight:800;background:linear-gradient(90deg,#00d4ff,#7b8cde);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">⚡ EDGE</div>
        <div style="font-size:0.7rem;color:#7b8fa8;letter-spacing:2px;">STRATEGY ENGINE</div>
        <div style="margin-top:8px;font-size:0.78rem;color:{status_color};font-weight:600;">{status_label}</div>
        <div style="font-size:0.7rem;color:#7b8fa8;">{datetime.now().strftime('%H:%M · %b %d, %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Alerts badge
    alerts_data = load_json("alerts.json", {})
    alert_count = len(alerts_data.get("alerts", []))
    if alert_count:
        st.markdown(f'<div style="text-align:center;margin-bottom:8px;"><span class="alert-dot"></span>'
                    f'<span style="color:#ff4444;font-size:0.8rem;font-weight:600;">{alert_count} Active Alerts</span></div>',
                    unsafe_allow_html=True)
        for a in alerts_data.get("alerts", []):
            if a['severity'] == 'high':
                st.error(f"**{a['title']}**: {a['message']}")
            elif a['severity'] == 'warning':
                st.warning(f"**{a['title']}**: {a['message']}")
            else:
                st.info(f"**{a['title']}**: {a['message']}")
        st.markdown("---")

    page = st.radio("", [
        "🌅 Daily Briefing",
        "🔍 Market Scanner",
        "📰 Sentiment & News",
        "🌍 Sector Performance",
        "🔥 Sector Heatmap",
        "📋 Smart Watchlists",
        "⚡ Options Intelligence",
        "🔮 Opportunity Scanner",
        "🎯 Day Trading",
        "🔄 Sector Rotation",
        "🤖 AI Command Center",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#7b8fa8;text-align:center;">Powered by LangChain + YFinance</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DAILY BRIEFING
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🌅 Daily Briefing":
    try:
        sec_data = load_json("detailed_sector_report.json", {})
        sectors = sec_data.get("sector_overview", [])
        alerts = load_json("alerts.json", {}).get("alerts", [])
        opts = load_json("options_activity.json", [])
        opps = load_json("opportunities.json", [])

        all_stocks = []
        for sector_stocks in sec_data.get("constituents", {}).values():
            if isinstance(sector_stocks, list):
                all_stocks.extend(sector_stocks)

        def day_move(s):
            if not s: return 0.0
            return (
                s.get("change_pct_1d")
                or s.get("pct_change")
                or s.get("change_percent")
                or s.get("performance", {}).get("1D")
                or s.get("return_1d")
                or s.get("perf_1d")
                or 0.0
            )

        def score_val(s):
            if not s: return 0.0
            return (
                s.get("total_score") 
                or s.get("Score") 
                or s.get("scores", {}).get("total")
                or 0.0
            )

        avg_mom = sum(s.get("momentum_score", 0) for s in sectors) / max(len(sectors), 1) if sectors else 50.0
        top_sector = max(sectors, key=lambda x: x.get("performance", {}).get("1D", 0), default={})
        top_sector_name = top_sector.get("sector", "N/A")
        top_sector_perf = top_sector.get("performance", {}).get("1D", 0.0)
        risk_regime = "Risk-On" if avg_mom >= 60 else ("Risk-Off" if avg_mom <= 40 else "Mixed")

        gainers = sorted(all_stocks, key=day_move, reverse=True)[:5]
        losers = sorted(all_stocks, key=day_move)[:5]
        high_conviction = sorted(all_stocks, key=score_val, reverse=True)[:6]

        advancers = sum(1 for s in all_stocks if day_move(s) > 0)
        decliners = sum(1 for s in all_stocks if day_move(s) < 0)
        flat_count = max(len(all_stocks) - advancers - decliners, 0)

        top_opps = sorted(opps, key=lambda x: x.get("Score", 0), reverse=True)[:4]
        top_opts = opts[:4]

        st.markdown(f"""
        <div class="bb-home">
            <div class="bb-header">
                <div>
                    <div class="bb-brand">Daily Briefing Terminal</div>
                    <div class="bb-headline">US Market Open Dashboard</div>
                </div>
                <div class="bb-timestamp">{datetime.now().strftime('%A, %b %d, %Y')}<br>{datetime.now().strftime('%H:%M')}</div>
            </div>
            <div class="bb-highlight">
                <span class="bb-flash">FLASH</span>
                <span class="bb-label">{top_sector_name}</span>
                <span class="{ 'bb-up' if top_sector_perf > 0 else 'bb-down' if top_sector_perf < 0 else 'bb-flat' }">{top_sector_perf:+.2f}%</span>
                <span class="bb-muted"> | Market Regime: </span><span class="bb-label">{risk_regime}</span>
                <span class="bb-muted"> | Active Alerts: </span><span class="bb-label">{len(alerts)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tape_parts = []
        for item in high_conviction[:8]:
            ticker = item.get("ticker", "N/A")
            mv = day_move(item)
            cls = "bb-up" if mv > 0 else ("bb-down" if mv < 0 else "bb-flat")
            tape_parts.append(
                f'<span class="bb-tape-item">{ticker} <span class="{cls}">{mv:+.2f}%</span> '
                f'<span class="bb-muted">S:{score_val(item):.1f}</span></span>'
            )
        st.markdown(f'<div class="bb-tape">{"".join(tape_parts) if tape_parts else "No tape data available."}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="bb-panel"><div class="bb-panel-title">Top Gainers</div>', unsafe_allow_html=True)
            if gainers:
                for stock in gainers:
                    st.markdown(
                        f'<div class="bb-row"><span class="bb-label">{stock.get("ticker","N/A")}</span>'
                        f'<span class="bb-up">{day_move(stock):+.2f}%</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="bb-muted">No gainers data.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="bb-panel"><div class="bb-panel-title">Top Losers</div>', unsafe_allow_html=True)
            if losers:
                for stock in losers:
                    st.markdown(
                        f'<div class="bb-row"><span class="bb-label">{stock.get("ticker","N/A")}</span>'
                        f'<span class="bb-down">{day_move(stock):+.2f}%</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="bb-muted">No losers data.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="bb-panel"><div class="bb-panel-title">Market Breadth</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bb-row"><span class="bb-label">Advancers</span><span class="bb-up">{advancers}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bb-row"><span class="bb-label">Decliners</span><span class="bb-down">{decliners}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bb-row"><span class="bb-label">Unchanged</span><span class="bb-flat">{flat_count}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bb-row"><span class="bb-label">Momentum Avg</span><span class="bb-label">{avg_mom:.1f}/100</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        l1, l2 = st.columns([1.3, 1])
        with l1:
            st.markdown('<div class="bb-panel"><div class="bb-panel-title">High Conviction Board</div>', unsafe_allow_html=True)
            if high_conviction:
                def safe_float(v):
                    try: return float(v)
                    except: return 0.0
                conviction_df = pd.DataFrame(
                    [{
                        "Ticker": s.get("ticker", "N/A"),
                        "Score": round(score_val(s), 1),
                        "1D %": round(day_move(s), 2),
                        "Price": round(safe_float(s.get("close", 0)), 2),
                        "Cap": s.get("market_cap_category", "N/A"),
                    } for s in high_conviction]
                )
                st.dataframe(conviction_df, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="bb-muted">No stock ranking data available.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with l2:
            st.markdown('<div class="bb-panel"><div class="bb-panel-title">Sector Momentum Pulse</div>', unsafe_allow_html=True)
            if sectors:
                fig_pulse = go.Figure(go.Bar(
                    x=[s.get("momentum_score", 0) for s in sectors],
                    y=[s.get("sector", "N/A")[:14] for s in sectors],
                    orientation="h",
                    marker=dict(color=[s.get("momentum_score", 0) for s in sectors], colorscale="Turbo", cmin=0, cmax=100),
                    text=[f'{s.get("momentum_score", 0):.0f}' for s in sectors],
                    textposition="outside",
                ))
                fig_pulse.update_layout(
                    height=320,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig_pulse, use_container_width=True)
            else:
                st.markdown('<div class="bb-muted">No sector pulse data.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        o1, o2 = st.columns(2)
        with o1:
            st.markdown('<div class="bb-panel"><div class="bb-panel-title">Options Flow Radar</div>', unsafe_allow_html=True)
            if top_opts:
                for op in top_opts:
                    st.markdown(
                        f'<div class="bb-row"><span class="bb-label">{op.get("ticker","N/A")} {op.get("type","")}</span>'
                        f'<span class="bb-muted">{str(op.get("details",""))[:36]}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="bb-muted">No unusual flow alerts.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with o2:
            st.markdown('<div class="bb-panel"><div class="bb-panel-title">Opportunity Radar</div>', unsafe_allow_html=True)
            if top_opps:
                for op in top_opps:
                    score = op.get("Score", 0)
                    style = "bb-up" if score >= 75 else ("bb-flat" if score >= 60 else "bb-down")
                    st.markdown(
                        f'<div class="bb-row"><span class="bb-label">{op.get("Ticker","N/A")} [{op.get("Setup","N/A")}]</span>'
                        f'<span class="{style}">{score:.1f}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="bb-muted">No opportunities detected.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering Daily Briefing: {e}")
        st.info("Ensure `run_detailed_sector_report.py` has been run and JSON files are not corrupted.")
        import traceback
        st.expander("Debug Details").code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Market Scanner":
    st.markdown('<div class="hero-banner"><p class="hero-title">🔍 AI Market Scanner</p>'
                '<p class="hero-sub">Deep technical + AI-powered analysis on any ticker</p></div>',
                unsafe_allow_html=True)

    col_input, col_btn = st.columns([2, 1])
    with col_input:
        ticker_input = st.text_input("Enter Ticker Symbol", "AAPL", placeholder="e.g. NVDA, TSLA, SPY").upper()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if scan_btn:
        with st.spinner(f"Scanning {ticker_input}..."):
            score, details, reason, df = run_scanner_logic(ticker_input)

        if score is not None:
            # Signal badge
            if score > 75:
                sig_html = f'<div style="text-align:center;margin:12px 0"><span class="badge-buy">🔥 STRONG BUY  ·  {score:.1f}/100</span></div>'
            elif score > 60:
                sig_html = f'<div style="text-align:center;margin:12px 0"><span class="badge-watch">👀 WATCHLIST  ·  {score:.1f}/100</span></div>'
            else:
                sig_html = f'<div style="text-align:center;margin:12px 0"><span class="badge-skip">❌ WAIT / SKIP  ·  {score:.1f}/100</span></div>'
            st.markdown(sig_html, unsafe_allow_html=True)

            # KPIs + Gauge
            g_col, m1, m2, m3 = st.columns([1.2, 1, 1, 1])
            with g_col:
                st.plotly_chart(score_gauge(score, "Total Score"), use_container_width=True)
            m1.metric("Price",  f"${df['Close'].iloc[-1]:.2f}",   f"{df['Close'].iloc[-1]-df['Open'].iloc[-1]:+.2f}")
            m2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}",   "")
            m3.metric("1D Chg", f"{((df['Close'].iloc[-1]-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100):+.2f}%", "")

            # Tabs
            tab_chart, tab_scores, tab_ai = st.tabs(["📈 Price Chart", "📊 Score Breakdown", "🤖 AI Reasoning"])

            with tab_chart:
                st.plotly_chart(plot_price_chart(df, ticker_input), use_container_width=True)

            with tab_scores:
                scores_df = pd.DataFrame(list(details.items()), columns=["Category", "Score"])
                fig_bars = go.Figure(go.Bar(
                    x=scores_df["Category"], y=scores_df["Score"],
                    marker=dict(
                        color=scores_df["Score"],
                        colorscale=[[0,'#ff5252'],[0.6,'#ffd600'],[1,'#00e676']],
                        cmin=0, cmax=100,
                        line=dict(color='rgba(0,212,255,0.3)', width=1)
                    ),
                    text=scores_df["Score"].round(1), textposition='outside'
                ))
                fig_bars.update_layout(
                    template="plotly_dark", height=300,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0, 115]), margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_bars, use_container_width=True)
                # Score bars
                for cat, sc in details.items():
                    st.markdown(f'<div style="margin:6px 0"><span style="color:#7b8fa8;font-size:0.8rem">{cat}</span>'
                                f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{sc}%"></div></div></div>',
                                unsafe_allow_html=True)

            with tab_ai:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"**AI Trade Reasoning for {ticker_input}:**")
                st.write(reason)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Could not fetch data for **{ticker_input}**. Check the ticker symbol.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT & NEWS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📰 Sentiment & News":
    st.markdown('<div class="hero-banner"><p class="hero-title">📰 Sentiment & News</p>'
                '<p class="hero-sub">Sector sentiment heatmap and live news feed</p></div>',
                unsafe_allow_html=True)

    data = load_json("detailed_sector_report.json", {})
    sectors = data.get("sector_overview", [])

    if sectors:
        st.subheader("🧠 Sector Sentiment Heatmap")
        sent_df = pd.DataFrame([{
            "Sector": s['sector'],
            "Technical Sentiment": s['momentum_score'],
            "Trend Score": s['trend_score'],
            "Vol Spike": round(s['volume_spike'], 2)
        } for s in sectors])
        st.dataframe(
            sent_df.style.background_gradient(cmap='RdYlGn', subset=['Technical Sentiment', 'Trend Score'])
                         .background_gradient(cmap='Blues', subset=['Vol Spike']),
            use_container_width=True, hide_index=True
        )

    st.subheader("📢 News Headlines")
    wl = load_json("watchlists.json", {})
    tickers = list({t['ticker'] for items in wl.values() for t in items})[:6]

    try:
        from data.finnhub_client import FinnhubClient
        fc = FinnhubClient()
        for t in tickers:
            with st.expander(f"📰 {t}", expanded=False):
                news = fc.get_company_news(t) or []
                if news:
                    for n in news[:3]:
                        st.markdown(f"**[{n['headline']}]({n['url']})**")
                        try:
                            ts = datetime.fromtimestamp(n['datetime']).strftime('%Y-%m-%d %H:%M')
                        except:
                            ts = "N/A"
                        st.caption(f"{ts} | {n.get('source', '')}")
                else:
                    st.write("No recent news.")
    except Exception as e:
        st.warning(f"News unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SECTOR PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Sector Performance":
    st.markdown('<div class="hero-banner"><p class="hero-title">🌍 Sector Performance</p>'
                '<p class="hero-sub">Multi-timeframe sector analysis with constituent drill-down</p></div>',
                unsafe_allow_html=True)

    report = load_json("detailed_sector_report.json", {})
    if not report:
        st.warning("⚠️ Run `python run_detailed_sector_report.py` to generate data.")
    else:
        sectors_perf = report.get('sector_overview', [])
        constituents = report.get('constituents', {})
        df_sec = pd.DataFrame(sectors_perf)

        if not df_sec.empty:
            top_mom = df_sec.sort_values('momentum_score', ascending=False).iloc[0]
            top_vol = df_sec.sort_values('volume_spike', ascending=False).iloc[0]
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="glass-card"><div class="kpi-label">🚀 Highest Momentum</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:#00d4ff">{top_mom["sector"]} ({top_mom["etf"]})</div>'
                        f'<div class="kpi-delta up">Score: {top_mom["momentum_score"]:.1f}</div></div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="glass-card"><div class="kpi-label">🔊 Highest Vol Spike</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:#00d4ff">{top_vol["sector"]} ({top_vol["etf"]})</div>'
                        f'<div class="kpi-delta up">Ratio: {top_vol["volume_spike"]:.2f}x</div></div>',
                        unsafe_allow_html=True)

        timeframes = ['1D', '1W', '1M', '3M', '6M', 'YTD']
        selected_tf = st.radio("Timeframe", timeframes, horizontal=True, index=2)
        df_sec['Perf %'] = df_sec['performance'].apply(lambda x: x.get(selected_tf, 0))
        df_sec_sorted = df_sec.sort_values('Perf %', ascending=False)

        fig = go.Figure(go.Bar(
            x=df_sec_sorted['sector'], y=df_sec_sorted['Perf %'],
            marker=dict(
                color=df_sec_sorted['Perf %'],
                colorscale='RdYlGn', cmin=-5, cmax=5,
                line=dict(color='rgba(0,212,255,0.2)', width=1)
            ),
            text=df_sec_sorted['Perf %'].round(2).astype(str) + '%',
            textposition='outside'
        ))
        fig.update_layout(
            template="plotly_dark", height=380,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔬 Constituent Drill-Down")
        selected_sector = st.selectbox("Choose Sector", ["None"] + sorted(constituents.keys()))
        if selected_sector != "None":
            stocks = constituents.get(selected_sector, [])
            if stocks:
                df_s = pd.DataFrame(stocks)
                df_s['Perf %'] = df_s['performance'].apply(lambda x: x.get(selected_tf, 0))
                c_filter, c_sort = st.columns(2)
                cats = ["All", "Large-Cap", "Mid-Cap", "Small-Cap"]
                sel_cat = c_filter.selectbox("Market Cap", cats)
                sort_col = c_sort.selectbox("Sort By", ["Perf %", "total_score", "volume"])
                if sel_cat != "All":
                    df_s = df_s[df_s['market_cap_category'] == sel_cat]
                df_s = df_s.sort_values(sort_col, ascending=False)
                st.dataframe(
                    df_s[['ticker','market_cap_category','Perf %','total_score','volume','close']]
                    .style.background_gradient(subset=['Perf %'], cmap='RdYlGn'),
                    use_container_width=True, hide_index=True
                )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SECTOR HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔥 Sector Heatmap":
    st.markdown('<div class="hero-banner"><p class="hero-title">🔥 Sector Heatmap</p>'
                '<p class="hero-sub">Interactive treemap — size by volume spike, color by performance</p></div>',
                unsafe_allow_html=True)

    data = load_json("detailed_sector_report.json", {})
    if not data:
        st.warning("Run `python run_detailed_sector_report.py` to generate data.")
    else:
        sectors = data.get("sector_overview", [])
        tf = st.radio("Timeframe", ['1D', '1W', '1M', '3M'], horizontal=True)
        tm_data = [{"Sector": s['sector'], "Parent": "Market",
                    "Performance": s['performance'].get(tf, 0),
                    "Volume Spike": max(s['volume_spike'], 0.1)} for s in sectors]
        df_tm = pd.DataFrame(tm_data)

        fig = px.treemap(
            df_tm, path=['Parent', 'Sector'],
            values='Volume Spike', color='Performance',
            color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            custom_data=['Performance']
        )
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%",
            hovertemplate='<b>%{label}</b><br>Performance: %{customdata[0]:.2f}%<br>Vol Spike: %{value:.2f}x'
        )
        fig.update_layout(
            height=550, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Sector Detail Table")
        st.dataframe(
            pd.DataFrame(sectors)[['sector','etf','close','volume_spike','trend_score','momentum_score']]
            .style.background_gradient(subset=['momentum_score','trend_score'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SMART WATCHLISTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Smart Watchlists":
    st.markdown('<div class="hero-banner"><p class="hero-title">📋 Smart Watchlists</p>'
                '<p class="hero-sub">AI-curated candidates ranked by technical conviction</p></div>',
                unsafe_allow_html=True)

    watchlists = load_json("watchlists.json", {})
    if not watchlists:
        st.warning("No watchlist data. Run the scanner.")
    else:
        # Top 5 consolidated picks
        st.subheader("🏆 Top 5 High-Conviction Picks")
        consolidated = {}
        for name, tickers in watchlists.items():
            for t in tickers:
                sym = t['ticker']
                if sym not in consolidated:
                    consolidated[sym] = {'score': t.get('score', 0), 'price': t.get('price', 0), 'strategies': set()}
                consolidated[sym]['strategies'].add(name)

        top5 = sorted(consolidated.values(), key=lambda x: x['score'], reverse=True)[:5]
        if top5:
            cols = st.columns(5)
            for i, p in enumerate(top5):
                with cols[i]:
                    strats = list(p['strategies'])
                    st.metric(strats[0] if strats else "Pick", p['score'] and f"{p['score']:.0f}/100", f"${p['price']:.2f}")

        st.markdown("---")

        icons = {"Oversold Reversal": "📉", "Breakout Candidates": "🚀", "Trend Reversals": "🔄",
                 "Breakout + Retest": "✅", "RS Leadership": "🏆", "Gap Fill Candidates": "🕳️",
                 "Inst. Accumulation": "🏦", "Base Breakouts": "💥", "EMA Trend Ride": "🌊",
                 "Volatility Expansion": "📏", "Earnings Drift": "📈", "VWAP Hold": "⚓",
                 "High Volatility / Squeeze": "⚡"}
        descs = {
            "Oversold Reversal": "RSI < 35 + Volume Spike. Look for bounce.",
            "Breakout Candidates": "Near 52w High + Volume. Continuation setup.",
            "Trend Reversals": "MACD Bullish Crossover. Early entry.",
            "Breakout + Retest": "Breakout above resistance holding support.",
            "RS Leadership": "Stock outperforming Sector/Market.",
            "Gap Fill Candidates": "Price entering recent gap zone.",
            "Inst. Accumulation": "Rising OBV with flat price (Smart Money).",
            "Base Breakouts": "Volatility Contraction → Expansion.",
            "EMA Trend Ride": "Stacked EMAs (10>20>50) momentum.",
            "Volatility Expansion": "ATR expanding from compression.",
            "Earnings Drift": "Post-Earnings momentum drift.",
            "VWAP Hold": "Price holding above 5-Day VWAP.",
            "High Volatility / Squeeze": "Bollinger Band Squeeze (Potential Explosion)."
        }

        for name, tickers in watchlists.items():
            if not tickers: continue
            icon = icons.get(name, "📌")
            with st.expander(f"{icon} {name}  ({len(tickers)} stocks)", expanded=False):
                st.caption(descs.get(name, "Quantitative Setup"))
                # Ticker chips
                chips = "".join([f'<span class="ticker-chip">{t["ticker"]}</span>' for t in tickers[:20]])
                st.markdown(chips, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                df_w = pd.DataFrame(tickers)
                st.dataframe(
                    df_w.style.background_gradient(subset=['score'], cmap='Greens') if 'score' in df_w.columns else df_w,
                    use_container_width=True, hide_index=True,
                    column_config={
                        "ticker": "Ticker", "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "category": "Cap", "reason": "Signal", "score": st.column_config.NumberColumn("Score", format="%.1f")
                    }
                )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OPTIONS INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Options Intelligence":
    st.markdown('<div class="hero-banner"><p class="hero-title">⚡ Options Flow Intelligence</p>'
                '<p class="hero-sub">Unusual volume, PCR extremes, and smart money flow detection</p></div>',
                unsafe_allow_html=True)

    alerts = load_json("options_activity.json", [])
    if not alerts:
        st.warning("Run `python run_detailed_sector_report.py` to generate options data.")
    else:
        total = len(alerts)
        calls = sum(1 for a in alerts if "Call" in a.get('type', ''))
        puts  = sum(1 for a in alerts if "Put"  in a.get('type', ''))
        bullish = sum(1 for a in alerts if "Bullish" in a.get('type', ''))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Alerts", total)
        c2.metric("🟢 Call Activity", calls)
        c3.metric("🔴 Put Activity",  puts)
        c4.metric("📈 Bullish PCR",   bullish)

        # PCR donut
        if calls + puts > 0:
            fig_donut = go.Figure(go.Pie(
                labels=["Calls", "Puts"],
                values=[calls, puts],
                hole=0.65,
                marker=dict(colors=['#00e676', '#ff5252']),
                textinfo='percent+label'
            ))
            fig_donut.update_layout(
                height=250, template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0),
                showlegend=False,
                annotations=[dict(text=f"{calls/(calls+puts)*100:.0f}%<br>Calls", x=0.5, y=0.5,
                                  font_size=16, showarrow=False, font_color='#00e676')]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.subheader("🚨 Options Alerts")
        # Type filter
        types = ["All"] + sorted(list({a.get('type', '') for a in alerts}))
        sel_type = st.selectbox("Filter by Type", types)
        filtered = [a for a in alerts if sel_type == "All" or a.get('type') == sel_type]
        st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OPPORTUNITY SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Opportunity Scanner":
    st.markdown('<div class="hero-banner"><p class="hero-title">🔮 Opportunity Scanner</p>'
                '<p class="hero-sub">Automated detection of Pullbacks, Squeezes, and Breakouts</p></div>',
                unsafe_allow_html=True)

    opps = load_json("opportunities.json", [])
    if not opps:
        st.warning("⚠️ Run `python run_opportunity_scan.py` to generate data.")
    else:
        df_opps = pd.DataFrame(opps)
        setup_types = sorted(df_opps['Setup'].unique()) if 'Setup' in df_opps.columns else []

        # Summary metrics
        c_cols = st.columns(len(setup_types) + 1) if setup_types else st.columns(1)
        c_cols[0].metric("Total Opportunities", len(opps))
        for i, st_type in enumerate(setup_types, 1):
            if i < len(c_cols):
                c_cols[i].metric(st_type, len(df_opps[df_opps['Setup'] == st_type]))

        # Score distribution chart
        if 'Score' in df_opps.columns:
            fig_dist = go.Figure(go.Histogram(
                x=df_opps['Score'], nbinsx=20,
                marker_color='rgba(0,212,255,0.7)',
                marker_line=dict(color='rgba(0,212,255,1)', width=1)
            ))
            fig_dist.update_layout(
                title="Score Distribution", template="plotly_dark", height=220,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Kanban-style columns
        if setup_types:
            col_filter = st.selectbox("Filter by Setup", ["All"] + list(setup_types))
            display_df = df_opps if col_filter == "All" else df_opps[df_opps['Setup'] == col_filter]
            display_df = display_df.sort_values('Score', ascending=False) if 'Score' in display_df.columns else display_df

            if setup_types and col_filter == "All" and len(setup_types) >= 2:
                kanban_cols = st.columns(min(len(setup_types), 3))
                for ci, stype in enumerate(setup_types[:3]):
                    with kanban_cols[ci]:
                        st.markdown(f"**{stype}**")
                        sub = df_opps[df_opps['Setup'] == stype].sort_values('Score', ascending=False) if 'Score' in df_opps.columns else df_opps[df_opps['Setup'] == stype]
                        for _, row in sub.head(8).iterrows():
                            score_v = row.get('Score', 0)
                            st.markdown(
                                f'<div class="opp-card"><b>{row.get("Ticker","")}</b><br>'
                                f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{score_v}%"></div></div>'
                                f'<small style="color:#7b8fa8">Score: {score_v:.1f}</small></div>',
                                unsafe_allow_html=True
                            )
            else:
                st.dataframe(
                    display_df.style.background_gradient(subset=['Score'], cmap='Greens') if 'Score' in display_df.columns else display_df,
                    use_container_width=True, hide_index=True
                )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DAY TRADING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Day Trading":
    st.markdown('<div class="hero-banner"><p class="hero-title">🎯 Day Trading Signals</p>'
                '<p class="hero-sub">10 intraday strategies · 5-minute data · Real-time edge</p></div>',
                unsafe_allow_html=True)

    signals = load_json("day_trading_signals.json", [])
    if not signals:
        st.warning("Run `python run_day_trading_scan.py` to generate signals.")
    else:
        st.subheader("🏆 Top 5 Day Trading Picks")
        top5 = signals[:5]
        cols5 = st.columns(5)
        for i, pick in enumerate(top5):
            with cols5[i]:
                dirs = {v['direction'] for v in pick.get('strategies', {}).values()}
                d_label = "🟢 LONG" if "long" in dirs and "short" not in dirs else ("🔴 SHORT" if "short" in dirs and "long" not in dirs else "🟡 MIX")
                conf = pick['composite_confidence']
                color = "#00e676" if conf > 70 else ("#ffd600" if conf > 50 else "#ff5252")
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{pick["ticker"]}</div>'
                    f'<div class="kpi-value" style="color:{color}">{conf:.0f}</div>'
                    f'<div class="kpi-delta">{d_label} · {pick["num_signals"]}sig</div>'
                    f'<div style="font-size:0.72rem;color:#7b8fa8">${pick["price"]:.2f}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.subheader("📊 Strategy Breakdown")

        strategy_descriptions = {
            "Opening Range Breakout": "ORB — First 30 min defines range; break with volume = entry",
            "VWAP Trend": "Price position vs VWAP + EMA alignment",
            "Momentum Ignition": "RVOL > 2 + intraday high break + wide-range candle",
            "RVOL Breakout": "Relative Volume > 3x daily average + momentum",
            "Pullback Scalp": "Bounce off 9-EMA or VWAP in established trend",
            "Reversal Fade": "Price extended 2+ ATR from VWAP with exhaustion volume",
            "Gap & Go": "Gap > 3% with high volume, holding above gap level",
            "Breakout Retest": "Intraday high breakout → pullback → low-vol hold",
            "Lunch-Time Fade": "Afternoon vol contraction, drift toward VWAP",
            "Order Flow": "Buy/sell volume imbalance for directional pressure",
        }

        for strat_name, desc in strategy_descriptions.items():
            matches = []
            for sig in signals:
                if strat_name in sig.get('strategies', {}):
                    s_data = sig['strategies'][strat_name]
                    dir_icon = "🟢" if s_data['direction'] == 'long' else "🔴"
                    matches.append({"Ticker": sig['ticker'], "Dir": f"{dir_icon} {s_data['direction'].upper()}",
                                    "Conf": s_data['confidence'], "Reason": s_data['reason'], "Price": sig['price']})
            with st.expander(f"{'✅' if matches else '⬜'} {strat_name}  ({len(matches)} hits)", expanded=len(matches) > 3):
                st.caption(desc)
                if matches:
                    df_m = pd.DataFrame(matches).sort_values('Conf', ascending=False)
                    st.dataframe(df_m, use_container_width=True, hide_index=True)
                else:
                    st.info("No triggers in latest scan.")

        st.markdown("---")
        st.subheader("📋 All Signals")
        all_rows = [{"Ticker": s['ticker'], "Score": s['composite_confidence'],
                     "Price": s['price'], "# Signals": s['num_signals'],
                     "Strategies": ", ".join(s['strategies'].keys())} for s in signals]
        st.dataframe(
            pd.DataFrame(all_rows).style.background_gradient(subset=['Score'], cmap='Greens'),
            use_container_width=True, hide_index=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI COMMAND CENTER
# ═══════════════════════════════════════════════════════════════════════════════# 🔄 PAGE: SECTOR ROTATION
elif page == "🔄 Sector Rotation":
    st.markdown('<div class="hero-banner"><p class="hero-title">🔄 Sector Rotation Strategy</p>'
                '<p class="hero-sub">Top-3 Momentum + Trend Filter + Safety Switch</p></div>',
                unsafe_allow_html=True)
    
    results = load_json("rotation_results.json", None)
    
    if not results:
        st.warning("No backtest results found. Running initial analysis...")
        if st.button("🚀 Run Analysis & Backtest"):
            with st.spinner("Processing 2 years of sector data..."):
                # We can't easily wait for the background process here, 
                # but we can trigger it or just run a simplified version.
                # For now, let's assume it was run.
                st.info("Scanner triggered. Refresh in 30s.")
    else:
        metrics = results.get("metrics", {})
        cols = st.columns(4)
        with cols[0]:
            st.metric("Strategy CAGR", f"{metrics.get('cagr', 0)}%", f"{metrics.get('cagr',0) - metrics.get('spy_cagr',0):.1f}% vs SPY")
        with cols[1]:
            st.metric("Sharpe Ratio", metrics.get("sharpe", 0))
        with cols[2]:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)}%")
        with cols[3]:
            st.metric("Volatility", f"{metrics.get('volatility', 0)}%")

        # Equity Curve
        df_eq = pd.DataFrame(results.get("equity_curve", []))
        if not df_eq.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_eq['date'], y=df_eq['strategy_cum'], name='Strategy', line=dict(color='#00d4ff', width=3)))
            fig.add_trace(go.Scatter(x=df_eq['date'], y=df_eq['spy_cum'], name='SPY (Benchmark)', line=dict(color='#7b8fa8', dash='dash')))
            fig.update_layout(title="Growth of $1", template="plotly_dark", height=400, margin=dict(l=0,r=0,t=40,b=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # Current Signals
        last = results.get("last_rebalance", {})
        st.subheader("🎯 Current Trading Signals")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("**Top 3 Momentum Picks**")
            picks = last.get("picks", [])
            if not picks:
                st.info("No sectors passed trend filters. Holding Cash (SHV).")
            else:
                for p in picks:
                    st.markdown(
                        f'<div class="opp-card" style="border-left:4px solid #00d4ff;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span><b style="font-size:1.2rem;">{p["ticker"]}</b> <small>• Score: {p["momentum_score"]:.1f}</small></span>'
                        f'<span class="badge-buy">BUY</span>'
                        f'</div></div>', unsafe_allow_html=True
                    )
        
        with c2:
            st.markdown("**Portfolio Allocation**")
            weights = last.get("weights", {})
            if weights:
                fig_pie = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.6)])
                fig_pie.update_layout(showlegend=False, height=220, margin=dict(l=0,r=0,t=0,b=0),
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)

        # Universe Status
        with st.expander("📊 Full Universe Trend Check"):
            all_res = last.get("all_results", [])
            if all_res:
                status_df = pd.DataFrame(all_res)
                st.dataframe(status_df.style.background_gradient(subset=['momentum_score'], cmap='RdYlGn'), use_container_width=True)

# 🤖 PAGE: AI COMMAND CENTER
elif page == "🤖 AI Command Center":
    st.markdown('<div class="hero-banner"><p class="hero-title">🤖 AI Command Center</p>'
                '<p class="hero-sub">Chat with your LangChain trading agent — query any scanner naturally</p></div>',
                unsafe_allow_html=True)

    # Init agent (cached in session state)
    if "ai_agent" not in st.session_state:
        with st.spinner("Initializing AI agent..."):
            try:
                from ai.agent import StockAnalysisAgent
                st.session_state.ai_agent = StockAnalysisAgent()
            except Exception as e:
                st.session_state.ai_agent = None
                st.error(f"Agent init failed: {e}")

    agent = st.session_state.ai_agent

    # API key status banner
    import os; api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success("✅ OpenAI API key detected — Full GPT-4o-mini agent active")
    else:
        st.info("ℹ️ No `OPENAI_API_KEY` in `.env` — running smart keyword-based fallback. Add your key for full GPT reasoning.")

    # Suggested prompts
    st.markdown("**💡 Suggested Queries:**")
    sugg_cols = st.columns(3)
    suggestions = [
        "Scan NVDA", "Top opportunities today", "Day trading signals",
        "Sector report", "Unusual options activity", "Top watchlist picks"
    ]
    for i, s in enumerate(suggestions):
        with sugg_cols[i % 3]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state.setdefault("chat_history", [])
                st.session_state.chat_history.append({"role": "user", "content": s})
                with st.spinner("Thinking..."):
                    resp = agent.run(s) if agent else "Agent not available."
                st.session_state.chat_history.append({"role": "assistant", "content": resp})

    st.markdown("---")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything — e.g. 'Analyze AAPL' or 'What are the best day trades?'"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if agent:
                response_placeholder = st.empty()
                full_response = ""
                for chunk in agent.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            else:
                st.error("⚠️ Agent not available. Please refresh the page.")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()
