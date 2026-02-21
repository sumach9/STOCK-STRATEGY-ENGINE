"""
LangChain-powered Stock Analysis Agent.
Uses ChatOpenAI with tool-calling (function calling) for agentic dispatch.
Falls back to smart keyword routing when no API key is present.
Compatible with LangChain >=0.2 / 1.x
"""
import os
import json
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── LangChain imports — graceful fallback ─────────────────────────────────────
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ── Tool implementations ──────────────────────────────────────────────────────

def _load_json(path: str, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default if default is not None else {}


def _scan_ticker(ticker: str) -> str:
    """Full technical scan of a stock ticker."""
    ticker = ticker.strip().upper()
    try:
        from data.yfinance_client import YFinanceClient
        from data.mock_client import MockDataClient
        from strategy.indicators import IndicatorLibrary
        from strategy.scoring import StrategyScorer

        df = YFinanceClient().get_historical_data(ticker, period="1y")
        if df.empty:
            df = MockDataClient().get_historical_data(ticker, period="1y")
        if df.empty:
            return f"❌ No data found for {ticker}."

        df = IndicatorLibrary.add_all_indicators(df)
        scorer = StrategyScorer()
        scores = {
            "Trend": scorer.score_trend(df),
            "Breakout": scorer.score_breakout(df),
            "Momentum": scorer.score_momentum(df),
            "Squeeze": scorer.score_squeeze(df),
        }
        total = scorer.calculate_total_score(scores)
        signal = "🔥 STRONG BUY" if total > 75 else ("👀 WATCHLIST" if total > 60 else "❌ WAIT/SKIP")
        price = df['Close'].iloc[-1]
        vol   = df['Volume'].iloc[-1]
        return (
            f"**{ticker}** — {signal}  (Score: {total:.1f}/100)\n"
            f"Price: ${price:.2f} | Volume: {int(vol):,}\n"
            f"Trend: {scores['Trend']:.0f} | Breakout: {scores['Breakout']:.0f} | "
            f"Momentum: {scores['Momentum']:.0f} | Squeeze: {scores['Squeeze']:.0f}"
        )
    except Exception as e:
        return f"Error scanning {ticker}: {e}"


def _top_opportunities(_: str = "") -> str:
    opps = _load_json("opportunities.json", [])
    if not opps:
        return "No opportunity data. Run `python run_opportunity_scan.py`."
    top = sorted(opps, key=lambda x: x.get("Score", 0), reverse=True)[:10]
    lines = [f"• **{o.get('Ticker','')}** [{o.get('Setup','')}] — Score {o.get('Score',0):.1f}" for o in top]
    return "**Top Opportunities:**\n" + "\n".join(lines)


def _top_day_trades(_: str = "") -> str:
    sigs = _load_json("day_trading_signals.json", [])
    if not sigs:
        return "No day trading signals. Run `python run_day_trading_scan.py`."
    lines = []
    for s in sigs[:8]:
        dirs = {v['direction'] for v in s.get('strategies', {}).values()}
        d = "🟢 LONG" if "long" in dirs and "short" not in dirs else "🔴 SHORT"
        lines.append(f"• **{s['ticker']}** — {s['composite_confidence']:.0f}/100 | {d} | ${s['price']:.2f}")
    return "**Top Day Trades:**\n" + "\n".join(lines)


def _sector_report(_: str = "") -> str:
    data = _load_json("detailed_sector_report.json", {})
    sectors = data.get("sector_overview", [])
    if not sectors:
        return "No sector data. Run `python run_detailed_sector_report.py`."
    lines = []
    for s in sorted(sectors, key=lambda x: x.get("momentum_score", 0), reverse=True)[:8]:
        p1d = s.get("performance", {}).get("1D", 0)
        lines.append(f"• **{s['sector']}** ({s.get('etf','')}) — Momentum: {s['momentum_score']:.0f} | 1D: {p1d:+.2f}%")
    return "**Sector Overview:**\n" + "\n".join(lines)


def _options_activity(_: str = "") -> str:
    alerts = _load_json("options_activity.json", [])
    if not alerts:
        return "No options activity data found."
    lines = [f"• **{a['ticker']}** [{a.get('type','')}] — {a.get('details','')}" for a in alerts[:10]]
    return "**Unusual Options Activity:**\n" + "\n".join(lines)


def _top_watchlists(_: str = "") -> str:
    wl = _load_json("watchlists.json", {})
    if not wl:
        return "No watchlist data."
    consolidated = {}
    for name, tickers in wl.items():
        for t in tickers:
            sym = t["ticker"]
            if sym not in consolidated:
                consolidated[sym] = {"score": t.get("score", 0), "strategies": set()}
            consolidated[sym]["strategies"].add(name)
    top = sorted(consolidated.items(), key=lambda x: x[1]["score"], reverse=True)[:10]
    lines = [
        f"• **{sym}** — Score {info['score']:.1f} | {', '.join(list(info['strategies'])[:2])}"
        for sym, info in top
    ]
    return "**Top Watchlist Picks:**\n" + "\n".join(lines)


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "ScanTicker":       (_scan_ticker,       "Scan a specific stock ticker with full technical analysis"),
    "TopOpportunities": (_top_opportunities,  "Get the top opportunity setups from the latest scan"),
    "TopDayTrades":     (_top_day_trades,     "Get the best intraday day trading signals"),
    "SectorReport":     (_sector_report,      "Get current sector performance and momentum overview"),
    "OptionsActivity":  (_options_activity,   "Get unusual options activity alerts"),
    "TopWatchlists":    (_top_watchlists,     "Get the top AI-generated watchlist picks"),
}

SYSTEM_PROMPT = """You are an expert AI stock trading assistant with access to real-time market scanner tools.
You help traders find opportunities, analyze tickers, and understand market trends.
Be concise, data-driven, and actionable. Respond in structured markdown.

Available tools: ScanTicker(ticker), TopOpportunities(), TopDayTrades(), SectorReport(), OptionsActivity(), TopWatchlists()

To use a tool, say: TOOL: ToolName | Input
Then after getting the result, give your Final Answer.
"""


# ── Keyword fallback router ───────────────────────────────────────────────────
SKIP_WORDS = {"A","I","AT","OR","AN","ALL","THE","FOR","TOP","BEST","BUY","SCAN","ME","MY","WHAT","ARE","IS","GET"}

def _keyword_route(user_input: str) -> str:
    q = user_input.lower()
    tickers = [t for t in re.findall(r'\b([A-Z]{1,5})\b', user_input.upper()) if t not in SKIP_WORDS]

    if tickers and any(w in q for w in ["scan","analyze","analyse","check","look at","tell me about","what about"]):
        return _scan_ticker(tickers[0])
    elif any(w in q for w in ["opportunit","setup","coil","squeeze","pullback","breakout"]):
        return _top_opportunities()
    elif any(w in q for w in ["day trad","intraday","scalp","signal"]):
        return _top_day_trades()
    elif any(w in q for w in ["sector","industry","etf","market"]):
        return _sector_report()
    elif any(w in q for w in ["option","call","put","flow","iv "]):
        return _options_activity()
    elif any(w in q for w in ["watchlist","pick","watch"]):
        return _top_watchlists()
    elif tickers:
        return _scan_ticker(tickers[0])
    else:
        return (
            "🤖 **AI Command Center** — I can help with:\n\n"
            "• `Scan NVDA` — Full technical analysis\n"
            "• `Top opportunities` — Best setup candidates\n"
            "• `Day trading signals` — Today's intraday picks\n"
            "• `Sector report` — Current sector momentum\n"
            "• `Options activity` — Unusual flow alerts\n"
            "• `Top watchlist picks` — AI-curated candidates\n\n"
            "⚠️ Add `OPENAI_API_KEY` to `.env` for full GPT-4o-mini agent reasoning."
        )


# ── Main Agent Class ──────────────────────────────────────────────────────────
class StockAnalysisAgent:
    """
    LangChain-powered trading agent.
    Uses ChatOpenAI with tool-call parsing when API key is available,
    otherwise uses smart keyword routing.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model   = model
        self.history = []  # In-memory conversation history
        self.llm     = None

        if LANGCHAIN_AVAILABLE and self.api_key:
            try:
                self.llm = ChatOpenAI(
                    model=self.model,
                    temperature=0.1,
                    api_key=self.api_key
                )
            except Exception as e:
                print(f"[Agent] LLM init failed: {e}")

    @property
    def available(self) -> bool:
        return self.llm is not None

    def stream(self, user_input: str):
        """Streaming version of the run method for Streamlit."""
        if not self.available:
            yield _keyword_route(user_input)
            return

        self.history.append(HumanMessage(content=user_input))
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + self.history[-10:]

        try:
            # First pass to check for tool calls
            resp = self.llm.invoke(messages)
            content = resp.content if hasattr(resp, 'content') else str(resp)

            if "TOOL:" in content:
                yield "⏳ *Analyzing market data...*"
                tool_result = self._execute_tool_call(content)
                
                # Second pass with tool results
                follow_up = [SystemMessage(content=SYSTEM_PROMPT)] + self.history[-10:] + [
                    AIMessage(content=content),
                    HumanMessage(content=f"Tool result:\n{tool_result}\n\nNow give your final answer.")
                ]
                
                # Stream the final response
                full_resp = ""
                for chunk in self.llm.stream(follow_up):
                    chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_resp += chunk_content
                    yield chunk_content
                
                self.history.append(AIMessage(content=full_resp))
            else:
                # Stream the direct response
                full_resp = ""
                for chunk in self.llm.stream(messages):
                    chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_resp += chunk_content
                    yield chunk_content
                self.history.append(AIMessage(content=full_resp))

        except Exception as e:
            err_msg = f"\n\n*(LLM error: {e})*"
            yield _keyword_route(user_input) + err_msg

    def run(self, user_input: str) -> str:
        """Synchronous run method (non-streaming)."""
        if not self.available:
            return _keyword_route(user_input)

        # Simply collect the stream
        return "".join(list(self.stream(user_input)))

    def _execute_tool_call(self, llm_output: str) -> str:
        """Parse TOOL: ToolName | Input from LLM output and execute."""
        for line in llm_output.split('\n'):
            if line.strip().startswith("TOOL:"):
                parts = line.split("TOOL:", 1)[1].split("|")
                tool_name = parts[0].strip()
                tool_input = parts[1].strip() if len(parts) > 1 else ""
                if tool_name in TOOLS:
                    func, _ = TOOLS[tool_name]
                    try:
                        return func(tool_input)
                    except Exception as e:
                        return f"Tool error: {e}"
        return "No tool call found."

    def get_tools_summary(self) -> str:
        return "\n".join([f"• **{name}**: {desc}" for name, (_, desc) in TOOLS.items()])

    def clear_history(self):
        self.history = []
