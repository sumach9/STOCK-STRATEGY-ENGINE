"""
AI Refiner — now powered by LangChain + ChatOpenAI.
Maintains the same public API as the original openai-based refiner.
"""
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class AIRefiner:
    """Uses LangChain + ChatOpenAI to refine trade signals with sentiment and reasoning."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.llm = None

        if self.api_key and LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model=self.model,
                    temperature=0.2,
                    api_key=self.api_key
                )
            except Exception as e:
                print(f"[AIRefiner] Could not init LLM: {e}")
        else:
            if not self.api_key:
                print("Warning: OpenAI API key not found. AI features will be mocked.")
            if not LANGCHAIN_AVAILABLE:
                print("Warning: LangChain not installed. AI features will be mocked.")

    def _run_chain(self, template: str, **kwargs) -> str:
        """Helper: run an LLMChain with the given prompt template and variables."""
        if not self.llm:
            return None
        try:
            prompt = PromptTemplate.from_template(template)
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(**kwargs)
        except Exception as e:
            print(f"[AIRefiner] Chain error: {e}")
            return None

    def get_news_sentiment(self, ticker: str, headlines: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of news headlines."""
        template = (
            "Analyze the sentiment of the following news headlines for {ticker}.\n"
            "Provide a sentiment score between 0 (very bearish) and 1 (very bullish).\n"
            "Headlines:\n{headlines}\n\n"
            "Return ONLY valid JSON: {{\"score\": float, \"summary\": string}}"
        )
        raw = self._run_chain(template, ticker=ticker, headlines="\n".join(headlines))
        if raw:
            try:
                result = json.loads(raw.strip())
                return {
                    "sentiment_score": result.get("score", 0.5),
                    "reasoning": result.get("summary", "")
                }
            except Exception:
                pass
        return {"sentiment_score": 0.5, "reasoning": "Mock sentiment (API key missing or parse error)"}

    def generate_trade_reasoning(self, ticker: str, technical_scores: Dict[str, float],
                                  sentiment: Dict[str, Any]) -> str:
        """Generate a natural language explanation for a trade signal."""
        template = (
            "Generate a concise, professional trade reasoning for {ticker}.\n"
            "Technical Scores: {scores}\n"
            "News Sentiment: {sentiment}\n\n"
            "In 3-4 sentences, explain the key bullish/bearish signals, "
            "risk factors, and whether this is actionable. Be direct and specific."
        )
        result = self._run_chain(
            template,
            ticker=ticker,
            scores=str(technical_scores),
            sentiment=str(sentiment)
        )
        if result:
            return result
        # Fallback mock
        total = sum(technical_scores.values()) / max(len(technical_scores), 1)
        direction = "bullish" if total > 60 else "bearish"
        return (
            f"**{ticker}** shows {direction} technicals with an average component score of {total:.0f}/100. "
            f"Key signals — Trend: {technical_scores.get('Trend', 0):.0f}, "
            f"Momentum: {technical_scores.get('Momentum', 0):.0f}, "
            f"Breakout: {technical_scores.get('Breakout', 0):.0f}. "
            f"Add an OpenAI API key for detailed AI reasoning."
        )

    def get_sector_report(self, sector_name: str, etf_ticker: str, recent_news: List[str]) -> str:
        """Generate a high-level sector sentiment report."""
        template = (
            "Provide a concise market sentiment report for the {sector} sector (ETF: {etf}).\n"
            "Recent News:\n{news}\n\n"
            "Summarize current sentiment (Bullish/Bearish/Neutral) and highlight 1-2 key drivers or risks in 2-3 sentences."
        )
        result = self._run_chain(
            template,
            sector=sector_name,
            etf=etf_ticker,
            news="\n".join(recent_news[:5])
        )
        if result:
            return result
        return f"**{sector_name} ({etf_ticker})**: Neutral outlook. Add OpenAI API key for real-time AI sector analysis."


if __name__ == "__main__":
    refiner = AIRefiner()
    sentiment = refiner.get_news_sentiment("AAPL", ["Apple reports record earnings", "iPhone sales slump in China"])
    print(sentiment)
    reasoning = refiner.generate_trade_reasoning("AAPL", {"Trend": 100, "Momentum": 85, "Breakout": 50, "Squeeze": 0}, sentiment)
    print(reasoning)
