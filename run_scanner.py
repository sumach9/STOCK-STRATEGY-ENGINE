import pandas as pd
from typing import List
from data.yfinance_client import YFinanceClient
from data.finnhub_client import FinnhubClient
from data.mock_client import MockDataClient
from strategy.indicators import IndicatorLibrary
from strategy.scoring import StrategyScorer
from ai.refiner import AIRefiner

def run_scan(tickers: List[str]):
    """Scan a list of tickers and print technical scores with detailed breakdown."""
    yf_client = YFinanceClient()
    finnhub_client = FinnhubClient()
    mock_client = MockDataClient()
    scorer = StrategyScorer()
    refiner = AIRefiner()
    
    print("\n" + "="*80)
    print(f"{'Ticker':<10} | {'Total':<8} | {'Trend':<8} | {'Break':<8} | {'Mom.':<8} | {'Sqze':<8} | {'Status'}")
    print("-" * 80)
    
    for ticker in tickers:
        # 1. Fetch Data
        df = yf_client.get_historical_data(ticker, period="1y")
        if df.empty:
            print(f"Falling back to Mock data for {ticker}")
            df = mock_client.get_historical_data(ticker, period="1y")
            
        if df.empty:
            continue
            
        # 2. Add Indicators
        df = IndicatorLibrary.add_all_indicators(df)
        
        # 3. Calculate Scores
        tech_scores = {
            "trend": scorer.score_trend(df),
            "breakout": scorer.score_breakout(df),
            "momentum": scorer.score_momentum(df),
            "squeeze": scorer.score_squeeze(df)
        }
        
        # Total base score
        base_score = scorer.calculate_total_score(tech_scores)
        
        status = "🔥 BUY" if base_score > 75 else "👀 WATCH" if base_score > 60 else "SKIP"
        
        print(f"{ticker:<10} | {base_score:<8.1f} | {tech_scores['trend']:<8.1f} | {tech_scores['breakout']:<8.1f} | {tech_scores['momentum']:<8.1f} | {tech_scores['squeeze']:<8.1f} | {status}")
        
        # Determine the strongest category
        # Sort tech_scores by value * weight to see relative importance
        weighted_scores = {k: v * scorer.weights[k] for k, v in tech_scores.items()}
        strongest = max(weighted_scores, key=weighted_scores.get)
        
        if base_score > 75:
            print(f"   >>> Strongest factor: {strongest.upper()} ({tech_scores[strongest]:.1f}%)")
            
            # Fetch News & Sentiment from Finnhub
            news = finnhub_client.get_company_news(ticker)
            headlines = [n['headline'] for n in news[:5]] if news else []
            sentiment = finnhub_client.get_sentiment(ticker)
            
            # Refine with AI
            sentiment_context = {
                "finnhub_sentiment": sentiment,
                "recent_headlines": headlines
            }
            reasoning = refiner.generate_trade_reasoning(ticker, tech_scores, sentiment_context)
            print(f"   Reasoning: {reasoning}\n")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Example tickers
    tickers_to_scan = ["AAPL", "TSLA", "MSFT", "NVDA", "AMD", "META", "GOOGL"]
    run_scan(tickers_to_scan)
