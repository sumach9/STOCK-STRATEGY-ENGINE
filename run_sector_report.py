import pandas as pd
import time
from data.yfinance_client import YFinanceClient
from data.mock_client import MockDataClient
from data.sector_constituents import SECTOR_ETFS
from strategy.indicators import IndicatorLibrary
from strategy.scoring import StrategyScorer
from ai.refiner import AIRefiner

from data.finnhub_client import FinnhubClient

def run_sector_sentiment_report():
    """Analyze all major sectors to find which are strongest/weakest."""
    yf_client = YFinanceClient()
    finnhub_client = FinnhubClient()
    mock_client = MockDataClient()
    scorer = StrategyScorer()
    refiner = AIRefiner()
    
    print("\n" + "="*90)
    print(f"{'Sector':<25} | {'ETF':<8} | {'Score':<8} | {'Trend':<8} | {'Sentiment Summary'}")
    print("-" * 90)
    
    sector_rankings = []
    
    for sector_name, etf_ticker in SECTOR_ETFS.items():
        # 1. Fetch Data for the Sector ETF
        df = yf_client.get_historical_data(etf_ticker, period="1y")
        if df.empty:
            df = mock_client.get_historical_data(etf_ticker, period="1y")
            
        # 2. Add Indicators
        df = IndicatorLibrary.add_all_indicators(df)
        
        # 3. Calculate Scores
        tech_scores = {
            "trend": scorer.score_trend(df),
            "breakout": scorer.score_breakout(df),
            "momentum": scorer.score_momentum(df),
            "squeeze": scorer.score_squeeze(df)
        }
        
        total_score = scorer.calculate_total_score(tech_scores)
        
        # 4. Get AI Sentiment Summary
        # Use Finnhub to get real news for the ETF
        news = finnhub_client.get_company_news(etf_ticker)
        if news:
            news_headlines = [n['headline'] for n in news[:5]]
        else:
            # Fallback if no news found or key missing
            news_headlines = [f"{sector_name} sector reporting season begins", f"Market volatility impacts {etf_ticker}"]
            
        sentiment_summary = refiner.get_sector_report(sector_name, etf_ticker, news_headlines)
        # Limit summary length for table
        summary_short = sentiment_summary.replace('\n', ' ')[:45] + "..."
        
        print(f"{sector_name:<25} | {etf_ticker:<8} | {total_score:<8.1f} | {tech_scores['trend']:<8.1f} | {summary_short}")
        
        sector_rankings.append({
            "sector": sector_name,
            "etf": etf_ticker,
            "score": total_score,
            "summary": sentiment_summary
        })
        # Small delay to avoid hitting rate limits
        time.sleep(0.2)
        
    # Sort and show top 3
    top_3 = sorted(sector_rankings, key=lambda x: x['score'], reverse=True)[:3]
    
    print("-" * 90)
    print("\n🚀 TOP 3 STRONGEST SECTORS:")
    for i, s in enumerate(top_3):
        print(f"{i+1}. {s['sector'].upper()} ({s['etf']}) - Score: {s['score']:.1f}%")
        print(f"   Insight: {s['summary']}\n")
    print("="*90 + "\n")
    
    # Save to JSON for Dashboard
    import json
    with open("sector_report.json", "w") as f:
        json.dump(sector_rankings, f, indent=4)
    print("Saved sector report to sector_report.json")

if __name__ == "__main__":
    run_sector_sentiment_report()
