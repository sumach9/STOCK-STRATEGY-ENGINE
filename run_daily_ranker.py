import pandas as pd
import time
from typing import List, Dict
from data.finnhub_client import FinnhubClient
from data.sector_constituents import get_sector_constituents
from ai.vader_sentiment import VaderSentimentAnalyzer

def run_daily_sentiment_ranking(limit_per_sector: int = 10, save_csv: bool = True):
    """
    Run daily sentiment ranking for all sectors.
    
    Args:
        limit_per_sector: Number of tickers to scan per sector.
        save_csv: Whether to save the results to a CSV file.
    """
    finnhub_client = FinnhubClient()
    vader = VaderSentimentAnalyzer()
    
    # Fetch dynamic sector list
    print("Fetching S&P 500 constituents from Wikipedia...")
    sector_constituents = get_sector_constituents()
    
    all_results = []
    
    print(f"Starting Daily Sentiment Scan... (Limit: {limit_per_sector} tickers/sector)\n")
    
    for sector, tickers in sector_constituents.items():
        print(f"Scanning {sector}...", end="", flush=True)
        sector_results = []
        
        # Limit tickers for speed/API limits
        scan_list = tickers[:limit_per_sector]
        
        for ticker in scan_list:
            # 1. Fetch News
            news_items = finnhub_client.get_company_news(ticker, days_back=3)
            headlines = [item['headline'] for item in news_items] if news_items else []
            
            # 2. Analyze Sentiment
            sentiment = vader.analyze_headlines(headlines)
            
            # 3. Store Result
            result = {
                "Ticker": ticker,
                "Sector": sector,
                "Sentiment_Score": sentiment['score'],
                "Label": sentiment['label'],
                "News_Count": sentiment['count']
            }
            sector_results.append(result)
            
            # Rate limit compliance (approx 1 call/sec)
            print(".", end="", flush=True)
            time.sleep(1.1) 
            
        print(" Done!")
        
        # Sort sector by sentiment score (descending)
        sector_results.sort(key=lambda x: x['Sentiment_Score'], reverse=True)
        all_results.extend(sector_results)
        
    df = pd.DataFrame(all_results)
    
    # Global Ranking
    print("\n" + "="*80)
    print(f"{'Ticker':<8} | {'Sector':<20} | {'Score':<6} | {'Label':<10} | {'News'}")
    print("-" * 80)
    
    # Show Top 10 Overall
    top_10 = df.sort_values(by="Sentiment_Score", ascending=False).head(10)
    for _, row in top_10.iterrows():
        print(f"{row['Ticker']:<8} | {row['Sector']:<20} | {row['Sentiment_Score']:<6.2f} | {row['Label']:<10} | {row['News_Count']}")
        
    print("="*80 + "\n")
    
    if save_csv:
        # Save CSV
        filename_csv = "daily_sentiment_report.csv"
        df.to_csv(filename_csv, index=False)
        print(f"Saved full report to {filename_csv}")
        
        # Save HTML
        filename_html = "daily_sentiment.html"
        html_content = f"""
        <html>
        <head>
            <title>Daily Sentiment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .Positive {{ color: green; font-weight: bold; }}
                .Negative {{ color: red; font-weight: bold; }}
                .Neutral {{ color: gray; }}
            </style>
        </head>
        <body>
            <h1>Daily Stock Sentiment Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Sector</th>
                    <th>Score</th>
                    <th>Label</th>
                    <th>News Count</th>
                </tr>
        """
        
        for _, row in df.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['Ticker']}</td>
                    <td>{row['Sector']}</td>
                    <td>{row['Sentiment_Score']:.4f}</td>
                    <td class="{row['Label']}">{row['Label']}</td>
                    <td>{row['News_Count']}</td>
                </tr>
            """
            
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(filename_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Saved HTML report to {filename_html}")

if __name__ == "__main__":
    # Run with a smaller limit for testing (e.g., 3 tickers per sector)
    run_daily_sentiment_ranking(limit_per_sector=3)
