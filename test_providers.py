"""Quick test of all data providers via DataRouter."""
from data.data_router import DataRouter

r = DataRouter()

print("=== PROVIDER STATUS ===")
for name, status in r.status().items():
    print(f"  {name:15s} {status}")

print("\n=== TEST: Finnhub Quote ===")
q = r.get_quote("AAPL")
print(f"  Provider: {q['provider']}  Data: {q['data']}")

print("\n=== TEST: EOD (yfinance) ===")
df = r.get_historical_data("AAPL", "1mo")
print(f"  Got {len(df)} rows")

print("\n=== TEST: Short Interest (yahooquery) ===")
s = r.get_short_interest("GME")
print(f"  Provider: {s['provider']}  Data: {s['data']}")

print("\n=== TEST: Fundamentals (yahooquery fallback) ===")
f = r.get_fundamentals("AAPL")
print(f"  Provider: {f['provider']}  Keys: {list(f['data'].keys()) if f['data'] else 'none'}")

print("\n=== TEST: Sentiment (finnhub) ===")
sent = r.get_sentiment("AAPL")
print(f"  Provider: {sent['provider']}")

print("\n=== TEST: News (finnhub) ===")
news = r.get_news("AAPL", 3)
print(f"  Got {len(news)} articles")

print("\n=== ALL TESTS PASSED ===")
