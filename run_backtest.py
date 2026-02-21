import pandas as pd
from data.yfinance_client import YFinanceClient
from data.mock_client import MockDataClient
from strategy.indicators import IndicatorLibrary
from strategy.scoring import StrategyScorer
from backtest.engine import BacktestEngine

def run_backtest_on_ticker(ticker: str, period: str = "5y"):
    """Fetch historical data and run backtest for a ticker."""
    yf_client = YFinanceClient()
    mock_client = MockDataClient()
    scorer = StrategyScorer()
    engine = BacktestEngine()
    
    print(f"--- Backtest Results for {ticker} ({period}) ---")
    
    # 1. Fetch Data
    df = yf_client.get_historical_data(ticker, period=period)
    if df.empty:
        print(f"Falling back to Mock data for {ticker}")
        df = mock_client.get_historical_data(ticker, period=period)
        
    if df.empty:
        print("Error: No data found.")
        return
        
    # 2. Add Indicators
    df = IndicatorLibrary.add_all_indicators(df)
    
    # 3. Calculate Scores for the whole history (Vectorized)
    df = scorer.score_dataframe(df)
    
    # 4. Run Backtest
    results = engine.run_backtest(df, score_column='total_score', threshold=75.0)
    
    if "error" in results:
        print(f"Backtest Error: {results['error']}")
        return
        
    metrics = results['metrics']
    print(f"Total Return:     {metrics['total_return'] * 100:.2f}%")
    print(f"Annualized Sharpe: {metrics['sharpe']:.2f}")
    print(f"Annualized Sortino:{metrics['sortino']:.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Daily Win Rate:   {metrics['win_rate'] * 100:.2f}%")
    
    final_equity = df.iloc[-1]['equity_curve'] if 'equity_curve' in df.columns else 0
    print(f"Final Capital:    ${final_equity:.2f}")

if __name__ == "__main__":
    run_backtest_on_ticker("SPY", period="5y")
