import pandas as pd
import numpy as np
from typing import Dict, Any

class BacktestEngine:
    """Vectorized backtesting engine for technical strategies."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

    def run_backtest(self, df: pd.DataFrame, score_column: str = 'total_score', threshold: float = 75.0) -> Dict[str, Any]:
        """
        Run a vectorized backtest based on strategy scores.
        
        Args:
            df: DataFrame containing 'Close' and the score column.
            score_column: The column to use for entry signals.
            threshold: The score threshold to enter a long position.
            
        Returns:
            Dict: Performance metrics and equity curve data.
        """
        if df.empty or 'Close' not in df.columns or score_column not in df.columns:
            return {"error": "Invalid data for backtest"}

        # Calculate daily returns
        df = df.copy()
        df['daily_return'] = df['Close'].pct_change()
        
        # Signals: 1 if score > threshold, else 0
        # Shift signal by 1 to avoid look-ahead bias (we enter at next day's open/close)
        df['signal'] = (df[score_column] > threshold).astype(int).shift(1).fillna(0)
        
        # Strategy Return
        df['strategy_return'] = df['signal'] * df['daily_return']
        
        # Equity Curve
        df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod()
        df['cum_market_return'] = (1 + df['daily_return']).cumprod()
        
        # Convert to capital
        df['equity_curve'] = self.initial_capital * df['cum_strategy_return']
        
        # Metrics
        metrics = self.calculate_metrics(df)
        
        return {
            "metrics": metrics,
            "df": df # Return the df for plotting/analysis if needed
        }

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Sharpe, Sortino, Drawdown, Win Rate."""
        returns = df['strategy_return'].dropna()
        if returns.empty or returns.std() == 0:
            return {"sharpe": 0, "sortino": 0, "max_drawdown": 0, "win_rate": 0, "total_return": 0}
            
        # Annualized Sharpe Ratio (risk-free rate = 0 for simplicity)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # Annualized Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252) if not downside_returns.empty else np.inf
        
        # Max Drawdown
        equity = df['equity_curve']
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate (on a trade-by-trade basis is harder in vectorized, so we use daily win rate of active signal)
        active_returns = returns[df['signal'] == 1]
        win_rate = (active_returns > 0).mean() if not active_returns.empty else 0
        
        total_return = (df['equity_curve'].iloc[-1] / self.initial_capital) - 1
        
        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "total_return": float(total_return)
        }

if __name__ == "__main__":
    # Dummy test
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        'Close': np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
        'total_score': np.random.uniform(50, 100, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    engine = BacktestEngine()
    results = engine.run_backtest(df)
    print("Metrics:", results['metrics'])
