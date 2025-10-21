"""
Backtest Module
Simple long-only strategy backtesting and performance evaluation
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def backtest_long_only(close_series, preds_next_close):
    """
    Backtest a simple long-only strategy based on price predictions
    
    Buy if predicted next close > current close; hold for 1 day
    
    Args:
        close_series: Series of actual close prices
        preds_next_close: Series of predicted next-day close prices
    
    Returns:
        DataFrame with cumulative returns:
        - "Strategy": Cumulative returns from signal-based strategy
        - "BuyHold": Cumulative returns from buy-and-hold baseline
    """
    # Realized next-day return
    actual_ret = close_series.pct_change().shift(-1)
    
    # Buy signal: predicted next close > today close
    signal = (preds_next_close > close_series).astype(int)
    
    # Strategy returns: buy (1) or no position (0)
    strat_ret = (actual_ret * signal).fillna(0)
    
    # Cumulative returns
    cum_strategy = (1 + strat_ret).cumprod()
    cum_bh = (1 + actual_ret.fillna(0)).cumprod()  # buy & hold baseline
    
    return pd.DataFrame({
        "Strategy": cum_strategy,
        "BuyHold": cum_bh
    })


class BacktestEngine:
    """
    Backtest trading strategies on historical data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize backtest engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
    
    def run_backtest(self, df: pd.DataFrame, strategy_params: Optional[Dict] = None) -> Dict:
        """
        Run backtest on historical data with trading signals
        
        Args:
            df: DataFrame with OHLCV and trading signals
            strategy_params: Strategy parameters
        
        Returns:
            Backtest results with performance metrics
        """
        strategy_params = strategy_params or {}
        
        # Generate signals
        signals = self._generate_signals(df, strategy_params)
        
        # Simulate trades
        equity_curve = self._simulate_trades(df, signals)
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, df)
        
        return metrics
    
    def _generate_signals(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            df: Historical data
            params: Strategy parameters
        
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Simple moving average crossover strategy
        if 'ma_10' in df.columns and 'ma_20' in df.columns:
            # Buy signal: MA10 > MA20
            signals.loc[df['ma_10'] > df['ma_20'], 'signal'] = 1
            # Sell signal: MA10 < MA20
            signals.loc[df['ma_10'] < df['ma_20'], 'signal'] = -1
        
        # Generate position signals
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def _simulate_trades(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        Simulate trades and calculate equity curve
        
        Args:
            df: Historical data
            signals: Trading signals
        
        Returns:
            Equity curve series
        """
        equity = pd.Series(index=df.index, dtype=float)
        equity.iloc[0] = self.initial_capital
        
        position = 0  # Current position: 0 = no position, 1 = long
        shares = 0
        cash = self.initial_capital
        
        for i in range(1, len(df)):
            if signals['position'].iloc[i] == 1:  # Buy signal
                if position == 0:
                    # Calculate position size
                    shares = (cash * (1 - self.risk_per_trade)) / df['Close'].iloc[i]
                    cash -= shares * df['Close'].iloc[i]
                    position = 1
            
            elif signals['position'].iloc[i] == -1:  # Sell signal
                if position == 1:
                    # Close position
                    cash += shares * df['Close'].iloc[i]
                    shares = 0
                    position = 0
            
            # Update equity value
            portfolio_value = cash + (shares * df['Close'].iloc[i])
            equity.iloc[i] = portfolio_value
        
        # Close any remaining position
        if position == 1:
            cash += shares * df['Close'].iloc[-1]
        
        equity.iloc[-1] = cash
        
        return equity
    
    def _calculate_metrics(self, equity_curve: pd.Series, df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            equity_curve: Equity curve series
            df: Historical data for dates
        
        Returns:
            Dictionary of performance metrics
        """
        returns = equity_curve.pct_change().dropna()
        
        # Total return
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        trading_days = len(df)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_equity': equity_curve.iloc[-1],
            'num_trades': total_trades
        }


class RiskAnalyzer:
    """
    Analyze and manage trading risk
    """
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Series of returns
            confidence: Confidence level (0.95 = 95%)
        
        Returns:
            VaR value
        """
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence: Confidence level
        
        Returns:
            CVaR value
        """
        var = np.percentile(returns, (1 - confidence) * 100)
        return returns[returns <= var].mean()
