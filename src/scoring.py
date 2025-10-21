"""
Scoring Module
Intelligent ticker ranking and prediction quality metrics
"""
import numpy as np
import pandas as pd
from typing import Optional


def score_tickers(pivot_close):
    """
    Score and rank tickers based on risk/return metrics, trend, and stability
    
    Args:
        pivot_close: DataFrame with dates as index and tickers as columns (Close prices)
    
    Returns:
        DataFrame with rankings sorted by composite score (descending)
        Columns: SharpeLike, Trend, WorstDrawdown, Score
    """
    # Risk/return on daily % changes
    rets = pivot_close.pct_change().dropna()
    avg = rets.mean()
    risk = rets.std()
    sharpe = (avg / (risk + 1e-9)).replace([np.inf, -np.inf], 0)

    # Trend & stability
    trend = (pivot_close.iloc[-1] / pivot_close.iloc[0] - 1.0)
    drawdown = (pivot_close / pivot_close.cummax()).min()  # worst drawdown

    # Normalize and combine
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    score = 0.45*z(sharpe) + 0.35*z(trend) + 0.20*(-z(drawdown))
    
    ranking = pd.DataFrame({
        "SharpeLike": sharpe,
        "Trend": trend,
        "WorstDrawdown": drawdown,
        "Score": score
    }).sort_values("Score", ascending=False)
    
    return ranking


class QualityMetrics:
    """
    Calculate prediction quality metrics
    """
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAPE value (%)
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Percentage of correct directional predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Directional accuracy (%)
        """
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        
        return (correct / total) * 100 if total > 0 else 0
