"""
Feature Generation Module
Technical indicators and feature engineering for stock prediction
"""
import pandas as pd
import numpy as np
from typing import Optional


def add_indicators(df_one):
    """
    Add technical indicators to stock data
    
    Args:
        df_one: DataFrame with 'Close' column
    
    Returns:
        DataFrame with added indicators: ret, ma10, ma20, rsi14, vol20
    """
    s = df_one["Close"]
    df_one["ret"] = s.pct_change()
    df_one["ma10"] = s.rolling(10).mean()
    df_one["ma20"] = s.rolling(20).mean()
    df_one["rsi14"] = _rsi(s, 14)
    df_one["vol20"] = df_one["ret"].rolling(20).std()
    df_one = df_one.dropna().copy()
    return df_one


def _rsi(series, n=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        series: Price series
        n: Period for RSI calculation (default: 14)
    
    Returns:
        RSI values (0-100)
    """
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))
