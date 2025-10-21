"""
Model Management Module
LSTM and GradientBoosting models for stock price prediction and direction classification
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def make_supervised(series, lookback=60):
    """
    Convert time series into supervised learning format
    
    Args:
        series: 1D time series array
        lookback: Number of past timesteps to use as input
    
    Returns:
        X: Array of shape (n, lookback, 1)
        y: Array of target values
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    X = np.array(X)[..., None]  # (n, lookback, 1)
    y = np.array(y)
    return X, y


def train_lstm(close, lookback=60, epochs=10):
    """
    Train LSTM model on close prices
    
    Args:
        close: Series of close prices
        lookback: Number of past timesteps
        epochs: Number of training epochs
    
    Returns:
        model: Trained Keras LSTM model
        scaler: Fitted MinMaxScaler for inverse transform
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.values.reshape(-1, 1)).ravel()
    X, y = make_supervised(scaled, lookback)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=32, verbose=0)
    
    return model, scaler


def train_direction_classifier(df_with_feats):
    """
    Train GradientBoosting classifier to predict next-day price direction
    
    Args:
        df_with_feats: DataFrame with columns ['Close', 'ret', 'ma10', 'ma20', 'rsi14', 'vol20']
    
    Returns:
        clf: Trained GradientBoostingClassifier
        acc: Accuracy on test set
    """
    # Label: next-day up (1) / down (0)
    y = (df_with_feats["Close"].shift(-1) > df_with_feats["Close"]).astype(int)[:-1]
    X = df_with_feats[["ret", "ma10", "ma20", "rsi14", "vol20"]].iloc[:-1]
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    clf = GradientBoostingClassifier()
    clf.fit(Xtr, ytr)
    
    acc = (clf.predict(Xte) == yte).mean()
    
    return clf, acc
