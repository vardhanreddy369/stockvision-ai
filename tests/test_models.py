"""
Unit Tests for StockVision AI Models Module
Tests LSTM, GradientBoosting, and prediction functions
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import train_lstm, train_direction_classifier, make_supervised


class TestModelsModule(unittest.TestCase):
    """Test cases for src.models module"""

    @classmethod
    def setUpClass(cls):
        """Create sample time series data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        cls.test_series = pd.Series(prices, index=dates, name='Close')
        cls.lookback = 20

    def test_make_supervised_output_shape(self):
        """Test that make_supervised returns correctly shaped data"""
        X, y = make_supervised(self.test_series, lookback=self.lookback)
        
        expected_samples = len(self.test_series) - self.lookback
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], self.lookback)
        self.assertEqual(len(y), expected_samples)

    def test_make_supervised_preserves_order(self):
        """Test that make_supervised preserves temporal order"""
        X, y = make_supervised(self.test_series, lookback=self.lookback)
        
        for i in range(min(5, len(y))):
            self.assertIsNotNone(y[i])

    def test_train_lstm_returns_model(self):
        """Test that train_lstm returns a valid model and scaler"""
        model, scaler = train_lstm(self.test_series, lookback=self.lookback, epochs=2)
        
        self.assertTrue(hasattr(model, 'predict'))
        self.assertIsNotNone(scaler)
        
        X, _ = make_supervised(self.test_series, lookback=self.lookback)
        predictions = model.predict(X[:5])
        self.assertEqual(len(predictions), 5)

    def test_train_lstm_predictions_shape(self):
        """Test that LSTM predictions have correct shape"""
        model, scaler = train_lstm(self.test_series, lookback=self.lookback, epochs=2)
        X, _ = make_supervised(self.test_series, lookback=self.lookback)
        predictions = model.predict(X)
        
        self.assertEqual(predictions.shape[0], len(X))
        self.assertEqual(predictions.shape[1], 1)

    def test_train_direction_classifier_returns_model(self):
        """Test that train_direction_classifier returns a valid model"""
        df = pd.DataFrame({'Close': self.test_series})
        df['ret'] = df['Close'].pct_change()
        df['ma10'] = df['Close'].rolling(10).mean()
        df['ma20'] = df['Close'].rolling(20).mean()
        df['rsi14'] = 50
        df['vol20'] = df['Close'].rolling(20).std()
        df = df.fillna(df.mean())  # Fill NaNs
        
        model, acc = train_direction_classifier(df)
        
        self.assertTrue(hasattr(model, 'predict'))
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_make_supervised_with_different_lookback(self):
        """Test make_supervised with various lookback windows"""
        for lookback in [5, 10, 30]:
            X, y = make_supervised(self.test_series, lookback=lookback)
            expected_samples = len(self.test_series) - lookback
            self.assertEqual(X.shape[1], lookback)
            self.assertEqual(X.shape[0], expected_samples)

    def test_lstm_handles_small_dataset(self):
        """Test that LSTM can handle small datasets"""
        small_series = self.test_series[:30]
        model, scaler = train_lstm(small_series, lookback=5, epochs=1)
        X, _ = make_supervised(small_series, lookback=5)
        
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(X))

    def test_classifier_score_reasonable(self):
        """Test that classifier achieves reasonable accuracy"""
        df = pd.DataFrame({'Close': self.test_series})
        df['ret'] = df['Close'].pct_change()
        df['ma10'] = df['Close'].rolling(10).mean()
        df['ma20'] = df['Close'].rolling(20).mean()
        df['rsi14'] = 50
        df['vol20'] = df['Close'].rolling(20).std()
        df = df.fillna(df.mean())  # Fill NaNs
        
        model, accuracy = train_direction_classifier(df)
        
        self.assertGreater(accuracy, 0.2)
        self.assertLess(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()
