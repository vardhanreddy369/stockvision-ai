"""
Unit Tests for StockVision AI Orchestrator
Tests pipeline coordination and data flow
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import run_pipeline


class TestOrchestratorModule(unittest.TestCase):
    """Test cases for src.orchestrator module"""

    @classmethod
    def setUpClass(cls):
        """Create sample test data CSV for orchestrator tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_csv = Path(cls.temp_dir) / "test_pipeline.csv"
        
        # Create larger sample dataset for pipeline testing
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = []
        
        for date in dates:
            for ticker in ['AAPL', 'MSFT', 'GOOGL']:
                close_price = np.random.uniform(100, 300)
                data.append({'Date': date, 'Ticker': ticker, 'Close': close_price})
        
        df = pd.DataFrame(data)
        df.to_csv(cls.test_csv, index=False)

    def test_run_pipeline_returns_tuple(self):
        """Test that run_pipeline returns (ranking, results) tuple"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        self.assertIsInstance(ranking, pd.DataFrame)
        self.assertIsInstance(results, dict)

    def test_run_pipeline_ranking_has_required_columns(self):
        """Test that ranking output has required columns"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        required_cols = ['Score', 'SharpeLike', 'Trend', 'WorstDrawdown']
        for col in required_cols:
            self.assertIn(col, ranking.columns)

    def test_run_pipeline_results_keys(self):
        """Test that results dict contains all analyzed tickers"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        # Results should have 2 tickers (top_k=2)
        self.assertEqual(len(results), 2)

    def test_run_pipeline_result_structure(self):
        """Test that each result has required fields"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        required_keys = ['current_close', 'next_pred_close', 'direction_signal', 
                         'direction_confidence', 'clf_accuracy', 'rank_metrics', 'backtest_df']
        
        for ticker, result in results.items():
            for key in required_keys:
                self.assertIn(key, result)

    def test_run_pipeline_predictions_are_numeric(self):
        """Test that model predictions are numeric"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        for ticker, result in results.items():
            self.assertIsInstance(result['current_close'], (int, float, np.number))
            self.assertIsInstance(result['next_pred_close'], (int, float, np.number))
            self.assertIsInstance(result['direction_confidence'], (int, float, np.number))
            self.assertIsInstance(result['clf_accuracy'], (int, float, np.number))

    def test_run_pipeline_direction_signal_valid(self):
        """Test that direction signals are UP or DOWN"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        for ticker, result in results.items():
            direction = result['direction_signal']
            # Direction can be "UP", "DOWN", "↑ UP", "↓ DOWN", or "N/A"
            self.assertTrue(any(x in direction for x in ['UP', 'DOWN', 'N/A']))

    def test_run_pipeline_ranking_is_sorted(self):
        """Test that ranking is sorted by score in descending order"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        scores = ranking['Score'].values
        # Check if sorted in descending order (allowing for near-equal scores)
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_run_pipeline_different_top_k(self):
        """Test run_pipeline with different top_k values"""
        for top_k in [1, 2, 3]:
            ranking, results = run_pipeline(str(self.test_csv), top_k=top_k, lookback=30, epochs=2)
            self.assertEqual(len(results), top_k)

    def test_run_pipeline_backtest_data_structure(self):
        """Test that backtest data is properly formatted"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=1, lookback=30, epochs=2)
        
        for ticker, result in results.items():
            backtest_df = result['backtest_df']
            if not backtest_df.empty:
                # Should have at least Strategy and BuyHold columns
                self.assertTrue(any(col in backtest_df.columns for col in ['Strategy', 'BuyHold']))

    def test_run_pipeline_confidence_in_range(self):
        """Test that confidence scores are in valid range [0, 1]"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        for ticker, result in results.items():
            confidence = result['direction_confidence']
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_run_pipeline_accuracy_in_range(self):
        """Test that model accuracy is in valid range [0, 1]"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=2, lookback=30, epochs=2)
        
        for ticker, result in results.items():
            accuracy = result['clf_accuracy']
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

    def test_run_pipeline_small_lookback(self):
        """Test pipeline with small lookback window"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=1, lookback=10, epochs=1)
        
        self.assertIsNotNone(ranking)
        self.assertIsNotNone(results)

    def test_run_pipeline_large_epochs(self):
        """Test pipeline with more training epochs"""
        ranking, results = run_pipeline(str(self.test_csv), top_k=1, lookback=30, epochs=3)
        
        self.assertIsNotNone(ranking)
        self.assertIsNotNone(results)


if __name__ == '__main__':
    unittest.main()
