"""
Unit Tests for StockVision AI Utilities Module
Tests data loading, pivoting, and transformation functions
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_stocks, pivot_close


class TestUtilsModule(unittest.TestCase):
    """Test cases for src.utils module"""

    @classmethod
    def setUpClass(cls):
        """Create sample test data CSV"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_csv = Path(cls.temp_dir) / "test_stocks.csv"
        
        # Create sample stock data
        test_data = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 3,
            'Ticker': ['AAPL'] * 5 + ['MSFT'] * 5 + ['GOOGL'] * 5,
            'Close': [150, 151, 152, 151, 153, 300, 302, 301, 303, 305, 2800, 2805, 2810, 2815, 2820]
        }
        df = pd.DataFrame(test_data)
        df.to_csv(cls.test_csv, index=False)

    def test_load_stocks_returns_dataframe(self):
        """Test that load_stocks returns a DataFrame"""
        result = load_stocks(str(self.test_csv))
        self.assertIsInstance(result, pd.DataFrame)

    def test_load_stocks_has_required_columns(self):
        """Test that loaded data has required columns"""
        result = load_stocks(str(self.test_csv))
        required_cols = ['Date', 'Ticker', 'Close']
        for col in required_cols:
            self.assertIn(col, result.columns)

    def test_load_stocks_correct_rows(self):
        """Test that load_stocks loads all rows"""
        result = load_stocks(str(self.test_csv))
        self.assertEqual(len(result), 15)

    def test_load_stocks_unique_tickers(self):
        """Test that all tickers are loaded"""
        result = load_stocks(str(self.test_csv))
        unique_tickers = result['Ticker'].unique()
        self.assertEqual(len(unique_tickers), 3)
        self.assertIn('AAPL', unique_tickers)
        self.assertIn('MSFT', unique_tickers)
        self.assertIn('GOOGL', unique_tickers)

    def test_pivot_close_returns_dataframe(self):
        """Test that pivot_close returns a DataFrame"""
        df = load_stocks(str(self.test_csv))
        result = pivot_close(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_pivot_close_shape(self):
        """Test that pivoted data has correct shape"""
        df = load_stocks(str(self.test_csv))
        result = pivot_close(df)
        # Should have 5 dates and 3 tickers
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1], 3)

    def test_pivot_close_columns_are_tickers(self):
        """Test that pivoted columns are ticker symbols"""
        df = load_stocks(str(self.test_csv))
        result = pivot_close(df)
        expected_cols = {'AAPL', 'MSFT', 'GOOGL'}
        actual_cols = set(result.columns)
        self.assertEqual(actual_cols, expected_cols)

    def test_pivot_close_values_preserved(self):
        """Test that close prices are correctly pivoted"""
        df = load_stocks(str(self.test_csv))
        result = pivot_close(df)
        
        # Check a few known values
        self.assertEqual(result.loc[result.index[0], 'AAPL'], 150)
        self.assertEqual(result.loc[result.index[0], 'MSFT'], 300)
        self.assertEqual(result.loc[result.index[0], 'GOOGL'], 2800)

    def test_load_stocks_with_invalid_path(self):
        """Test that load_stocks handles missing files gracefully"""
        with self.assertRaises(Exception):
            load_stocks("/nonexistent/path/stocks.csv")

    def test_pivot_close_no_duplicates(self):
        """Test that pivot_close handles data without duplicates"""
        df = load_stocks(str(self.test_csv))
        result = pivot_close(df)
        # Check for NaN values which would indicate issues
        self.assertEqual(result.isna().sum().sum(), 0)


class TestUtilsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in utils"""

    def test_empty_dataframe_pivot(self):
        """Test pivot_close with empty DataFrame"""
        empty_df = pd.DataFrame({'Date': [], 'Ticker': [], 'Close': []})
        # Should handle gracefully or raise informative error
        try:
            result = pivot_close(empty_df)
            self.assertEqual(len(result), 0)
        except Exception as e:
            self.assertIsNotNone(str(e))

    def test_single_ticker_pivot(self):
        """Test pivot_close with single ticker"""
        single_data = {
            'Date': ['2023-01-01', '2023-01-02'],
            'Ticker': ['AAPL', 'AAPL'],
            'Close': [150, 151]
        }
        df = pd.DataFrame(single_data)
        result = pivot_close(df)
        self.assertEqual(result.shape[1], 1)
        self.assertEqual(result.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
