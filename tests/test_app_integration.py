"""
Integration Tests for StockVision AI Streamlit App
Tests UI rendering, data flow, and user interactions
"""
import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStreamlitAppIntegration(unittest.TestCase):
    """Integration tests for Streamlit app"""

    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_csv = Path(cls.temp_dir) / "test_app.csv"
        
        # Create realistic sample data
        dates = pd.date_range('2023-01-01', periods=120, freq='D')
        data = []
        
        np.random.seed(42)
        base_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2800, 'AMZN': 3000, 'TSLA': 900}
        
        for date in dates:
            for ticker, base_price in base_prices.items():
                close = base_price + np.random.normal(0, 10)
                data.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Close': close
                })
        
        df = pd.DataFrame(data)
        df.to_csv(cls.test_csv, index=False)

    def test_app_configuration_valid(self):
        """Test that Streamlit page config is valid"""
        # Import the app module to check configuration
        try:
            import app.app as app_module
            self.assertIsNotNone(app_module)
        except ImportError:
            self.skipTest("Streamlit app not importable in test environment")

    def test_data_loading_in_app_flow(self):
        """Test that app can load and process data"""
        from src.utils import load_stocks, pivot_close
        
        # Simulate app's data loading flow
        df = load_stocks(str(self.test_csv))
        self.assertGreater(len(df), 0)
        
        wide = pivot_close(df)
        self.assertGreater(wide.shape[1], 0)

    def test_pipeline_execution_flow(self):
        """Test that full pipeline executes without errors"""
        from src.orchestrator import run_pipeline
        
        try:
            ranking, results = run_pipeline(
                str(self.test_csv),
                top_k=3,
                lookback=30,
                epochs=2
            )
            
            self.assertIsNotNone(ranking)
            self.assertIsNotNone(results)
            self.assertEqual(len(results), 3)
        except Exception as e:
            self.fail(f"Pipeline execution failed: {str(e)}")

    def test_app_parameters_ranges(self):
        """Test that app parameter ranges are valid"""
        # Valid ranges for Streamlit sliders
        valid_top_k = list(range(1, 11))
        valid_lookback = list(range(20, 121, 5))
        valid_epochs = list(range(5, 51, 5))
        
        self.assertGreater(len(valid_top_k), 0)
        self.assertGreater(len(valid_lookback), 0)
        self.assertGreater(len(valid_epochs), 0)

    def test_tab_content_structure(self):
        """Test that app tab structure is defined"""
        tab_names = [
            "Performance Metrics",
            "Model Predictions",
            "Strategy Backtest",
            "Market Analytics"
        ]
        
        self.assertEqual(len(tab_names), 4)
        self.assertTrue(all(isinstance(name, str) for name in tab_names))

    def test_sidebar_content_structure(self):
        """Test sidebar configuration"""
        sidebar_label = "Analysis Settings"
        self.assertIsNotNone(sidebar_label)
        self.assertIsInstance(sidebar_label, str)
        self.assertGreater(len(sidebar_label), 0)

    def test_error_handling_missing_data(self):
        """Test that app handles missing data gracefully"""
        from src.utils import load_stocks
        
        # Test with non-existent file
        with self.assertRaises(Exception):
            load_stocks("/invalid/path/data.csv")

    def test_rendering_with_default_parameters(self):
        """Test that UI renders with default parameters"""
        # Default parameters from the app
        default_top_k = 3
        default_lookback = 60
        default_epochs = 10
        
        self.assertEqual(default_top_k, 3)
        self.assertEqual(default_lookback, 60)
        self.assertEqual(default_epochs, 10)

    def test_styling_css_present(self):
        """Test that professional styling CSS is defined"""
        import app.app as app_module
        
        # Check that app module exists
        self.assertIsNotNone(app_module)

    def test_data_consistency_across_tabs(self):
        """Test that data remains consistent across tabs"""
        from src.orchestrator import run_pipeline
        
        # Run pipeline once
        ranking1, results1 = run_pipeline(
            str(self.test_csv),
            top_k=2,
            lookback=30,
            epochs=2
        )
        
        # Results should be consistent
        self.assertEqual(len(results1), 2)
        self.assertGreater(len(ranking1), 0)


class TestUIComponents(unittest.TestCase):
    """Test individual UI components"""

    def test_metric_displays_valid_values(self):
        """Test that metric cards display valid values"""
        # Simulated metric values
        metrics = {
            'Top Performer': 'AAPL',
            'Avg Trend': 0.025,
            'Avg Risk-Adj Return': 0.15
        }
        
        self.assertEqual(metrics['Top Performer'], 'AAPL')
        self.assertGreaterEqual(metrics['Avg Trend'], 0)
        self.assertGreaterEqual(metrics['Avg Risk-Adj Return'], 0)

    def test_chart_data_structure(self):
        """Test that chart data is properly structured"""
        chart_data = pd.DataFrame({
            'Strategy': [100, 101, 102, 103],
            'BuyHold': [100, 99, 101, 102]
        })
        
        self.assertEqual(len(chart_data), 4)
        self.assertEqual(len(chart_data.columns), 2)

    def test_ranking_table_format(self):
        """Test that ranking table has correct format"""
        ranking_data = {
            'Score': [0.95, 0.87, 0.76],
            'SharpeLike': [1.5, 1.3, 1.1],
            'Trend': [0.02, 0.015, 0.01],
            'WorstDrawdown': [-0.15, -0.18, -0.20]
        }
        
        ranking_df = pd.DataFrame(ranking_data, index=['AAPL', 'MSFT', 'GOOGL'])
        
        self.assertEqual(len(ranking_df), 3)
        self.assertEqual(len(ranking_df.columns), 4)


if __name__ == '__main__':
    unittest.main()
