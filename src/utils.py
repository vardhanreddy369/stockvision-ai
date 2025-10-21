"""
Utilities Module
Helper functions and utilities for the entire pipeline
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path


def load_stocks(path="data/stocks.csv"):
    """
    Load stock data from CSV file
    
    Args:
        path: Path to stocks CSV file
    
    Returns:
        DataFrame with stock data sorted by Ticker and Date
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])
    return df


def pivot_close(df):
    """
    Pivot Close prices by Date and Ticker
    Creates a matrix with dates as index and tickers as columns
    
    Args:
        df: DataFrame with Date, Ticker, and Close columns
    
    Returns:
        Pivoted DataFrame with dates as index and tickers as columns
    """
    return df.pivot(index="Date", columns="Ticker", values="Close")


class Logger:
    """Logging utilities"""
    
    @staticmethod
    def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
        """
        Setup a logger with console and file handlers
        
        Args:
            name: Logger name
            level: Logging level
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(console_handler)
        
        return logger


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_stock_data(df: pd.DataFrame) -> bool:
        """
        Validate stock data format and content
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for missing values
        if df[required_columns].isnull().any().any():
            return False
        
        # Check for negative values
        if (df[required_columns] < 0).any().any():
            return False
        
        # Check High >= Close, Low
        if not (df['High'] >= df['Close']).all():
            return False
        if not (df['High'] >= df['Low']).all():
            return False
        
        # Check Close >= Low
        if not (df['Close'] >= df['Low']).all():
            return False
        
        return True
    
    @staticmethod
    def validate_predictions(predictions: pd.DataFrame) -> bool:
        """
        Validate prediction data
        
        Args:
            predictions: DataFrame with predictions
        
        Returns:
            True if valid, False otherwise
        """
        if predictions.empty:
            return False
        
        required_columns = ['predicted_price']
        
        if not all(col in predictions.columns for col in required_columns):
            return False
        
        if predictions['predicted_price'].isnull().any():
            return False
        
        if (predictions['predicted_price'] <= 0).any():
            return False
        
        return True


class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize stock data
        
        Args:
            df: Raw stock data
        
        Returns:
            Cleaned data
        """
        df = df.copy()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        # Fill missing dates (forward fill)
        df = df.asfreq('D').fillna(method='ffill')
        
        # Remove rows with NaN
        df = df.dropna()
        
        return df
    
    @staticmethod
    def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for training data
        
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        return df[:split_idx], df[split_idx:]
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Normalize features using min-max scaling
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to normalize
        
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        for col in feature_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
        
        return df


class FileManager:
    """File management utilities"""
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """
        Ensure directory exists, create if not
        
        Args:
            path: Directory path
        
        Returns:
            Path object
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str) -> bool:
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filepath: Path to save to
        
        Returns:
            True if successful, False otherwise
        """
        try:
            FileManager.ensure_directory(Path(filepath).parent)
            df.to_csv(filepath)
            return True
        except Exception as e:
            print(f"Error saving DataFrame: {e}")
            return False
    
    @staticmethod
    def load_dataframe(filepath: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from CSV file
        
        Args:
            filepath: Path to load from
        
        Returns:
            DataFrame or None if error
        """
        try:
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return None


class ConfigManager:
    """Configuration management utilities"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """
        Load configuration from file
        
        Args:
            config_path: Path to config file
        
        Returns:
            Configuration dictionary
        """
        import json
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict, config_path: str) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            config_path: Path to save to
        
        Returns:
            True if successful, False otherwise
        """
        import json
        
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
