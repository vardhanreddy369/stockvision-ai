# 📊 StockVision AI

Intelligent Stock Analysis & Predictive Modeling System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)

**🌐 [View Live Dashboard](https://stockvision-ai.streamlit.app)** | **📖 [GitHub Pages Site](https://vardhanreddy369.github.io/stockvision-ai)** | **📄 [LICENSE](LICENSE)**

## Overview

StockVision AI is an advanced stock analysis platform that combines machine learning models with real-time market data to provide actionable insights for investors. The system analyzes multiple tickers simultaneously, trains predictive models, and delivers comprehensive performance analytics.

## 🚀 Features

- **📈 Real-time Stock Analysis** - Instant analysis of stock performance with up-to-date market data
- **🤖 AI-Powered Predictions** - LSTM neural networks for price prediction + GradientBoosting for direction classification
- **💼 Portfolio Management** - Comprehensive portfolio summaries with risk metrics and correlation analysis
- **⚡ Backtesting Engine** - Validate trading strategies against historical data
- **📊 Market Analytics** - Volatility analysis, correlation matrices, and statistical summaries
- **🎯 Automated Ranking** - Smart ticker ranking based on Sharpe ratio, trend, and drawdown recovery

## 🏗️ Architecture

### Components

```
stockvision-ai/
├── app/
│   └── app.py              # Main Streamlit dashboard
├── src/
│   ├── orchestrator.py     # Pipeline orchestration
│   ├── models.py           # LSTM and ML models
│   ├── scoring.py          # Ticker ranking logic
│   ├── backtest.py         # Backtesting engine
│   ├── features.py         # Feature engineering
│   └── utils.py            # Utility functions
├── tests/                  # Comprehensive test suite
└── data/
    └── stocks.csv          # Sample stock data
```

### Technology Stack

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit 1.28.0
- **ML/AI**: TensorFlow, Keras, Scikit-Learn
- **Data Processing**: Pandas, NumPy
- **Market Data**: yfinance
- **Visualization**: Plotly

## 📋 Requirements

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
tensorflow==2.13.0
streamlit==1.28.0
plotly==5.16.1
yfinance==0.2.28
python-dateutil==2.8.2
protobuf==3.20.0
```

## 🚀 Getting Started

### Local Installation

```bash
# Clone the repository
git clone https://github.com/vardhanreddy369/stockvision-ai.git
cd stockvision-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

The app will be available at `http://localhost:8501`

### Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and branch
5. Set main file path to `app/app.py`
6. Click Deploy

## 📊 Dashboard Tabs

### 1. **Portfolio Summary**
- Top ticker rankings based on composite scores
- Performance metrics (Sharpe ratio, trend, drawdown recovery)
- Visual comparison of ranked tickers

### 2. **Model Predictions**
- Next-day price predictions using LSTM
- Direction signals from GradientBoosting classifier
- Confidence scores and prediction details

### 3. **Backtest Results**
- Strategy performance vs. Buy & Hold
- Cumulative returns visualization
- Day-by-day performance breakdown

### 4. **Market Analytics**
- Volatility analysis across tickers
- Correlation matrix
- Statistical summaries (mean, std dev, skewness, kurtosis)

## 🧠 Machine Learning Models

### LSTM (Long Short-Term Memory)
- **Purpose**: Price prediction
- **Architecture**: 60-day lookback window
- **Training**: 10 epochs per ticker
- **Output**: Next-day price forecast

### GradientBoosting Classifier
- **Purpose**: Direction classification (Up/Down)
- **Features**: 42 engineered features after NaN removal
- **Output**: Direction signal with confidence score

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Test coverage includes:
- Data loading and preprocessing
- Model training and predictions
- Backtesting logic
- Scoring and ranking algorithms
- End-to-end integration tests

## 📈 Data Pipeline

1. **Load**: Historical stock data from CSV/yfinance
2. **Process**: Calculate returns, clean NaN values
3. **Score**: Compute Sharpe ratio, trend, drawdown metrics
4. **Rank**: Select top 3 tickers
5. **Train**: LSTM and GradientBoosting models
6. **Predict**: Generate price and direction predictions
7. **Backtest**: Evaluate strategy performance
8. **Visualize**: Dashboard display with professional styling

## 🎨 UI/UX Features

- Professional gradient design with navy blue (#0f3460) and green accents (#16a34a)
- Light theme for all visualizations
- Responsive layout with full-width tables
- Light gray column headers (#e2e8f0) for better visibility
- Consistent spacing and typography using Inter and Poppins fonts
- Smooth hover effects and active state indicators

## 🔧 Configuration

### Streamlit Config

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#0f3460"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f7fa"
textColor = "#1a1d23"
font = "sans serif"

[client]
toolbarMode = "minimal"
```

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ **Commercial use** - You can use this software for commercial purposes
- ✅ **Modification** - You can modify the source code
- ✅ **Distribution** - You can distribute the software
- ✅ **Private use** - You can use this software privately
- ⚠️ **Liability** - The software is provided "as is" without warranty

---

**Built with ❤️ using Python, TensorFlow, and Streamlit**

Copyright © 2025 Vardhan Reddy Gutta. All rights reserved.
