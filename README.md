<img width="2934" height="204" alt="image" src="https://github.com/user-attachments/assets/bcfdd710-c8d8-4960-b8ac-a017110aab32" />


**Intelligent Stock Analysis and Predictive Modeling Platform**

[![License MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-46%2F46%20passing-brightgreen.svg)](tests)

---

## Overview

StockVision AI is an enterprise-grade stock analysis platform that leverages deep learning and machine learning to deliver predictive insights for quantitative trading and portfolio management. The system processes historical market data, trains predictive models, and generates actionable signals across multiple equity tickers.

**Key Capabilities:** Real-time price predictions using LSTM neural networks, automated direction classification with gradient boosting, comprehensive backtesting, and market correlation analysis.

---

## Quick Start

### Access Live Platform

- **Interactive Dashboard:** [https://stockvision-ai-hglrftaeu7t3wxxis5hyjs.streamlit.app](https://stockvision-ai-hglrftaeu7t3wxxis5hyjs.streamlit.app)
- **Project Documentation:** [https://vardhanreddy369.github.io/stockvision-ai/](https://vardhanreddy369.github.io/stockvision-ai/)

### Local Development

```bash
git clone https://github.com/vardhanreddy369/stockvision-ai.git
cd stockvision-ai

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app/app.py
```

The application will be available at `http://localhost:8501`

---

## Core Features

**Predictive Analytics**
- LSTM-based price prediction with 60-day lookback window
- Gradient boosting classifier for directional signals
- Confidence scoring on all predictions

**Portfolio Intelligence**
- Automated ranking of equity securities
- Risk-adjusted performance metrics (Sharpe ratio, volatility)
- Correlation analysis across ticker universe

**Backtesting Engine**
- Historical strategy evaluation
- Performance comparison against buy-and-hold baseline
- Cumulative return analysis

**Market Insights**
- Volatility analysis and trend detection
- Correlation matrices for portfolio optimization
- Statistical summaries (mean, standard deviation, skewness, kurtosis)

---

## System Architecture

### Project Structure

```
stockvision-ai/
├── app/
│   └── app.py                 # Streamlit web application
├── src/
│   ├── orchestrator.py        # ML pipeline orchestration
│   ├── models.py              # Neural network and ML models
│   ├── scoring.py             # Portfolio scoring algorithms
│   ├── backtest.py            # Strategy backtesting
│   ├── features.py            # Feature engineering pipeline
│   └── utils.py               # Data utilities
├── tests/                     # Automated test suite (46 tests)
├── data/
│   └── stocks.csv             # Sample market data
├── requirements.txt           # Python dependencies
└── .streamlit/config.toml     # Streamlit configuration
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Web Framework | Streamlit 1.40+ |
| Deep Learning | TensorFlow 2.13, Keras |
| Machine Learning | Scikit-Learn 1.3+ |
| Data Processing | Pandas 2.0+, NumPy 1.24+ |
| Market Data | yfinance |
| Visualization | Plotly |
| Testing | Pytest |

---

## Installation and Deployment

### Requirements

- Python 3.8 or higher
- 512 MB disk space for dependencies

### Setup Instructions

**Step 1: Clone Repository**
```bash
git clone https://github.com/vardhanreddy369/stockvision-ai.git
cd stockvision-ai
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# or
venv\Scripts\activate          # Windows
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Run Application**
```bash
streamlit run app/app.py
```

### Cloud Deployment

Deploy to Streamlit Cloud:

1. Push repository to GitHub
2. Visit [https://share.streamlit.io](https://share.streamlit.io)
3. Authenticate with GitHub
4. Create new app pointing to `app/app.py`
5. Select branch and deploy

---

## Platform Capabilities

### Portfolio Summary Dashboard
- Top-ranked equity tickers based on composite scoring
- Risk-adjusted performance metrics
- Visual ranking comparison and heat maps

### Predictive Models
- Next-day price forecast with confidence intervals
- Direction signals (bullish/bearish) from ensemble methods
- Model confidence scores and prediction uncertainty

### Backtesting Module
- Historical performance evaluation
- Strategy returns vs. benchmark comparison
- Day-by-day performance attribution

### Market Analytics
- Volatility surface and trend analysis
- Inter-security correlation matrices
- Descriptive statistics across portfolio

---

## Machine Learning Models

### LSTM Neural Network
**Purpose:** Time series price prediction

- **Architecture:** Multi-layer LSTM with dropout regularization
- **Lookback Window:** 60 trading days
- **Training:** 10 epochs per security
- **Output:** Point estimate of next-day closing price

### Gradient Boosting Classifier
**Purpose:** Directional movement classification

- **Base Learners:** Decision trees with 5-10 depth
- **Features:** 42 engineered technical indicators
- **Output:** Binary classification (up/down movement)
- **Metrics:** Probability estimate per direction

---

## Testing and Validation

### Test Suite

Execute comprehensive test coverage:

```bash
pytest tests/ -v --cov=src
```

**Test Coverage:**
- Unit tests for all modules (46 tests)
- Integration tests for data pipeline
- Model training and prediction validation
- Backtesting engine verification

**Current Status:** 46/46 tests passing

---

## Data Processing Pipeline

1. **Data Ingestion** → Historical prices from CSV or API
2. **Preprocessing** → Normalization and NaN handling
3. **Feature Engineering** → Technical indicators and returns calculation
4. **Model Training** → LSTM and gradient boosting fitting
5. **Prediction Generation** → Price and direction forecasts
6. **Strategy Backtesting** → Historical performance evaluation
7. **Results Visualization** → Dashboard rendering

---

## Configuration

### Streamlit Configuration

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#0f3460"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f7fa"
textColor = "#1a1d23"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

---

## Performance and Scalability

- **Load Time:** 3-5 seconds with caching enabled
- **Data Processing:** Handles 50+ equity tickers per analysis
- **Model Training:** ~30 seconds per ticker (LSTM + Gradient Boosting)
- **Concurrency:** Single-user design for free tier deployment

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**MIT License Terms:**
- Commercial use permitted
- Source code modification allowed
- Redistribution permitted with license notice
- No warranty provided
- No liability assumed

---

## Contributing

Contributions are welcome. Please review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and best practices.

---

## Support and Documentation

- **Live Dashboard:** [https://stockvision-ai-hglrftaeu7t3wxxis5hyjs.streamlit.app](https://stockvision-ai-hglrftaeu7t3wxxis5hyjs.streamlit.app)
- **Project Repository:** [https://github.com/vardhanreddy369/stockvision-ai](https://github.com/vardhanreddy369/stockvision-ai)
- **Issues:** [GitHub Issues](https://github.com/vardhanreddy369/stockvision-ai/issues)

---

**Copyright © 2025 Vardhan Reddy Gutta. All rights reserved.**
