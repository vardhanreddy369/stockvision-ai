"""
StockVision AI — Intelligent Stock Analysis & Modeling
Orchestrator-powered Streamlit app with professional dashboard
"""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import importlib.util

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

from src.orchestrator import run_pipeline
from src.utils import load_stocks, pivot_close

# ============================================================================
# CACHING FUNCTIONS - Speed up app loading
# ============================================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_pipeline_data(csv_path, top_k, lookback, epochs):
    """Cache pipeline results"""
    return run_pipeline(csv_path, top_k=top_k, lookback=lookback, epochs=epochs)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e9ecf1 100%);
            font-family: 'Inter', 'Segoe UI', sans-serif;
            color: #1a1d23;
        }

        .main-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            text-align: center;
            font-weight: 700;
            color: #0f3460;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }

        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #4a5568;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }

        .info-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0px 2px 12px rgba(15, 52, 96, 0.06);
            border-left: 5px solid #16a34a;
            margin-bottom: 1.5rem;
            min-height: 280px;
        }

        .info-card h4 {
            font-family: 'Poppins', sans-serif;
            color: #0f3460;
            margin-bottom: 1.2rem;
            font-size: 1.2rem;
            margin-top: 0;
        }

        .info-card p {
            font-size: 0.95rem;
            color: #4a5568;
            line-height: 1.7;
            margin: 0;
        }

        .tip-card {
            background: linear-gradient(135deg, #ecfdf5 0%, #dcfce7 100%);
            border-radius: 12px;
            padding: 2rem;
            border: 1px solid #bbf7d0;
            box-shadow: 0px 2px 8px rgba(22, 163, 74, 0.08);
            margin-bottom: 1.5rem;
            min-height: 280px;
        }

        .tip-card h4 {
            font-family: 'Poppins', sans-serif;
            color: #065f46;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            margin-top: 0;
        }

        .tip-card p {
            font-size: 0.95rem;
            color: #047857;
            margin: 0;
            line-height: 1.7;
        }

        h2, h3 {
            font-family: 'Poppins', sans-serif;
            color: #0f3460;
        }

        .stMetric {
            background: #ffffff;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.04);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            font-family: 'Poppins', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: #4a5568;
            border-radius: 8px 8px 0 0;
        }

        .stTabs [aria-selected="true"] {
            color: #0f3460 !important;
            border-bottom: 3px solid #16a34a !important;
        }

        .showcase-container {
            background: linear-gradient(135deg, #0f3460 0%, #1e5a96 100%);
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0px 8px 32px rgba(15, 52, 96, 0.15);
        }

        .showcase-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.3rem;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 1.5rem;
            letter-spacing: 0.5px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .metric-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }

        .metric-label {
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-family: 'Poppins', sans-serif;
            font-size: 1.8rem;
            color: #22c55e;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .metric-subtext {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.6);
            font-weight: 500;
        }

        .ticker-badge {
            display: inline-block;
            background: #16a34a;
            color: #ffffff;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        .performance-table {
            background: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0px 4px 16px rgba(15, 52, 96, 0.08);
            margin-top: 1.5rem;
        }

        .performance-table table {
            width: 100%;
            border-collapse: collapse;
        }

        .performance-table th {
            background: #0f3460;
            color: #ffffff;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .performance-table td {
            padding: 0.9rem 1rem;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.95rem;
        }

        .performance-table tbody tr:hover {
            background: #f8fafb;
        }

        .performance-table tbody tr:last-child td {
            border-bottom: none;
        }

        .ticker-name {
            font-weight: 700;
            color: #0f3460;
            font-size: 1rem;
        }

        .score-badge {
            display: inline-block;
            background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
            color: #ffffff;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .score-low {
            background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        }

        .score-medium {
            background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        }

        .direction-up {
            color: #16a34a;
            font-weight: 700;
        }

        .direction-down {
            color: #ef4444;
            font-weight: 700;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f3460 0%, #1e5a96 50%, #0f3460 100%);
        }

        [data-testid="stSidebar"] > div {
            background: transparent;
        }

        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            letter-spacing: 0.5px;
        }

        [data-testid="stSidebar"] label {
            color: #e8eef5 !important;
            font-weight: 600;
            font-size: 0.9rem;
        }

        [data-testid="stSidebar"] .stSlider {
            margin: 1rem 0;
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: #d4d9e3 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2) !important;
            margin: 1.5rem 0 !important;
        }

        [data-testid="stSidebar"] .stMetric {
            background: transparent;
        }

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.8rem;
            border-bottom: none;
            display: flex;
            justify-content: space-between;
            margin-bottom: 2rem;
            width: 100%;
        }

        .stTabs [data-baseweb="tab"] {
            font-family: 'Poppins', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: #4a5568;
            background: #f0f4f8;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
            flex: 1;
            text-align: center;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: #e8eef5;
            border-color: #16a34a;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: #16a34a !important;
            border-color: #16a34a !important;
            box-shadow: 0px 4px 12px rgba(22, 163, 74, 0.3);
        }

        /* Table & Metric Styling */
        [data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.04);
        }

        [data-testid="stMetric"] > div {
            background: transparent !important;
        }

        [data-testid="stMetric"] > div:first-child {
            background: linear-gradient(135deg, #0f3460 0%, #1e5a96 100%) !important;
            color: #ffffff !important;
            padding: 0.9rem 1.2rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }

        [data-testid="stMetric"] label {
            background: linear-gradient(135deg, #0f3460 0%, #1e5a96 100%) !important;
            color: #ffffff !important;
            padding: 0.9rem 1.2rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }

        /* DataFrame Table Styling */
        [data-testid="stDataFrame"] {
            background: transparent;
        }

        [data-testid="stDataFrame"] thead {
            background: linear-gradient(135deg, #0f3460 0%, #1e5a96 100%) !important;
        }

        [data-testid="stDataFrame"] th {
            background: linear-gradient(135deg, #0f3460 0%, #1e5a96 100%) !important;
            color: #e2e8f0 !important;
            padding: 1.2rem 1rem !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            border-bottom: 3px solid #16a34a !important;
        }

        [data-testid="stDataFrame"] tbody tr {
            border-bottom: 1px solid #e2e8f0 !important;
        }

        [data-testid="stDataFrame"] tbody tr:hover {
            background: #f8fafb !important;
        }

        [data-testid="stDataFrame"] td {
            padding: 0.9rem 1rem !important;
            color: #1a1d23 !important;
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
            color: #1a1d23 !important;
        }

        /* Subheader Styling */
        h2, [data-testid="stHeadingContainer"] h2 {
            color: #0f3460 !important;
            border-bottom: 3px solid #16a34a;
            padding-bottom: 0.8rem;
            margin-bottom: 1.5rem;
        }

        h3 {
            color: #0f3460 !important;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# TITLE & DESCRIPTION
# ============================================================================
st.markdown('<h1 class="main-title">StockVision AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Intelligent Stock Analysis & Predictive Modeling Platform</p>',
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    st.markdown(
        """
        <div class="info-card">
            <h4>Comprehensive Analysis</h4>
            <p>Intelligent ticker ranking, neural network predictions, and strategic backtesting to help you understand market dynamics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        """
        <div class="info-card">
            <h4>AI-Powered Insights</h4>
            <p>Advanced machine learning models analyze price trends, predict direction signals, and evaluate strategy performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        """
        <div class="tip-card">
            <h4>Getting Started</h4>
            <p>Adjust settings on the left sidebar to customize your analysis. Results update automatically with each change.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown("### ⚙️ Analysis Settings")
csv_path = "data/stocks.csv"
top_k = st.sidebar.slider(
    "Number of Tickers",
    1,
    10,
    3,
    help="Select the number of top-ranked securities to analyze.",
)
lookback = st.sidebar.slider(
    "Lookback Window (Days)",
    20,
    120,
    60,
    5,
    help="Historical period for model training and feature extraction.",
)
epochs = st.sidebar.slider(
    "Training Epochs",
    5,
    50,
    10,
    5,
    help="Number of iterations for neural network training.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### Dashboard Overview
    
    **Performance Metrics**
    View ticker rankings based on comprehensive scoring that combines risk-adjusted returns, trend momentum, and drawdown recovery metrics.
    
    **Model Predictions**
    Get AI-powered price forecasts and direction predictions for each ticker, along with model confidence levels.
    
    **Strategy Backtest**
    Compare how the AI-guided trading strategy would have performed historically against a simple buy-and-hold approach.
    
    **Market Analytics**
    Analyze correlations between tickers, volatility trends, and other statistical metrics to understand market dynamics.
    
    ---
    
    ### Supported Securities
    
    • **Microsoft (MSFT)** — Technology & Software
    • **Apple (AAPL)** — Consumer Electronics & Services  
    • **Google (GOOG)** — Digital Advertising & Cloud
    • **Netflix (NFLX)** — Entertainment & Streaming
    """
)

# ============================================================================
# LOAD DATA & RUN PIPELINE
# ============================================================================
with st.spinner("Processing analysis..."):
    try:
        ranking, results = load_pipeline_data(csv_path, top_k, lookback, epochs)
        pipeline_success = True
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        pipeline_success = False

# ============================================================================
# TABS
# ============================================================================
if pipeline_success:
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Metrics",
        "Model Predictions",
        "Strategy Backtest",
        "Market Analytics"
    ])

    # ========================================================================
    # TAB 1: PERFORMANCE METRICS
    # ========================================================================
    with tab1:
        st.subheader("Ticker Performance Rankings")
        
        # Top performer showcase
        top_ticker = ranking.index[0]
        top_score = ranking.loc[top_ticker, "Score"]
        top_trend = ranking.loc[top_ticker, "Trend"]
        top_sharpe = ranking.loc[top_ticker, "SharpeLike"]
        top_drawdown = ranking.loc[top_ticker, "WorstDrawdown"]
        
        st.markdown(f"""
        <div class="showcase-container">
            <div class="showcase-title">⭐ TOP PERFORMER</div>
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div class="ticker-badge">{top_ticker}</div>
            </div>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Composite Score</div>
                    <div class="metric-value">{top_score:.3f}</div>
                    <div class="metric-subtext">Overall Performance</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Trend Momentum</div>
                    <div class="metric-value">{top_trend:+.2%}</div>
                    <div class="metric-subtext">Price Direction</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Risk-Adjusted Return</div>
                    <div class="metric-value">{top_sharpe:.3f}</div>
                    <div class="metric-subtext">Sharpe-Like Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Drawdown Recovery</div>
                    <div class="metric-value">{top_drawdown:.3f}</div>
                    <div class="metric-subtext">Resilience Metric</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### All Tickers Ranking")
        st.markdown("""
        **Scoring Methodology:**
        - **Risk-Adj Return (45%)**: How well returns compensate for volatility
        - **Trend Momentum (35%)**: Current price direction and performance
        - **Drawdown Recovery (20%)**: Ability to bounce back from losses
        """)
        
        # Create HTML table for all rankings
        html_table = '<div class="performance-table"><table><thead><tr><th style="text-align: center; width: 15%; background: #0f3460; color: white; padding: 1rem; text-align: left; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Rank</th><th style="background: #0f3460; color: white; padding: 1rem; text-align: left; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Ticker</th><th style="text-align: right; background: #0f3460; color: white; padding: 1rem; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Score</th><th style="text-align: right; background: #0f3460; color: white; padding: 1rem; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Risk-Adj Return</th><th style="text-align: right; background: #0f3460; color: white; padding: 1rem; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Trend</th><th style="text-align: right; background: #0f3460; color: white; padding: 1rem; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Drawdown Recovery</th></tr></thead><tbody>'
        
        for rank, (ticker, row) in enumerate(ranking.iterrows(), 1):
            score = row['Score']
            sharpe = row['SharpeLike']
            trend = row['Trend']
            drawdown = row['WorstDrawdown']
            
            # Determine score badge color
            if score >= 0.3:
                score_bg = '#16a34a'  # Green
            elif score >= 0:
                score_bg = '#f59e0b'  # Orange
            else:
                score_bg = '#ef4444'  # Red
            
            trend_color = '#16a34a' if trend >= 0 else '#ef4444'
            trend_symbol = '▲' if trend >= 0 else '▼'
            
            html_table += f'<tr style="border-bottom: 1px solid #e2e8f0;"><td style="text-align: center; font-weight: 700; color: #0f3460; padding: 0.9rem 1rem;">#{rank}</td><td style="padding: 0.9rem 1rem; font-weight: 700; color: #0f3460; font-size: 1rem;">{ticker}</td><td style="text-align: right; padding: 0.9rem 1rem;"><span style="display: inline-block; background: {score_bg}; color: white; padding: 0.4rem 0.8rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">{score:.3f}</span></td><td style="text-align: right; padding: 0.9rem 1rem;">{sharpe:.4f}</td><td style="text-align: right; font-weight: 600; padding: 0.9rem 1rem; color: {trend_color};">{trend_symbol} {trend:+.2%}</td><td style="text-align: right; padding: 0.9rem 1rem;">{drawdown:.3f}</td></tr>'
        
        html_table += '</tbody></table></div>'
        
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Summary metrics at bottom
        st.markdown("---")
        st.markdown("### Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tickers", len(ranking), None)

        with col2:
            avg_trend = ranking["Trend"].mean()
            st.metric("Avg Trend", f"{avg_trend:+.2%}")

        with col3:
            avg_sharpe = ranking["SharpeLike"].mean()
            st.metric("Avg Risk-Adj Return", f"{avg_sharpe:.3f}")
        
        with col4:
            best_resilience = ranking["WorstDrawdown"].max()
            st.metric("Best Resilience", f"{best_resilience:.3f}")

    # ========================================================================
    # TAB 2: MODEL PREDICTIONS
    # ========================================================================
    with tab2:
        st.subheader("AI Model Forecasts")
        st.markdown("""
        Each ticker is analyzed with two complementary models:
        - **Price Forecaster**: LSTM neural network predicts next-day closing price
        - **Direction Detector**: Gradient boosting classifier predicts UP/DOWN direction
        """)

        st.markdown("---")

        for idx, (ticker, res) in enumerate(results.items(), 1):
            with st.container():
                # Header
                st.markdown(f"### {idx}. {ticker}")

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)

                current_close = res.get('current_close', 0)
                next_pred = res.get('next_pred_close', 0)
                pct_change = ((next_pred - current_close) / current_close * 100) if current_close > 0 else 0

                with col1:
                    st.metric(
                        "Current Price",
                        f"${current_close:.2f}",
                        delta=None
                    )

                with col2:
                    st.metric(
                        "Predicted Price",
                        f"${next_pred:.2f}",
                        delta=f"{pct_change:+.2f}%",
                        delta_color="inverse"
                    )

                with col3:
                    direction = res.get('direction_signal', 'N/A')
                    confidence = res.get('direction_confidence', 0)
                    direction_indicator = "Up" if direction == "UP" else "Down"
                    st.metric(
                        "Direction Signal",
                        direction_indicator,
                        delta=f"{confidence*100:.1f}% confidence"
                    )

                with col4:
                    clf_acc = res.get('clf_accuracy', 0)
                    st.metric(
                        "Model Accuracy",
                        f"{clf_acc*100:.1f}%"
                    )

                # Ranking metrics
                rank_metrics = res.get('rank_metrics', {})
                caption_text = (
                    f"Overall Score: {rank_metrics.get('Score', 0):.3f} | "
                    f"Risk-Adj Return: {rank_metrics.get('SharpeLike', 0):.4f} | "
                    f"Trend: {rank_metrics.get('Trend', 0):.2%}"
                )
                st.caption(caption_text)

                st.markdown("---")

    # ========================================================================
    # TAB 3: STRATEGY BACKTEST
    # ========================================================================
    with tab3:
        st.subheader("Historical Strategy Performance")
        st.markdown("""
        Backtesting results: Following the direction signals vs. buy-and-hold strategy.
        This shows how the AI-guided trading approach would have performed historically.
        """)

        st.markdown("---")

        for idx, (ticker, res) in enumerate(results.items(), 1):
            st.markdown(f"### {idx}. {ticker}")

            # Get backtest data
            backtest_df = res.get('backtest_df', pd.DataFrame())
            backtest_returns = res.get('backtest_returns', {})

            if not backtest_df.empty:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)

                strategy_ret = backtest_returns.get('strategy_pct', 0)
                buyhold_ret = backtest_returns.get('buyhold_pct', 0)
                outperformance = strategy_ret - buyhold_ret

                with col1:
                    st.metric("AI Strategy Return", f"{strategy_ret:.2f}%")

                with col2:
                    st.metric("Buy & Hold Return", f"{buyhold_ret:.2f}%")

                with col3:
                    delta_color = "off" if abs(outperformance) < 0.01 else "normal"
                    st.metric(
                        "Outperformance",
                        f"{outperformance:+.2f}%",
                        delta_color=delta_color
                    )

                with col4:
                    # Simple Win rate calculation
                    if 'Strategy' in backtest_df.columns and 'BuyHold' in backtest_df.columns:
                        strategy_wins = (backtest_df['Strategy'] > backtest_df['BuyHold']).sum()
                        total_days = len(backtest_df)
                        st.metric("Winning Days", f"{strategy_wins}/{total_days}")

                # Chart
                st.line_chart(
                    backtest_df[['Strategy', 'BuyHold']] if 'Strategy' in backtest_df.columns else backtest_df,
                    use_container_width=True
                )
            else:
                st.info(f"No backtest data available for {ticker}")

            st.markdown("---")

    # ========================================================================
    # TAB 4: MARKET ANALYTICS
    # ========================================================================
    with tab4:
        st.subheader("Market Analytics")

        try:
            # Load raw data
            df = load_stocks(csv_path)
            wide = pivot_close(df)
            rets = wide.pct_change().dropna()

            # Volatility section
            st.markdown("#### Volatility Analysis")
            vol_data = rets.std().sort_values(ascending=False)
            col1, col2 = st.columns([2, 1])

            with col1:
                st.bar_chart(vol_data)

            with col2:
                st.dataframe(
                    vol_data.to_frame("Volatility").round(4),
                    use_container_width=True
                )

            st.markdown("---")

            # Correlation section
            st.markdown("#### Correlation Matrix")
            corr_matrix = rets.corr()

            st.dataframe(
                corr_matrix.style.format("{:.3f}"),
                use_container_width=True
            )

            st.markdown("---")

            # Summary statistics
            st.markdown("#### Summary Statistics")

            summary_stats = pd.DataFrame({
                "Mean Daily Return": rets.mean(),
                "Std Dev": rets.std(),
                "Min": rets.min(),
                "Max": rets.max(),
                "Skewness": rets.skew(),
                "Kurtosis": rets.kurtosis()
            }).round(4)

            st.dataframe(summary_stats, use_container_width=True)

        except Exception as e:
            st.error(f"Analytics Error: {str(e)}")

else:
    st.error("Unable to run analysis. Please check your data and settings.")
