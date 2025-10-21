"""
Orchestrator Module - Intelligent Brain of StockVision AI
Coordinates all pipeline components for end-to-end analysis
"""
from src.utils import load_stocks, pivot_close
from src.features import add_indicators
from src.scoring import score_tickers
from src.models import train_lstm, train_direction_classifier, make_supervised
from src.backtest import backtest_long_only
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def run_pipeline(csv_path="data/stocks.csv", top_k=2, lookback=60, epochs=10):
    """
    Main orchestrator function - the "brain" of StockVision AI
    
    Executes complete pipeline:
    1. Load and pivot stock data
    2. Score and rank tickers
    3. For top tickers:
       - Train LSTM for price prediction
       - Train direction classifier
       - Run backtest
    
    Args:
        csv_path: Path to stocks CSV file
        top_k: Number of top tickers to analyze (default: 2)
        lookback: LSTM lookback window (default: 60)
        epochs: LSTM training epochs (default: 10)
    
    Returns:
        Tuple of (ranking DataFrame, results dict)
        ranking: Ticker scores sorted by composite score
        results: Dict with predictions and backtest for each top ticker
    """
    print("\n" + "="*80)
    print("🧠 STOCKVISION AI ORCHESTRATOR - RUNNING FULL PIPELINE")
    print("="*80)
    
    # Step 1: Load and pivot data
    print(f"\n📊 Step 1: Loading stocks from {csv_path}...")
    df = load_stocks(csv_path)
    print(f"   ✓ Loaded {len(df)} records for {df['Ticker'].nunique()} tickers")
    
    wide = pivot_close(df)
    print(f"   ✓ Pivoted to {wide.shape[0]} dates × {wide.shape[1]} tickers")
    
    # Step 2: Score and rank tickers
    print(f"\n📈 Step 2: Scoring and ranking tickers...")
    ranking = score_tickers(wide)
    print(f"   ✓ Rankings computed")
    print(f"\n   Ranking Results:")
    print(ranking.to_string())
    
    top = ranking.head(top_k).index.tolist()
    print(f"\n   🏆 Top {top_k} tickers selected: {top}")
    
    # Step 3: Analyze each top ticker
    results = {}
    print(f"\n🤖 Step 3: Training models for top tickers...")
    print("="*80)
    
    for ticker in top:
        print(f"\n🔹 Analyzing {ticker}...")
        print("-"*80)
        
        # Get data for this ticker
        dft = df[df["Ticker"] == ticker].sort_values("Date").copy()
        print(f"   Data: {len(dft)} trading days")
        
        # Add technical indicators
        feats = add_indicators(dft[["Close"]].copy())
        print(f"   Features: {len(feats)} after NaN removal")
        
        # Train LSTM for price prediction
        print(f"   Training LSTM (lookback={lookback}, epochs={epochs})...")
        model, scaler = train_lstm(dft["Close"], lookback=lookback, epochs=epochs)
        print(f"   ✓ LSTM trained")
        
        # Predict next close price
        scaled = scaler.transform(dft["Close"].values.reshape(-1, 1)).ravel()
        if len(scaled) >= lookback:
            X_last = np.array([scaled[-lookback:]])[..., None]
            yhat_scaled = model.predict(X_last, verbose=0)[0, 0]
            next_pred = scaler.inverse_transform([[yhat_scaled]])[0, 0]
            current_close = dft["Close"].iloc[-1]
            pred_direction = "↑ UP" if next_pred > current_close else "↓ DOWN"
            price_diff = (next_pred - current_close) / current_close * 100
            print(f"   Next-day prediction: ${next_pred:.2f} {pred_direction} ({price_diff:+.2f}%)")
        else:
            next_pred = dft["Close"].iloc[-1]
            print(f"   ⚠️  Insufficient data for prediction")
        
        # Train direction classifier
        print(f"   Training GradientBoosting classifier...")
        clf, acc = train_direction_classifier(feats)
        print(f"   ✓ Classifier trained (accuracy: {acc:.1%})")
        
        # Get last features for direction prediction
        X_last_feats = feats[["ret", "ma10", "ma20", "rsi14", "vol20"]].iloc[-1:]
        if len(X_last_feats) > 0:
            try:
                direction_pred = clf.predict(X_last_feats)[0]
                direction_conf = clf.predict_proba(X_last_feats)[0].max()
                direction = "↑ UP" if direction_pred == 1 else "↓ DOWN"
                print(f"   Direction signal: {direction} ({direction_conf:.1%} confidence)")
            except:
                direction = "N/A"
                direction_conf = 0
        
        # Simple backtest (using placeholder predictions for demo)
        print(f"   Running backtest...")
        close_series = dft["Close"].reset_index(drop=True)
        preds_series = pd.Series([next_pred] * len(close_series))
        bt = backtest_long_only(close_series, preds_series)
        
        final_strat = bt["Strategy"].iloc[-1]
        final_bh = bt["BuyHold"].iloc[-1]
        strat_return = (final_strat - 1) * 100
        bh_return = (final_bh - 1) * 100
        print(f"   Strategy: {strat_return:+.2f}% | Buy&Hold: {bh_return:+.2f}%")
        
        # Store results
        results[ticker] = {
            "rank_metrics": ranking.loc[ticker].to_dict(),
            "current_close": float(current_close),
            "next_pred_close": float(next_pred),
            "lstm_model": model,
            "scaler": scaler,
            "clf_accuracy": float(acc),
            "direction_signal": direction,
            "direction_confidence": float(direction_conf),
            "backtest_returns": {
                "strategy_pct": float(strat_return),
                "buyhold_pct": float(bh_return),
                "outperformance": float(strat_return - bh_return)
            },
            "backtest_df": bt
        }
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    
    return ranking, results


def print_summary(ranking: pd.DataFrame, results: Dict):
    """
    Print a nice summary of results
    
    Args:
        ranking: Ticker ranking DataFrame
        results: Results dictionary from run_pipeline
    """
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    print("\n🏆 TICKER RANKINGS:")
    print("-"*80)
    for ticker in ranking.index:
        score = ranking.loc[ticker, "Score"]
        trend = ranking.loc[ticker, "Trend"]
        sharpe = ranking.loc[ticker, "SharpeLike"]
        print(f"   {ticker:6s} | Score: {score:7.3f} | Trend: {trend:+7.2%} | Sharpe: {sharpe:7.3f}")
    
    if results:
        print("\n🤖 MODEL PREDICTIONS:")
        print("-"*80)
        for ticker in results:
            curr = results[ticker]["current_close"]
            pred = results[ticker]["next_pred_close"]
            direction = results[ticker]["direction_signal"]
            conf = results[ticker]["direction_confidence"]
            returns = results[ticker]["backtest_returns"]
            
            print(f"\n   {ticker}:")
            print(f"      Current:    ${curr:.2f}")
            print(f"      Predicted:  ${pred:.2f}")
            print(f"      Direction:  {direction} ({conf:.1%})")
            print(f"      Strategy:   {returns['strategy_pct']:+.2f}%")
            print(f"      Buy&Hold:   {returns['buyhold_pct']:+.2f}%")


if __name__ == "__main__":
    # Example usage
    ranking, results = run_pipeline(
        csv_path="data/stocks.csv",
        top_k=2,
        lookback=10,
        epochs=5
    )
    print_summary(ranking, results)
