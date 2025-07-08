"""
Tests para backtesting
"""

import sys
import os

# Añade la raíz del proyecto al sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtest import BacktestEngine, Trade, BacktestResults

def make_data(n=100, start_price=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = start_price + np.cumsum(np.random.normal(0, 1, n))
    return pd.DataFrame({"close": prices}, index=idx)

def make_signals(n=100, freq=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    signals = np.zeros(n)
    signals[::freq] = 1
    signals[1::freq] = -1
    return pd.Series(signals, index=idx)

@pytest.fixture
def engine():
    config = {
        'backtest': {
            'initial_capital': 10000,
            'commission': 0.001,
            'spread': 0.0,
            'slippage': 0.0,
        }
    }
    return BacktestEngine(config)

def test_execution_with_spread_slippage(engine):
    data = make_data(20, 100)
    signals = make_signals(20, 5)
    # Ejecuta sin spread/slippage
    engine.commission = 0.002
    engine.spread = 0.0
    engine.slippage = 0.0
    results_no_cost = engine.run_backtest(data, signals, position_size=0.2)
    pnl_no_cost = sum([t.pnl for t in results_no_cost.trades if not t.is_open])

    # Ejecuta con spread/slippage altos
    engine.spread = 1.0
    engine.slippage = 0.5
    results_cost = engine.run_backtest(data, signals, position_size=0.2)
    pnl_cost = sum([t.pnl for t in results_cost.trades if not t.is_open])

    # El P&L total debe ser menor (más negativo o menos positivo) con costos altos
    assert pnl_cost < pnl_no_cost

def test_metric_calculation(engine):
    data = make_data(60, 100)
    signals = make_signals(60, 10)
    results = engine.run_backtest(data, signals, position_size=0.1)
    metrics = engine.get_performance_metrics(results)
    for k in ['total_trades', 'winning_trades', 'losing_trades', 'win_rate',
              'total_return', 'max_drawdown', 'sharpe_ratio', 'profit_factor',
              'initial_capital', 'final_capital']:
        assert k in metrics
        assert isinstance(metrics[k], (int, float, np.integer, np.floating))
    assert np.isfinite(metrics['sharpe_ratio'])

def test_distribution_and_winrate(engine):
    data = make_data(40, 100)
    signals = make_signals(40, 8)
    results = engine.run_backtest(data, signals, position_size=0.15)
    assert 0.0 <= results.win_rate <= 100.0
    assert results.total_trades == results.winning_trades + results.losing_trades

def test_lookahead_bias_detection(engine, caplog):
    data = make_data(15, 100)
    signals = make_signals(15, 5)
    results = engine.run_backtest(data, signals, position_size=0.2)
    for t in results.trades:
        if not t.is_open:
            assert t.entry_time < t.exit_time

def test_equity_curve_and_drawdown(engine):
    data = make_data(30, 100)
    signals = make_signals(30, 6)
    results = engine.run_backtest(data, signals, position_size=0.1)
    # Equity curve debe tener misma longitud que timestamps
    assert len(results.equity_curve) == len(results.equity_curve.index)
    # Drawdown no debe ser positivo
    assert results.max_drawdown <= 0.0
    # El primer valor de la equity curve nunca debe ser mayor que el capital inicial
    assert results.equity_curve.iloc[0] <= engine.initial_capital + 1e-6

def test_walkforward_analysis(engine):
    data = make_data(120, 100)
    signals = make_signals(120, 12)
    results = engine.run_backtest(data, signals, position_size=0.1)
    walk_metrics = engine._walk_forward_analysis(results.equity_curve, window=30)
    assert isinstance(walk_metrics, list)
    assert len(walk_metrics) >= 3
    for wm in walk_metrics:
        assert 'return' in wm and 'drawdown' in wm

def test_pdf_report_generation(tmp_path, engine):
    data = make_data(30, 100)
    signals = make_signals(30, 6)
    results = engine.run_backtest(data, signals, position_size=0.1)
    metrics = engine.get_performance_metrics(results)
    pdf_path = tmp_path / "report.pdf"
    engine.generate_pdf_report(results, metrics, filename=str(pdf_path))
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 100
