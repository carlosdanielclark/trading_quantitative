# test/strategy.py

import sys
import os

# Añade la raíz del proyecto al sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

import pytest
import pandas as pd
from src.strategy import (
    build_target, windowed_train_predict,
    generate_signals, manage_positions
)

def test_build_target():
    df = pd.DataFrame({'close': [100, 101, 99, 102, 102]})
    target = build_target(df)
    assert list(target) == [1, -1, 1, -1]

def test_windowed_train_predict():
    X = pd.DataFrame({'f1': range(50), 'f2': range(50, 100)})
    y = pd.Series([1 if i % 2 == 0 else -1 for i in range(50)])
    preds, metrics = windowed_train_predict(X, y, {'n_estimators': 10, 'max_depth': 3}, window_size=10)
    assert len(preds.dropna()) > 0
    assert 'accuracy' in metrics

def test_generate_signals_and_positions():
    preds = pd.Series([1, 1, -1, -1, 1, 0, -1])
    signals = generate_signals(preds)
    positions = manage_positions(signals)
    # Solo una posición abierta, se cierra al cambiar la señal
    assert positions.iloc[0] == 1
    assert positions.iloc[2] == -1
    assert positions.iloc[5] == 0

