import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from src.utils import load_config

logger = logging.getLogger(__name__)

def load_features(features_path: str) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    df = df.sort_index()
    df = df.dropna()
    logger.info(f"Features cargados desde {features_path} con shape {df.shape}")
    return df

def build_target(df: pd.DataFrame) -> pd.Series:
    """Construye la etiqueta de dirección del mercado."""
    close = df['close']
    target = np.where(close.shift(-1) > close, 1, -1)
    target = pd.Series(target, index=close.index)
    target = target[:-1]  # Elimina el último registro (sin futuro)
    return target

def windowed_train_predict(X: pd.DataFrame, y: pd.Series, model_params: Dict[str, Any], window_size: Optional[int] = None) -> Tuple[pd.Series, Dict[str, Any]]:
    """Entrena y predice usando ventana deslizante o todo el set si window_size es None."""
    preds = pd.Series(index=y.index, dtype=int)
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {}

    if window_size:
        for start in range(0, len(X) - window_size):
            end = start + window_size
            X_train, y_train = X.iloc[start:end], y.iloc[start:end]
            X_test = X.iloc[[end]]
            model = RandomForestClassifier(**model_params, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            preds.iloc[end] = pred[0]
        logger.info("Predicción con ventana móvil finalizada.")
    else:
        model = RandomForestClassifier(**model_params, random_state=42)
        model.fit(X, y)
        preds[:] = model.predict(X)
        logger.info("Predicción sobre todo el set finalizada.")

    # Métricas simples (solo para registros predichos)
    valid_idx = preds.dropna().index
    y_valid = y.loc[valid_idx]
    y_pred = preds.loc[valid_idx]
    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, average="macro"),
        "recall": recall_score(y_valid, y_pred, average="macro"),
        "f1": f1_score(y_valid, y_pred, average="macro"),
    }
    logger.info(f"Métricas de validación: {metrics}")
    return preds, metrics

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, Any]) -> Dict[str, Any]:
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    logger.info(f"Mejores parámetros encontrados: {grid.best_params_}")
    return grid.best_params_

def generate_signals(preds: pd.Series) -> pd.Series:
    """Genera señales de trading a partir de las predicciones del modelo."""
    return preds

def manage_positions(signals: pd.Series) -> pd.Series:
    """
    Gestiona las posiciones: abre/cierra en el mismo tick cuando la señal cambia.
    - +1: posición larga (compra)
    - -1: posición corta (venta)
    - 0: sin posición
    """
    position = 0
    positions = []
    for signal in signals:
        if signal == 0:
            position = 0
        elif signal != position:
            position = signal
        # Si signal == position, se mantiene la posición
        positions.append(position)
    return pd.Series(positions, index=signals.index)


def export_results(signals: pd.Series, positions: pd.Series, metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
    Path(config['randomforest']['export_signals_path']).parent.mkdir(parents=True, exist_ok=True)
    Path(config['randomforest']['export_positions_path']).parent.mkdir(parents=True, exist_ok=True)
    Path(config['randomforest']['export_metrics_path']).parent.mkdir(parents=True, exist_ok=True)
    signals.to_frame('signal').to_parquet(config['randomforest']['export_signals_path'])
    positions.to_frame('position').to_parquet(config['randomforest']['export_positions_path'])
    pd.DataFrame([metrics]).to_csv(config['randomforest']['export_metrics_path'], index=False)
    logger.info("Resultados exportados correctamente.")

def main(config_path: Optional[str] = None):
    config = load_config(config_path)
    rf_cfg = config.get('randomforest', {})

    # Cargar features y construir target
    features = load_features(rf_cfg['features_path'])
    # Seleccionar solo columnas de indicadores (excluyendo 'close' si no es feature)
    X = features.drop(columns=[c for c in ['close', 'open', 'high', 'low', 'volume'] if c in features.columns])
    y = build_target(features)
    X, y = X.loc[y.index], y  # Alinear índices

    # Optimización de hiperparámetros si corresponde
    if rf_cfg.get('grid_search', False):
        best_params = optimize_hyperparameters(X, y, rf_cfg['param_grid'])
    else:
        best_params = {k: rf_cfg[k] for k in ['n_estimators', 'max_depth'] if k in rf_cfg}

    # Entrenamiento y predicción
    preds, metrics = windowed_train_predict(
        X, y, best_params, window_size=rf_cfg.get('window_size')
    )

    # Generar señales y posiciones
    signals = generate_signals(preds)
    positions = manage_positions(signals)

    # Exportar resultados
    export_results(signals, positions, metrics, config)

if __name__ == "__main__":
    main()
