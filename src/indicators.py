# src/indicators.py
"""
Sistema de plugins para indicadores técnicos
"""

import pandas as pd
import numpy as np

# Alias claros para los submódulos de ta
import ta.trend       as tt
import ta.momentum    as tm
import ta.volatility  as tv
import ta.volume      as tvl

from typing import Dict, Callable
import logging
logger = logging.getLogger(__name__)

# Registro de indicadores

class IndicatorRegistry:
    """Catálogo global de indicadores técnicos."""
    _indicators: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn: Callable):
            cls._indicators[name] = fn
            logger.info(f"Indicador '{name}' registrado")
            return fn
        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._indicators.get(name)

    @classmethod
    def list_indicators(cls):
        return list(cls._indicators.keys())

# 1) Indicadores de tendencia / medias móviles

@IndicatorRegistry.register("sma")
def sma(df: pd.DataFrame, period: int) -> pd.Series:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para SMA({period})")
        return pd.Series(np.nan, index=df.index)
    return df['close'].rolling(window=period, min_periods=period).mean()


@IndicatorRegistry.register("ema")
def ema(df: pd.DataFrame, period: int) -> pd.Series:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para EMA({period})")
        return pd.Series(np.nan, index=df.index)
    return df['close'].ewm(span=period, adjust=False).mean()


# 2) Osciladores

@IndicatorRegistry.register("rsi")
def rsi(df: pd.DataFrame, period: int, normalize: bool = False) -> pd.Series:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para RSI({period})")
        return pd.Series(np.nan, index=df.index)
    
    close = df['close'].ffill().bfill()
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(span=period, adjust=False).mean()
    ma_down = down.ewm(span=period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)  # evitar división por cero
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.clip(lower=0, upper=100)
    if normalize:
        rsi = (rsi - 0) / (100 - 0)
    return rsi


@IndicatorRegistry.register("stochastic")
def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    min_period = max(k_period, d_period)
    if len(df) < min_period:
        logger.warning(f"Datos insuficientes para Stochastic({k_period},{d_period})")
        return pd.DataFrame(index=df.index)

    stoch = tm.StochasticOscillator(
        df['high'], df['low'], df['close'],
        window=k_period, smooth_window=d_period
    )
    return pd.DataFrame({
        'stoch_k': stoch.stoch(),
        'stoch_d': stoch.stoch_signal()
    }, index=df.index)


# 3) Volatilidad

@IndicatorRegistry.register("bollinger_bands")
def bollinger_bands(df: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para Bollinger Bands({period})")
        return pd.DataFrame(index=df.index)

    bb = tv.BollingerBands(df['close'], window=period, window_dev=std_dev)
    return pd.DataFrame({
        'bb_upper': bb.bollinger_hband(),
        'bb_middle': bb.bollinger_mavg(),
        'bb_lower': bb.bollinger_lband(),
        'bb_width': bb.bollinger_wband(),
        'bb_position': bb.bollinger_pband()
    }, index=df.index)


@IndicatorRegistry.register("atr")
def atr(df: pd.DataFrame, period: int) -> pd.Series:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para ATR({period})")
        return pd.Series(np.nan, index=df.index)
    return tv.average_true_range(df['high'], df['low'], df['close'], window=period)


# 4) Momentum / convergencia

@IndicatorRegistry.register("macd")
def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    min_period = max(fast, slow, signal)
    if len(df) < min_period:
        logger.warning(f"Datos insuficientes para MACD({fast},{slow},{signal})")
        return pd.DataFrame(index=df.index)

    macd_ind = tt.MACD(
        df['close'],
        window_fast=fast,
        window_slow=slow,
        window_sign=signal
    )
    return pd.DataFrame({
        'macd': macd_ind.macd(),
        'macd_signal': macd_ind.macd_signal(),
        'macd_histogram': macd_ind.macd_diff()
    }, index=df.index)


@IndicatorRegistry.register("momentum")  # ROC clásico
def momentum_roc(df: pd.DataFrame, period: int) -> pd.Series:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para Momentum({period})")
        return pd.Series(np.nan, index=df.index)
    return df['close'].pct_change(periods=period) * 100


# 5) Volumen

@IndicatorRegistry.register("volume_sma")
def volume_sma(df: pd.DataFrame, period: int) -> pd.Series:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para Volume SMA({period})")
        return pd.Series(np.nan, index=df.index)
    return df['volume'].rolling(window=period, min_periods=period).mean()


@IndicatorRegistry.register("price_volume_trend")
def price_volume_trend(df: pd.DataFrame) -> pd.Series:
    return tvl.volume_price_trend(df['close'], df['volume'])


# 6) Soporte y resistencia

@IndicatorRegistry.register("support_resistance")
def support_resistance(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    if len(df) < period:
        logger.warning(f"Datos insuficientes para Support/Resistance({period})")
        return pd.DataFrame(index=df.index)

    rmax = df['high'].rolling(window=period).max()
    rmin = df['low'].rolling(window=period).min()
    return pd.DataFrame({
        'resistance': rmax,
        'support':    rmin,
        'range':      rmax - rmin
    }, index=df.index)
