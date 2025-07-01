# trading_quant_project/src/utils.py

import yaml
from pathlib import Path
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Carga la configuración desde config.yaml"""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    if not config_path.is_file():
        raise FileNotFoundError(f"No se encontró config.yaml en: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Configuración cargada exitosamente")
        return config
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        raise

def date_to_timestamp(date_str: str) -> int:
    """Convierte fecha YYYY-MM-DD a timestamp en milisegundos"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Usar UTC para consistencia
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError as e:
        logger.error(f"Error convirtiendo fecha {date_str}: {e}")
        raise

def timestamp_to_date(ts: int) -> str:
    """Convierte timestamp en milisegundos a fecha YYYY-MM-DD"""
    try:
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError) as e:
        logger.error(f"Error convirtiendo timestamp {ts}: {e}")
        raise

def validate_symbol(symbol: str) -> bool:
    """Valida formato de símbolo de trading"""
    if not isinstance(symbol, str):
        return False
    # Símbolos típicos: BTCUSDT, ETHUSDT, etc.
    return len(symbol) >= 6 and symbol.isalnum() and symbol.isupper()

def validate_interval(interval: str) -> bool:
    """Valida intervalo de tiempo"""
    valid_intervals = [
        '1m', '5m', '15m', '30m', '1h', '2h', '4h', 
        '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    ]
    return interval in valid_intervals

def safe_float_conversion(value, default=0.0):
    """Conversión segura a float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"No se pudo convertir {value} a float, usando {default}")
        return default
