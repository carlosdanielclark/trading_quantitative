# src/utils.py

import yaml
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from decimal import Decimal
import pandas as pd

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configura el sistema de logging centralizado.
    Garantiza que exista la carpeta 'logs/' en la raíz del proyecto.
    """
    # 1) Ubicar la raíz del proyecto (dos niveles arriba de este archivo)
    project_root = Path(__file__).resolve().parent.parent

    # 2) Crear carpeta logs/ si no existe
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 3) Definir archivo de log
    log_file = logs_dir / "trading_system.log"

    # 4) Configurar handlers
    file_handler   = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, stream_handler]
    )

    return logging.getLogger(__name__)

# Logger global
logger = setup_logging()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carga la configuración desde config/config.yaml

    Args:
        config_path: ruta opcional al YAML.
    Returns:
        Diccionario con la configuración.
    Raises:
        FileNotFoundError si no existe el archivo.
        yaml.YAMLError si hay errores de formato.
    """
    # Determinar ruta por defecto
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"No se encontró config.yaml en: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Configuración cargada exitosamente desde: {config_path}")
        return cfg
    except yaml.YAMLError as ye:
        logger.error(f"Error en formato YAML: {ye}")
        raise
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        raise


def date_to_timestamp(date_str: str) -> int:
    """
    Convierte fecha YYYY-MM-DD a timestamp en milisegundos.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError as ve:
        logger.error(f"Error convirtiendo fecha '{date_str}': {ve}")
        raise


def timestamp_to_date(ts: int) -> str:
    """
    Convierte timestamp (ms) a fecha YYYY-MM-DD.
    """
    try:
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError) as e:
        logger.error(f"Error convirtiendo timestamp '{ts}': {e}")
        raise


def validate_symbol(symbol: str) -> bool:
    """
    Valida formato de símbolo de trading (ej. 'BTCUSDT').
    """
    if not isinstance(symbol, str):
        logger.warning(f"Símbolo debe ser string, recibido: {type(symbol)}")
        return False

    valid = len(symbol) >= 6 and symbol.isalnum() and symbol.isupper()
    if not valid:
        logger.warning(f"Símbolo inválido: {symbol}")
    return valid


def validate_interval(interval: str) -> bool:
    """
    Valida intervalo de tiempo (ej. '1h', '1d').
    """
    valid_intervals = [
        '1m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    ]
    if interval not in valid_intervals:
        logger.warning(f"Intervalo inválido: {interval}. Válidos: {valid_intervals}")
        return False
    return True


def safe_float_conversion(
    value: Any,
    default: float = 0.0,
    field_name: str = "unknown"
) -> float:
    """
    Convierte un valor a float de forma segura.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"No se pudo convertir '{field_name}': {value} a float, usando {default}")
        return default


def validate_dataframe_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Valida que el DataFrame contenga todas las columnas requeridas.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error(f"Columnas faltantes en DataFrame: {missing}")
        return False
    logger.info("Validación de columnas exitosa")
    return True


def create_directory_structure(base_path: Path) -> None:
    """
    Crea la estructura de directorios estándar en el proyecto.
    """
    dirs = [
        'data/raw', 'data/processed',
        'logs', 'notebooks',
        'src', 'test', 'config'
    ]
    for d in dirs:
        path = base_path / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {path}")
