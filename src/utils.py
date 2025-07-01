# trading_quant_project/src/utils.py

import yaml
from pathlib import Path
from datetime import datetime

def load_config() -> dict:
    """Carga la configuración desde config.yaml"""
    # Obtener la ruta absoluta del directorio del proyecto
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    if not config_path.is_file():
        raise FileNotFoundError(f"No se encontró config.yaml en: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def date_to_timestamp(date_str: str) -> int:
    """Convierte fecha YYYY-MM-DD a timestamp en milisegundos"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

def timestamp_to_date(ts: int) -> str:
    """Convierte milisegundos desde epoch a 'YYYY-MM-DD'."""
    # Asegúrate de usar utc si todo lo manejas en UTC:
    dt = datetime.utcfromtimestamp(ts / 1000)
    return dt.strftime("%Y-%m-%d")