# trading_quant_project/src/data_fetcher.py

import requests
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional
import logging

from .utils import load_config, date_to_timestamp, timestamp_to_date

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataFetcher")

class DataFetcher:
    def __init__(self, config: Optional[Dict] = None):
        """Inicializa el fetcher con la configuración del proyecto"""
        self.config = config or load_config()['data_fetcher']
        self.symbol = self.config['symbol']
        self.timeframe = self.config['timeframe']
        self.api_config = self.config['api']
        self.storage_config = self.config['storage']

        # Crear directorios si no existen
        Path(self.storage_config['raw_path']).mkdir(parents=True, exist_ok=True)
        logger.info(f"DataFetcher inicializado para {self.symbol}/{self.timeframe}")

    def _get_api_url(self) -> str:
        """Construye la URL completa del endpoint"""
        return f"{self.api_config['base_url']}{self.api_config['endpoint']}"

    def _generate_filename(self, start_date: str, end_date: str) -> str:
        """Genera nombre de archivo basado en parámetros"""
        return f"{self.symbol}_{self.timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"

    def fetch_ohlcv(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtiene datos OHLCV desde la API con manejo de paginación"""
        # Verificar caché primero
        filename = self._generate_filename(start_date, end_date)
        filepath = Path(self.storage_config['raw_path']) / filename

        if self.storage_config['cache_enabled'] and filepath.exists():
            logger.info(f"Usando datos en caché: {filepath}")
            return pd.read_parquet(filepath)

        logger.info(f"Descargando datos desde {start_date} hasta {end_date}")

        # Convertir fechas a timestamp (milisegundos)
        start_ts = date_to_timestamp(start_date)
        end_ts = date_to_timestamp(end_date)

        all_data = []
        current_end = end_ts
        max_records = 1000  # Máximo de registros por petición

        while current_end > start_ts:
            # Calcular inicio del segmento actual
            current_start = max(start_ts, current_end - max_records * self._timeframe_to_ms())
            # Construir parámetros de la solicitud
            params = {
                'symbol': self.symbol,
                'interval': self.timeframe,
                'startTime': current_start,
                'endTime': current_end,
                'limit': max_records
            }

            # Realizar solicitud con manejo de errores
            for attempt in range(self.api_config['max_retries']):
                try:
                    response = requests.get(
                        self._get_api_url(),
                        params=params,
                        timeout=10
                    )
                    response.raise_for_status()
                    data = response.json()
                    # Binance devuelve una lista directamente
                    if not isinstance(data, list):
                        raise ValueError("Respuesta inesperada de la API")
                    # Convertir a DataFrame
                    df_segment = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    # Seleccionar solo las columnas necesarias
                    df_segment = df_segment[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    all_data.append(df_segment)
                    logger.info(f"Descargado segmento: {timestamp_to_date(current_start)} - {timestamp_to_date(current_end)}")
                    break
                except (requests.RequestException, ValueError) as e:
                    logger.warning(f"Intento {attempt+1} fallido: {str(e)}")
                    if attempt < self.api_config['max_retries'] - 1:
                        time.sleep(2 ** attempt)  # Espera exponencial
                    else:
                        logger.error(f"Error al descargar datos: {str(e)}")
                        raise

            # Actualizar para el próximo segmento
            current_end = current_start - self._timeframe_to_ms()
            # Respetar límite de tasa
            time.sleep(1 / self.api_config['rate_limit'])

        # Combinar todos los segmentos
        df = pd.concat(all_data, ignore_index=True).sort_values('timestamp')
        df = df.drop_duplicates('timestamp').reset_index(drop=True)
        # Convertir timestamp a datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('datetime')
        # Guardar en caché
        df.to_parquet(filepath)
        logger.info(f"Datos guardados en: {filepath}")
        return df

    def _timeframe_to_ms(self) -> int:
        """Convierte timeframe a milisegundos"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(self.timeframe, 60 * 60 * 1000)  # Default 1h

    def update_data(self):
        """Actualiza los datos existentes con nueva información"""
        # Implementación de actualización incremental pendiente
        pass

# Función auxiliar para uso externo
def fetch_and_save_data():
    """Función principal para ejecutar la extracción de datos"""
    config = load_config()['data_fetcher']
    fetcher = DataFetcher(config)
    # Descargar datos para el rango configurado
    df = fetcher.fetch_ohlcv(
        start_date=config['start_date'],
        end_date=config['end_date']
    )
    logger.info(f"Datos descargados exitosamente. Registros: {len(df)}")
    return df
