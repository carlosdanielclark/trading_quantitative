import requests
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional, List
import logging
import hashlib
import hmac
from src.utils import load_config, date_to_timestamp, timestamp_to_date
import random
import string

# Configuración del logging
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
        self.interval = self.config['interval']
        self.api_config = self.config['api']
        self.storage_config = self.config['storage']
        self.base_url = self.api_config['base_url']
        
        # Para endpoints públicos no necesitamos headers de autenticación
        self.headers = {
            'Content-Type': 'application/json'
        }
        
        # Solo añadir api-key si está configurada (no es requerida para klines)
        if self.api_config.get('key'):
            self.headers['api-key'] = self.api_config['key']
        
        self.rate_limit = self.api_config.get('rate_limit', 10)
        self.timeout = self.api_config.get('timeout', 30)
        self.max_limit = 200  # Límite máximo según documentación oficial
        
        Path(self.storage_config['raw_path']).mkdir(parents=True, exist_ok=True)
        logger.info(f"DataFetcher inicializado para {self.symbol}/{self.interval}")

    def fetch_ohlcv(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Método principal para obtener datos OHLCV"""
        return self.fetch_data(start_date, end_date)

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Método para descargar los datos con lógica de paginación automática"""
        estimated_records = self.estimate_records(start_date, end_date)
        
        if estimated_records <= self.max_limit:
            # Si el rango no excede el límite, una sola request
            return self.fetch_with_single_request(start_date, end_date)
        else:
            # Si el rango excede el límite, usar paginación
            return self.fetch_with_pagination(start_date, end_date)

    def estimate_records(self, start_date: str, end_date: str) -> int:
        """Estima el número de registros basado en fechas e intervalo"""
        start_ts = pd.to_datetime(start_date).timestamp() * 1000
        end_ts = pd.to_datetime(end_date).timestamp() * 1000
        
        # Calcular el número de registros basado en el intervalo
        interval_ms = self._timeframe_to_ms()
        total_ms = end_ts - start_ts
        estimated_records = int(total_ms / interval_ms)
        
        logger.info(f"Registros estimados: {estimated_records}")
        return estimated_records

    def fetch_with_single_request(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtiene datos en una sola request para rangos pequeños"""
        logger.info(f"Descargando datos sin paginación: {start_date} - {end_date}")
        
        start_ts = date_to_timestamp(start_date)
        end_ts = date_to_timestamp(end_date)

        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': min(self.max_limit, self.estimate_records(start_date, end_date))
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/futures/market/kline", 
                params=params, 
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Verificar formato de respuesta de Bitunix
            if data.get('code') == 0 and 'data' in data:
                return self._process_bitunix_response(data['data'])
            else:
                logger.error(f"Error en respuesta de API: {data}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en request: {e}")
            return pd.DataFrame()

    def fetch_with_pagination(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtiene datos con paginación para rangos grandes"""
        logger.info(f"Descargando datos con paginación: {start_date} - {end_date}")
        
        start_ts = date_to_timestamp(start_date)
        end_ts = date_to_timestamp(end_date)
        
        all_data = []
        current_start = start_ts
        interval_ms = self._timeframe_to_ms()
        
        while current_start < end_ts:
            # Calcular el end time para este chunk
            chunk_end = min(current_start + (self.max_limit * interval_ms), end_ts)
            
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'startTime': current_start,
                'endTime': chunk_end,
                'limit': self.max_limit
            }
            
            try:
                response = requests.get(
                    f"{self.base_url}/api/v1/futures/market/kline",
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('code') == 0 and 'data' in data:
                    df_chunk = self._process_bitunix_response(data['data'])
                    if not df_chunk.empty:
                        all_data.append(df_chunk)
                    
                    # Si recibimos menos datos de los esperados, hemos llegado al final
                    if len(data['data']) < self.max_limit:
                        break
                        
                else:
                    logger.error(f"Error en chunk: {data}")
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error en request de chunk: {e}")
                break
            
            # Actualizar para el siguiente chunk
            current_start = chunk_end + interval_ms
            
            # Respetar rate limit
            time.sleep(1 / self.rate_limit)
        
        if all_data:
            # CORRECCIÓN: Concatenar manteniendo el índice datetime
            df = pd.concat(all_data, ignore_index=False)
            df = df.drop_duplicates().sort_index().reset_index()
            
            # Restablecer datetime como índice después de concatenar
            df = df.set_index('datetime')
            
            return df
        else:
            return pd.DataFrame()

    def _process_bitunix_response(self, data: List[Dict]) -> pd.DataFrame:
        """Procesa la respuesta de Bitunix al formato DataFrame estándar"""
        if not data:
            return pd.DataFrame()
        
        # Crear DataFrame con el formato de respuesta de Bitunix
        df_data = []
        for item in data:
            df_data.append({
                'timestamp': item['time'],
                'open': float(item['open']),
                'high': float(item['high']),
                'low': float(item['low']),
                'close': float(item['close']),
                'volume': float(item['baseVol'])
            })
        
        df = pd.DataFrame(df_data)
        
        # CORRECCIÓN: Añadir columna datetime y establecer como índice
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('datetime')
        
        logger.info(f"Procesados {len(df)} registros")
        return df

    def _timeframe_to_ms(self) -> int:
        """Convierte el intervalo a milisegundos"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(self.interval, 60 * 60 * 1000)

    def _generate_signature(self, params: Dict) -> tuple:
        """
        Genera firma para endpoints privados (no necesario para klines)
        Solo se usa si tienes configuradas las claves API
        """
        if not self.api_config.get('secret'):
            raise ValueError("Secret key no configurado para autenticación")
            
        secret_key = self.api_config['secret']
        timestamp = str(int(time.time() * 1000))
        nonce = ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(32)])

        # Crear query string ordenado
        params_str = '&'.join([f"{key}={value}" for key, value in sorted(params.items())])
        
        # Primer hash: nonce + timestamp + api_key + query_params + body
        digest_input = f"{nonce}{timestamp}{self.api_config['key']}{params_str}"
        digest = hashlib.sha256(digest_input.encode()).hexdigest()
        
        # Segundo hash: digest + secret_key
        sign_input = digest + secret_key
        signature = hashlib.sha256(sign_input.encode()).hexdigest()

        return nonce, timestamp, signature

    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Guarda los datos en el formato especificado"""
        if df.empty:
            logger.warning("No hay datos para guardar")
            return ""
            
        if filename is None:
            filename = f"{self.symbol}_{self.interval}_{self.config['start_date']}_{self.config['end_date']}"
        
        file_format = self.storage_config['file_format']
        raw_path = Path(self.storage_config['raw_path'])
        
        if file_format == 'parquet':
            filepath = raw_path / f"{filename}.parquet"
            df.to_parquet(filepath)
        elif file_format == 'csv':
            filepath = raw_path / f"{filename}.csv"
            df.to_csv(filepath)
        else:
            raise ValueError(f"Formato no soportado: {file_format}")
        
        logger.info(f"Datos guardados en: {filepath}")
        return str(filepath)

# Función para usar externamente
def fetch_and_save_data():
    """Función principal para obtener y guardar datos"""
    config = load_config()['data_fetcher']
    fetcher = DataFetcher(config)
    
    df = fetcher.fetch_ohlcv(
        start_date=config['start_date'], 
        end_date=config['end_date']
    )
    
    if not df.empty:
        filepath = fetcher.save_data(df)
        logger.info(f"Datos descargados exitosamente. Registros: {len(df)}")
        logger.info(f"Archivo guardado: {filepath}")
        return df
    else:
        logger.error("No se pudieron obtener datos")
        return None
