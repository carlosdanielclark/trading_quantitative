import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import yaml
from src.data_fetcher import DataFetcher

def load_config() -> dict:
    """Carga la configuración de prueba"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    if not config_path.is_file():
        raise FileNotFoundError(f"No hay config.yaml aquí: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class DummyResponse:
    """Clase auxiliar para simular respuestas HTTP"""
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
    
    def json(self):
        # Formato de respuesta de Bitunix
        return {
            "code": 0,
            "data": self._data,
            "msg": "Success"
        }
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

def test_generate_signature():
    """Test de generación de firma solo si hay claves configuradas"""
    config = load_config()['data_fetcher']
    fetcher = DataFetcher(config)

    # Solo testear si hay secret configurado
    if not config['api'].get('secret'):
        pytest.skip("No hay secret configurado - test omitido")
    
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'startTime': 1633046400000,
        'endTime': 1633132800000,
        'limit': 200
    }

    nonce, timestamp, signature = fetcher._generate_signature(params)
    
    # Verificaciones básicas
    assert len(nonce) == 32
    assert timestamp.isdigit()
    assert len(signature) == 64  # SHA256 hex = 64 caracteres
    assert isinstance(signature, str)

def test_process_bitunix_response():
    """Test del procesamiento de respuesta en formato Bitunix"""
    # Datos simulados en formato Bitunix
    sample_data = [
        {
            "open": 60000,
            "high": 60100,
            "close": 60050,
            "low": 59900,
            "time": 1622505600000,
            "quoteVol": "1000000",
            "baseVol": "16.67",
            "type": "LAST_PRICE"
        },
        {
            "open": 60050,
            "high": 60200,
            "close": 60150,
            "low": 60000,
            "time": 1622509200000,
            "quoteVol": "1200000",
            "baseVol": "20.0",
            "type": "LAST_PRICE"
        }
    ]

    # Mock de la respuesta completa
    def fake_get(url, params=None, headers=None, timeout=None):
        return DummyResponse(status_code=200, data=sample_data)

    # Patch del requests.get
    with patch('src.data_fetcher.requests.get', side_effect=fake_get):
        config = load_config()['data_fetcher']
        fetcher = DataFetcher(config)

        # Ejecutar el método
        df = fetcher.fetch_ohlcv(start_date="2021-06-01", end_date="2021-06-02")
        
        # Verificaciones
        assert not df.empty
        assert len(df) == 2
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Verificar valores específicos
        assert df.iloc[0]['open'] == 60000
        assert df.iloc[0]['high'] == 60100
        assert df.iloc[0]['close'] == 60050
        assert df.iloc[0]['low'] == 59900
        assert df.iloc[0]['volume'] == 16.67
        
        # Verificar que el timestamp está convertido correctamente
        assert df.iloc[0]['timestamp'] == 1622505600000

def test_pagination_logic():
    """Test de la lógica de paginación"""
    sample_data = [
        {
            "open": 60000 + i,
            "high": 60100 + i,
            "close": 60050 + i,
            "low": 59900 + i,
            "time": 1622505600000 + (i * 3600000),  # +1 hora por registro
            "quoteVol": "1000000",
            "baseVol": "16.67",
            "type": "LAST_PRICE"
        }
        for i in range(5)
    ]

    def fake_get(url, params=None, headers=None, timeout=None):
        return DummyResponse(status_code=200, data=sample_data)

    with patch('src.data_fetcher.requests.get', side_effect=fake_get):
        config = load_config()['data_fetcher']
        fetcher = DataFetcher(config)

        # Test con rango que fuerza paginación
        df = fetcher.fetch_ohlcv(start_date="2021-01-01", end_date="2021-12-31")
        
        # Verificaciones mejoradas
        assert not df.empty
        assert len(df) >= 5  # Al menos los datos de prueba
        
        # Verificar que tiene índice datetime O columna datetime
        has_datetime_index = isinstance(df.index, pd.DatetimeIndex) or df.index.name == 'datetime'
        has_datetime_column = 'datetime' in df.columns
        
        assert has_datetime_index or has_datetime_column, f"DataFrame debe tener datetime como índice o columna. Índice: {df.index}, Columnas: {df.columns}"
        
        # Verificar columnas obligatorias
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns, f"Columna requerida '{col}' no encontrada"

def test_timeframe_conversion():
    """Test de conversión de intervalos a milisegundos"""
    config = load_config()['data_fetcher']
    fetcher = DataFetcher(config)
    
    # Verificar conversiones estándar
    assert fetcher._timeframe_to_ms() == 3600000  # 1h por defecto del config
    
    # Cambiar intervalo temporalmente para testear
    fetcher.interval = '1m'
    assert fetcher._timeframe_to_ms() == 60000
    
    fetcher.interval = '1d'
    assert fetcher._timeframe_to_ms() == 86400000

def test_error_handling():
    """Test del manejo de errores"""
    def fake_error_get(url, params=None, headers=None, timeout=None):
        response = MagicMock()
        response.json.return_value = {"code": 1, "msg": "Error", "data": None}
        response.raise_for_status.return_value = None
        return response

    with patch('src.data_fetcher.requests.get', side_effect=fake_error_get):
        config = load_config()['data_fetcher']
        fetcher = DataFetcher(config)

        df = fetcher.fetch_ohlcv(start_date="2022-01-01", end_date="2022-01-02")
        
        # Debe retornar DataFrame vacío en caso de error
        assert df.empty

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
