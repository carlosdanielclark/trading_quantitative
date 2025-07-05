"""
Tests para funciones de configuración
"""
import sys
import os

# Añade la raíz del proyecto al sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
    
import pytest
import yaml
from pathlib import Path
import tempfile
import os

from src.utils import load_config, validate_symbol, validate_interval

class TestConfig:
    
    def test_load_config_success(self):
        """Test de carga exitosa de configuración"""
        # Crear archivo de configuración temporal
        config_data = {
            'project': {'name': 'Test Project'},
            'data_fetcher': {
                'symbol': 'BTCUSDT',
                'interval': '1h'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Cargar configuración
            config = load_config(temp_path)
            
            # Verificar contenido
            assert config['project']['name'] == 'Test Project'
            assert config['data_fetcher']['symbol'] == 'BTCUSDT'
            assert config['data_fetcher']['interval'] == '1h'
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_file_not_found(self):
        """Test de archivo no encontrado"""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_file.yaml')
    
    def test_load_config_invalid_yaml(self):
        """Test de YAML inválido"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content:')
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_symbol_valid(self):
        """Test de validación de símbolos válidos"""
        valid_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
        
        for symbol in valid_symbols:
            assert validate_symbol(symbol) == True
    
    def test_validate_symbol_invalid(self):
        """Test de validación de símbolos inválidos"""
        invalid_symbols = ['btcusdt', 'BTC-USDT', 'BTC', '', 123, None]
        
        for symbol in invalid_symbols:
            assert validate_symbol(symbol) == False
    
    def test_validate_interval_valid(self):
        """Test de validación de intervalos válidos"""
        valid_intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        
        for interval in valid_intervals:
            assert validate_interval(interval) == True
    
    def test_validate_interval_invalid(self):
        """Test de validación de intervalos inválidos"""
        invalid_intervals = ['1min', '1hr', '1day', '2m', '10m', '', None, 123]
        
        for interval in invalid_intervals:
            assert validate_interval(interval) == False
