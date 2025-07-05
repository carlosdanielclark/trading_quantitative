# test/test_feature_engine.py
"""
Tests para el módulo feature_engine
"""
import sys
import os

# Añade la raíz del proyecto al sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
    
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.feature_engine import FeatureEngine
from src.indicators import IndicatorRegistry

class TestFeatureEngine:
    
    @pytest.fixture
    def sample_data(self):
        """Datos de prueba OHLCV"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)
        
        # Generar datos sintéticos realistas
        base_price = 60000
        returns = np.random.normal(0, 0.001, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0002, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0002, 100))),
            'close': prices,
            'volume': np.random.uniform(10, 100, 100)
        }, index=dates)
    
    def test_feature_engine_initialization(self, sample_data):
        """Test de inicialización del FeatureEngine"""
        fe = FeatureEngine(sample_data)
        
        assert len(fe.df) == 100
        assert not fe.features.empty or len(fe.features.columns) == 0
        assert fe._cache_enabled == True
    
    def test_feature_engine_empty_data(self):
        """Test con DataFrame vacío"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            FeatureEngine(empty_df)
    
    def test_feature_engine_missing_columns(self):
        """Test con columnas faltantes"""
        incomplete_df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1, 2, 3]
            # Faltan low, close, volume
        })
        
        with pytest.raises(ValueError):
            FeatureEngine(incomplete_df)
    
    def test_add_single_indicator(self, sample_data):
        """Test de añadir un indicador simple"""
        fe = FeatureEngine(sample_data)
        
        # Añadir RSI
        fe.add_indicator('rsi', {'period': 14})
        
        # Verificar que se añadió
        assert len(fe.features.columns) == 1
        assert 'rsi_period14' in fe.features.columns
        
        # Verificar valores válidos
        rsi_values = fe.features['rsi_period14'].dropna()
        assert len(rsi_values) > 0
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 1
    
    def test_add_multiple_indicators(self, sample_data):
        """Test de añadir múltiples indicadores"""
        fe = FeatureEngine(sample_data)
        
        # Añadir varios indicadores
        fe.add_indicator('sma', {'period': 20})
        fe.add_indicator('ema', {'period': 12})
        fe.add_indicator('rsi', {'period': 14})
        
        # Verificar que se añadieron
        assert len(fe.features.columns) == 3
        assert 'sma_period20' in fe.features.columns
        assert 'ema_period12' in fe.features.columns
        assert 'rsi_period14' in fe.features.columns
    
    def test_normalization_options(self, sample_data):
        """Test de opciones de normalización"""
        fe = FeatureEngine(sample_data)
        
        # Test sin normalización
        fe.add_indicator('sma', {'period': 20}, normalize=False)
        
        # Test con normalización estándar
        fe.add_indicator('ema', {'period': 12}, normalize=True, scaler='standard')
        
        # Test con normalización minmax
        fe.add_indicator('rsi', {'period': 14}, normalize=True, scaler='minmax')
        
        # Verificar que se añadieron
        assert len(fe.features.columns) == 3
        
        # Verificar rangos de normalización
        ema_values = fe.features['ema_period12'].dropna()
        rsi_values = fe.features['rsi_period14'].dropna()
        
        # EMA normalizada debería tener media ~0 y std ~1
        assert abs(ema_values.mean()) < 0.1
        assert abs(ema_values.std() - 1) < 0.1
        
        # RSI normalizada con minmax debería estar entre 0 y 1
        assert rsi_values.min() >= -0.1  # Permitir pequeña tolerancia
        assert rsi_values.max() <= 1.1
    
    def test_cache_functionality(self, sample_data):
        """Test de funcionalidad de cache"""
        fe = FeatureEngine(sample_data, cache_enabled=True)
        
        # Añadir indicador por primera vez
        fe.add_indicator('sma', {'period': 20})
        
        # Verificar que hay algo en cache
        cache_info = fe.get_cache_info()
        assert cache_info['enabled'] == True
        assert cache_info['size'] > 0
        
        # Limpiar cache
        fe.clear_cache()
        cache_info = fe.get_cache_info()
        assert cache_info['size'] == 0
    
    def test_insufficient_data_handling(self, sample_data):
        """Test de manejo de datos insuficientes"""
        # Crear DataFrame con pocos datos
        small_data = sample_data.head(5)
        fe = FeatureEngine(small_data)
        
        # Intentar añadir indicador que requiere más datos
        with pytest.raises(ValueError):
            fe.add_indicator('sma', {'period': 20})
    
    def test_multi_output_indicator(self, sample_data):
        """Test de indicador con múltiples salidas"""
        fe = FeatureEngine(sample_data)
        
        # Añadir Bollinger Bands (retorna DataFrame)
        fe.add_indicator('bollinger_bands', {'period': 20, 'std_dev': 2})
        
        # Verificar que se añadieron múltiples features
        bb_features = [col for col in fe.features.columns if 'bollinger_bands' in col]
        assert len(bb_features) > 1
    
    def test_feature_statistics(self, sample_data):
        """Test de estadísticas de features"""
        fe = FeatureEngine(sample_data)
        
        # Añadir algunos indicadores
        fe.add_indicator('sma', {'period': 20})
        fe.add_indicator('rsi', {'period': 14})
        
        # Obtener estadísticas
        stats = fe.get_feature_statistics()
        
        # Verificar estructura
        assert not stats.empty
        assert 'mean' in stats.index
        assert 'std' in stats.index
        assert 'missing_count' in stats.index
        assert 'missing_percent' in stats.index
    
    def test_export_functionality(self, sample_data, tmp_path):
        """Test de funcionalidad de exportación"""
        fe = FeatureEngine(sample_data)
        
        # Añadir indicadores
        fe.add_indicator('sma', {'period': 20})
        fe.add_indicator('rsi', {'period': 14})
        
        # Test exportar como parquet
        parquet_path = tmp_path / "test_features.parquet"
        result_path = fe.export_features(str(parquet_path), format='parquet')
        assert Path(result_path).exists()
    
    def test_compute_all_functionality(self, sample_data):
        """Test de compute_all()"""
        fe = FeatureEngine(sample_data)
        
        # Añadir indicadores
        fe.add_indicator('sma', {'period': 20})
        fe.add_indicator('rsi', {'period': 14})
        
        # Obtener todas las features
        all_features = fe.compute_all()
        
        # Verificar que es una copia
        assert id(all_features) != id(fe.features)
        
        # Verificar contenido
        assert len(all_features.columns) == 2
        assert not all_features.empty
    
    def test_add_multiple_from_config(self, sample_data):
        """Test de añadir múltiples indicadores desde configuración"""
        fe = FeatureEngine(sample_data)
        
        config = [
            {'name': 'sma', 'params': {'period': 20}},
            {'name': 'rsi', 'params': {'period': 14}, 'normalize': True, 'scaler': 'minmax'},
            {'name': 'ema', 'params': {'period': 12}, 'normalize': False}
        ]
        
        fe.add_multiple_indicators(config)
        
        # Verificar que se añadieron todos
        assert len(fe.features.columns) == 3
