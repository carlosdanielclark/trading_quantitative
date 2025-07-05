# src/feature_engine.py
"""
Motor de ingeniería de características para trading algorítmico
"""

import pandas as pd
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging

from src.indicators import IndicatorRegistry

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Motor de cálculo de features técnicos para trading algorítmico
    Permite añadir indicadores de forma modular y escalable
    """
    
    def __init__(self, df: pd.DataFrame, cache_enabled: bool = True):
        """
        Inicializa el motor de features
        
        Args:
            df: DataFrame con datos OHLCV
            cache_enabled: Si activar sistema de caché
        """
        # Validar DataFrame de entrada
        if df.empty:
            raise ValueError("DataFrame no puede estar vacío")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Columnas faltantes en DataFrame: {missing_columns}")
        
        # CORRECCIÓN: No redondear los datos originales
        self.df = df.copy()  # Mantener precisión original
        
        # Sistema de caché thread-safe
        self._cache_enabled = cache_enabled
        self._cache: Dict[tuple, pd.Series] = {}
        self._lock = threading.RLock()
        
        # Almacenar escaladores para posible uso posterior
        self.scalers: Dict[str, Any] = {}
        
        # DataFrame para almacenar features procesados
        self.features = pd.DataFrame(index=self.df.index)
        
        # Configuración para imputación de valores faltantes
        self.imputer = SimpleImputer(strategy='forward_fill')
        
        logger.info(f"FeatureEngine inicializado con {len(self.df)} registros")

    def add_indicator(self, name: str, params: Dict[str, Any], 
                    normalize: bool = True, scaler: str = "standard") -> None:
        """
        Añade un indicador técnico como feature

        Args:
            name: Nombre del indicador registrado
            params: Parámetros del indicador
            normalize: Si normalizar el indicador (pasa a la función del indicador)
            scaler: Tipo de escalador ('standard', 'minmax', 'robust')

        Raises:
            ValueError: Si el indicador no está registrado o hay datos insuficientes
        """
        fn = IndicatorRegistry.get(name)
        if not fn:
            raise ValueError(f"Indicador '{name}' no registrado. "
                            f"Disponibles: {IndicatorRegistry.list_indicators()}")

        # Crear clave única para el cache
        cache_key = (name, tuple(sorted(params.items())), normalize, scaler)

        with self._lock:
            # Verificar cache primero
            if self._cache_enabled and cache_key in self._cache:
                logger.info(f"Usando cache para {name}{params}")
                result = self._cache[cache_key]
            else:
                # Validar datos suficientes
                self._validate_data_sufficiency(name, params)

                # Calcular indicador
                try:
                    # PASA normalize como argumento al indicador
                    if 'normalize' in fn.__code__.co_varnames:
                        result = fn(self.df, **params, normalize=normalize)
                    else:
                        result = fn(self.df, **params)

                    # Manejar diferentes tipos de retorno
                    if isinstance(result, pd.DataFrame):
                        # Para indicadores que retornan múltiples series
                        for col in result.columns:
                            feature_name = f"{name}_{col}_{self._params_to_string(params)}"
                            series = result[col]
                            # Solo normalizar si el indicador no lo hizo
                            if normalize and not (name == "rsi" and 'normalize' in fn.__code__.co_varnames):
                                series = self._normalize_series(series, scaler, feature_name)
                            self.features[feature_name] = series

                        logger.info(f"Añadido indicador múltiple {name} con {len(result.columns)} features")
                        return

                    elif isinstance(result, pd.Series):
                        # Para indicadores que retornan una sola serie
                        # Solo normalizar si el indicador no lo hizo
                        if normalize and not (name == "rsi" and 'normalize' in fn.__code__.co_varnames):
                            result = self._normalize_series(result, scaler, name)

                        # Guardar en cache
                        if self._cache_enabled:
                            self._cache[cache_key] = result

                    else:
                        raise ValueError(f"Tipo de retorno no soportado: {type(result)}")

                except Exception as e:
                    logger.error(f"Error calculando indicador {name}: {e}")
                    raise

            # Añadir al DataFrame de features
            feature_name = f"{name}_{self._params_to_string(params)}"
            self.features[feature_name] = result

            logger.info(f"Feature '{feature_name}' añadido exitosamente")


    def add_multiple_indicators(self, indicators_config: List[Dict[str, Any]]) -> None:
        """
        Añade múltiples indicadores desde configuración
        
        Args:
            indicators_config: Lista de configuraciones de indicadores
        """
        for config in indicators_config:
            try:
                name = config['name']
                params = config.get('params', {})
                normalize = config.get('normalize', True)
                scaler = config.get('scaler', 'standard')
                
                self.add_indicator(name, params, normalize, scaler)
                
            except Exception as e:
                logger.error(f"Error añadiendo indicador {config}: {e}")
                continue

    def _validate_data_sufficiency(self, name: str, params: Dict[str, Any]) -> None:
        """
        Valida que hay suficientes datos para el indicador
        
        Args:
            name: Nombre del indicador
            params: Parámetros del indicador
            
        Raises:
            ValueError: Si no hay suficientes datos
        """
        # Buscar parámetros que indiquen ventana temporal
        window_params = ['period', 'window', 'n', 'fast', 'slow', 'k_period', 'd_period']
        max_window = 0
        
        for param in window_params:
            if param in params:
                max_window = max(max_window, params[param])
        
        if max_window > 0 and len(self.df) < max_window:
            raise ValueError(f"Datos insuficientes para {name}. "
                           f"Requeridos: {max_window}, Disponibles: {len(self.df)}")

    def _normalize_series(self, series: pd.Series, scaler: str, feature_name: str) -> pd.Series:
        """
        Normaliza una serie usando el escalador especificado
        
        Args:
            series: Serie a normalizar
            scaler: Tipo de escalador
            feature_name: Nombre del feature para logging
            
        Returns:
            Serie normalizada
        """
        # Manejar valores faltantes
        if series.isna().any():
            logger.warning(f"Valores faltantes detectados en {feature_name}, aplicando imputación")
            series = series.ffill().bfill()
        
        # Validar que quedan datos para normalizar
        if series.dropna().empty:
            logger.warning(f"No hay datos válidos para normalizar {feature_name}")
            return series
        
        # Seleccionar escalador
        if scaler == "minmax":
            scaler_obj = MinMaxScaler()
        elif scaler == "robust":
            scaler_obj = RobustScaler()
        else:  # standard por defecto
            scaler_obj = StandardScaler()
        
        try:
            # Aplicar normalización
            values_reshaped = series.values.reshape(-1, 1)
            normalized_values = scaler_obj.fit_transform(values_reshaped).flatten()
            
            # Guardar escalador para uso posterior
            self.scalers[feature_name] = scaler_obj
            
            normalized_series = pd.Series(normalized_values, index=series.index)
            logger.info(f"Serie {feature_name} normalizada con {scaler}")
            
            return normalized_series
            
        except Exception as e:
            logger.error(f"Error normalizando {feature_name}: {e}")
            return series

    def _params_to_string(self, params: Dict[str, Any]) -> str:
        """
        Convierte parámetros a string para nombres de columnas
        
        Args:
            params: Diccionario de parámetros
            
        Returns:
            String representando los parámetros
        """
        if not params:
            return ""
        
        param_strs = [f"{k}{v}" for k, v in sorted(params.items())]
        return "_".join(param_strs)

    def compute_all(self) -> pd.DataFrame:
        """
        Retorna DataFrame con todas las features añadidas
        
        Returns:
            DataFrame con todas las features
        """
        if self.features.empty:
            logger.warning("No se han añadido features")
            return pd.DataFrame()
        
        # Crear copia para evitar modificaciones accidentales
        result = self.features.copy()
        
        # Estadísticas básicas
        logger.info(f"Features computados: {len(result.columns)}")
        logger.info(f"Registros: {len(result)}")
        
        # Verificar valores faltantes
        na_counts = result.isna().sum()
        if na_counts.sum() > 0:
            logger.warning(f"Valores faltantes por columna:\n{na_counts[na_counts > 0]}")
        
        return result

    def get_feature_statistics(self) -> pd.DataFrame:
        """
        Obtiene estadísticas descriptivas de las features
        
        Returns:
            DataFrame con estadísticas
        """
        if self.features.empty:
            return pd.DataFrame()
        
        stats = self.features.describe()
        
        # Añadir estadísticas adicionales
        stats.loc['missing_count'] = self.features.isna().sum()
        stats.loc['missing_percent'] = (self.features.isna().sum() / len(self.features)) * 100
        
        return stats

    def export_features(self, path: Optional[str] = None, 
                       format: str = "parquet") -> str:
        """
        Exporta features a archivo
        
        Args:
            path: Ruta del archivo (opcional)
            format: Formato de archivo ('parquet', 'csv')
            
        Returns:
            Ruta del archivo guardado
        """
        if self.features.empty:
            raise ValueError("No hay features para exportar")
        
        if path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            path = f"features_{timestamp}.{format}"
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "parquet":
                self.features.to_parquet(path_obj)
            elif format.lower() == "csv":
                self.features.to_csv(path_obj)
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            logger.info(f"Features exportados a: {path_obj}")
            return str(path_obj)
            
        except Exception as e:
            logger.error(f"Error exportando features: {e}")
            raise

    def clear_cache(self) -> None:
        """
        Limpia el cache de indicadores
        """
        with self._lock:
            self._cache.clear()
            logger.info("Cache de indicadores limpiado")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtiene información del cache
        
        Returns:
            Diccionario con información del cache
        """
        with self._lock:
            return {
                'enabled': self._cache_enabled,
                'size': len(self._cache),
                'keys': list(self._cache.keys())
            }

    def __repr__(self) -> str:
        return (f"FeatureEngine(data_shape={self.df.shape}, "
                f"features={len(self.features.columns)}, "
                f"cache_size={len(self._cache)})")
