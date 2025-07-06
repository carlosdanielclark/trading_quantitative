# run_pipeline
import logging
from pathlib import Path
from datetime import datetime

from src.data_fetcher import DataFetcher
from src.feature_engine import FeatureEngine
from src.utils import load_config

# Crear directorio de logs si no existe
Path("logs").mkdir(exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Pipeline")

def main():
    """
    Ejecuta el pipeline de trading (solo extracción de datos en esta etapa).
    """
    logger.info("Iniciando pipeline de trading")
    config = load_config()
    data_config = config['data_fetcher']
    feature_config = config['feature_engine']   # <--- ¡Aquí defines feature_config!


    # PASO 1: Extracción de datos
    logger.info("PASO 1: Extracción de datos")
    fetcher = DataFetcher(data_config)
    df = fetcher.fetch_ohlcv(
        start_date=data_config['start_date'],
        end_date=data_config['end_date']
    )

    if df is None or df.empty:
        logger.error("No se pudieron obtener datos. Abortando pipeline.")
        return

    logger.info(f"Datos descargados exitosamente. Registros: {len(df)}")
    
    # PASO 2: Generación de features
    logger.info("PASO2: Generación de features (feature_engine)")

    # Instanciar el motor de features con los datos descargados
    feature_engine = FeatureEngine(df, cache_enabled=data_config['storage'].get('cache_enabled', True))

    # Añadir todos los indicadores definidos en el config
    indicators = feature_config.get('indicators', [])
    feature_engine.add_multiple_indicators(indicators)

    # Exportar features a la carpeta de processed
    processed_path = data_config['storage']['processed_path']
    symbol = data_config['symbol']
    interval = data_config['interval']
    start = data_config['start_date']
    end = data_config['end_date']
    file_format = data_config['storage'].get('file_format', 'parquet')
    features_filename = f"{processed_path}/features_{symbol}_{interval}_{start}_{end}.{file_format}"

    feature_engine.export_features(path=features_filename, format=file_format)

    logger.info(f"PASO2: Features generados y exportados correctamente a {features_filename}")
    logger.info("Pipeline finalizado.")
    

if __name__ == "__main__":
    main()