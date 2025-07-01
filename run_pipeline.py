import logging
from pathlib import Path
from datetime import datetime

from src.data_fetcher import DataFetcher
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

    # Extracción de datos
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
    logger.info("Pipeline finalizado (extracción de datos).")

if __name__ == "__main__":
    main()