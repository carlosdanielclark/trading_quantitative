# run_pipeline
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from src.data_fetcher import DataFetcher
from src.feature_engine import FeatureEngine
from src.backtest import BacktestEngine
from src.utils import load_config

# --- NUEVO: Importa el módulo de estrategia ---
from src.strategy import main as run_strategy

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


    # --- PASO 1: Extracción de datos ---
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
    
    # --- PASO 2: Cálculo de features ---
    logger.info("PASO2: Cálculo de features (feature_engine)")

    # Instanciar el motor de features con los datos descargados
    feature_engine = FeatureEngine(df, cache_enabled=data_config['storage'].get('cache_enabled', True))

    # Añadir todos los indicadores definidos en el config
    indicators = feature_config.get('indicators', [])
    feature_engine.add_multiple_indicators(indicators)
    
    # --- Agrega la columna 'close' al DataFrame de features ---
    feature_engine.features['close'] = df['close']

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

     # --- PASO 3: Estrategia Random Forest ---
    logger.info("PASO3: Ejecución de la estrategia Random Forest (strategy.py)")
    try:
        run_strategy()  # Usa la configuración global del proyecto
        logger.info("Estrategia ejecutada y señales generadas correctamente.")
    except Exception as e:
        logger.error(f"Error ejecutando la estrategia: {e}")

    # --- PASO 4: Ejecutar backtest ---
    logger.info("PASO4: Ejecución de backtest con módulo BacktestEngine")
    try:
        # Cargar configuración completa
        config = load_config()

        # Cargar features y señales desde paths configurados
        features_path = config['randomforest']['features_path']
        signals_path = config['randomforest']['export_signals_path']

        features = pd.read_parquet(features_path)
        signals = pd.read_parquet(signals_path)['signal']

        # Alinear índices
        features = features.loc[signals.index]

        # Instanciar motor de backtesting con configuración
        engine = BacktestEngine(config)

        # Ejecutar backtest con tamaño de posición 10%
        results = engine.run_backtest(features, signals, position_size=0.1)

        # Obtener métricas
        metrics = engine.get_performance_metrics(results)
        logger.info("Métricas de rendimiento del backtest:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")

        # Visualización 1: Curva de equity con drawdowns
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.equity_curve.index, y=results.equity_curve,
                                 mode='lines', name='Equity Curve'))
        fig.add_trace(go.Scatter(x=results.drawdowns.index,
                                 y=results.drawdowns * results.equity_curve,
                                 fill='tozeroy', mode='none', name='Drawdown',
                                 fillcolor='rgba(255,0,0,0.3)'))
        fig.update_layout(title='Curva de Equity con Drawdowns',
                          xaxis_title='Fecha', yaxis_title='Capital')
        fig.show()

        # Visualización 2: Distribución de retornos por trade
        trade_returns = [t.return_pct for t in results.trades if not t.is_open]
        fig2 = px.histogram(trade_returns, nbins=50,
                            title='Distribución de Retornos por Trade',
                            labels={'value': 'Retorno (%)'})
        fig2.update_layout(bargap=0.1)
        fig2.show()

        # Visualización 3: Heatmap de métricas por parámetros (simulado)
        if config['randomforest'].get('grid_search', False):
            import itertools
            n_estimators = config['randomforest']['param_grid'].get('n_estimators', [100])
            max_depth = config['randomforest']['param_grid'].get('max_depth', [5])
            data_heatmap = []
            for n, d in itertools.product(n_estimators, max_depth):
                metric_val = 0.5 + 0.4 * np.random.rand()  # Simulación métrica
                data_heatmap.append({'n_estimators': n, 'max_depth': d if d is not None else 'None', 'metric': metric_val})
            df_heatmap = pd.DataFrame(data_heatmap)
            heatmap_data = df_heatmap.pivot(index='max_depth', columns='n_estimators', values='metric')
            plt.figure(figsize=(8,6))
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis')
            plt.title('Heatmap de Métricas por Parámetros')
            plt.ylabel('max_depth')
            plt.xlabel('n_estimators')
            plt.show()

        # 9. Guardar reporte PDF con métricas
        pdf_filename = "backtest_report.pdf"
        engine.generate_pdf_report(results, metrics, filename=pdf_filename)
        print(f"Reporte PDF generado: {pdf_filename}")

        logger.info("Backtest y visualizaciones completados correctamente.")

    except Exception as e:
        logger.error(f"Error ejecutando el backtest: {e}")

    logger.info("Pipeline finalizado.")
    
if __name__ == "__main__":
    main()