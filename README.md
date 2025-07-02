# Diseño, implementación y evaluación de una estrategia de trading cuantitativo 
# sobre BTC/USDT usando datos históricos de Bitunix, indicadores técnicos y modelos de decisión

## Configuración del entorno de trabajo

**1. Instalar Python**
    Si no tienes Python instalado:

    Descarga la última versión de Python 3.11+ desde   python.org

    Ejecuta el instalador:
    ✅ Marcar "Add Python to PATH"
    ✅ Opción "Customize installation" > "Install for all users"
    ✅ Seleccionar ruta corta (ej: C:\Python311)
    
    > ⚠️ En este proyecto se usa Python 3.13.5 

**2. Crear entorno virtual**
    mkdir trading_quant_project
    cd trading_quant_project
    python -m venv trading_env

    Activar entorno:
    cmd: 
    > trading_env\Scripts\activate

**3. Instalar dependencias**
    Ejecuta en el entorno activado:
    pip install --upgrade pip

    # Librerías principales
    pip install pandas numpy scikit-learn matplotlib   seaborn requests pyarrow

    # Trading/ML
    pip install ta yfinance lightgbm 

    # Backtesting y visualización
    pip install backtesting pyfolio quantstats

    # Jupyter y gestión de entornos
    pip install jupyter ipykernel

    # Configuración y logs
    pip install python-dotenv configparser

    # Instalar kernel personalizado para Jupyter
    python -m ipykernel install --user --name=trading_env

    Verificación:
    cmd
    > pip list 
    # Debe mostrar las librerías instaladas

**4. Estructura de directorios**
trading_quant_project/
│
├── config/                # Archivos de configuración
│   └── config.yaml       # Parámetros configurables como fechas, intervalos, etc.
│
├── data/                  # Carpeta para almacenar datos históricos y resultados de backtesting
│   ├── raw/               # Datos crudos descargados
│   └── processed/         # Datos procesados y listos para análisis
│
├── notebooks/             # Notebooks Jupyter para análisis y prueba de estrategias
│
├── src/                   # Código fuente (módulos)
│   ├── __init__.py
│   ├── data_fetcher.py    # Módulo para extracción de datos
│   ├── feature_engine.py  # Módulo para cálculo de indicadores y features
│   ├── strategy.py       # Módulo para la implementación de la estrategia
│   ├── backtest.py       # Módulo para el backtesting
│   └── utils.py          # Funciones auxiliares
│
├── test/                   # Código fuente (módulos)
│   ├── __init__.py
│   ├── test_config.py
│   └──  test_data_fetcher.py  
│
├── pytest.ini
├── requirements.txt      # Dependencias del proyecto
├── README.md             # Documento de instrucciones y descripción del proyecto
└── run_pipeline.py       # Script principal para ejecutar el pipeline de trading

**5. Ejecutar proyecto**
> python run_pipeline