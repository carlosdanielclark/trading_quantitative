# Diseño, implementación y evaluación de una estrategia de trading cuantitativo sobre BTC/USDT usando datos históricos de Bitunix, indicadores técnicos y modelos de decisión

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

    ```text
    trading_quant_project/
    │
    ├── config/                # Archivos de configuración
    │   └── config.yaml        # Parámetros configurables (fechas, intervalos, etc.)
    │
    ├── data/                  # Datos históricos y resultados de backtesting
    │   ├── raw/               # Datos crudos descargados
    │   └── processed/         # Datos procesados listos para análisis
    │
    ├── notebooks/             # Notebooks Jupyter para análisis y prototipado
    │   └── exploratory.ipynb
    │
    ├── src/                   # Código fuente de los módulos
    │   ├── __init__.py
    │   ├── data_fetcher.py    # Extracción de datos
    │   ├── feature_engine.py  # Cálculo de indicadores y features
    │   ├── strategy.py        # Implementación de la estrategia
    │   ├── backtest.py        # Backtesting
    │   └── utils.py           # Funciones auxiliares
    │
    ├── test/                  # Tests unitarios
    │   ├── __init__.py        # Habilita discovery de pytest
    │   ├── test_config.py
    │   └── test_data_fetcher.py
    │
    ├── pytest.ini             # Configuración de pytest
    ├── requirements.txt       # Dependencias del proyecto
    ├── README.md              # Este archivo de instrucciones
    └── run_pipeline.py        # Script de ejecución del pipeline
    ```
**5. Ejecutar proyecto**
> python run_pipeline
