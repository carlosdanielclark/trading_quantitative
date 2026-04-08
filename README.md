# Quantitative Trading Strategy for BTC/USDT

This project implements, evaluates, and documents a quantitative trading strategy applied to the BTC/USDT pair using historical data from Bitunix. It integrates technical indicators and machine learning models (Random Forest) to generate trading signals. The pipeline includes data fetching, feature engineering, model training, backtesting, and performance analysis, providing a complete framework for algorithmic trading research.

## Design, implementation and evaluation of a quantitative trading strategy on BTC/USDT using historical data from Bitunix, technical indicators and decision models

## Environment setup

**1. Install Python**

    If you don't have Python installed:

    Download the latest version of Python 3.11+ from python.org

    Run the installer:
    ✅ Check "Add Python to PATH"
    ✅ "Customize installation" > "Install for all users"
    ✅ Select a short path (e.g., C:\Python311)
    
    > ⚠️ This project uses Python 3.13.5

**2. Create virtual environment**

    mkdir trading_quant_project
    cd trading_quant_project
    python -m venv trading_env

- Activate environment:

    trading_env\Scripts\activate

**3. Install dependencies**

    Run in the activated environment:
    pip install --upgrade pip

    # Main libraries
    pip install pandas numpy scikit-learn matplotlib seaborn requests pyarrow

    # Trading/ML
    pip install ta yfinance lightgbm 

    # Backtesting and visualisation
    pip install backtesting pyfolio quantstats

    # Jupyter and environment management
    pip install jupyter ipykernel

    # Configuration and logs
    pip install python-dotenv configparser

    # Install custom kernel for Jupyter
    python -m ipykernel install --user --name=trading_env

    Verification:
    cmd
    > pip list 
    # Should show the installed libraries

**4. Directory structure**

    ```text
    trading_quant_project/
    │
    ├── config/                # Configuration files
    │   └── config.yaml        # Configurable parameters (dates, intervals, etc.)
    │
    ├── data/                  # Historical data and backtesting results
    │   ├── raw/               # Raw downloaded data
    │   └── processed/         # Processed data ready for analysis
    │
    ├── notebooks/             # Jupyter notebooks for analysis and prototyping
    │   └── exploratory.ipynb
    │
    ├── src/                   # Source code modules
    │   ├── __init__.py
    │   ├── data_fetcher.py    # Data extraction
    │   ├── feature_engine.py  # Indicator calculation and features
    │   ├── strategy.py        # Strategy implementation
    │   ├── backtest.py        # Backtesting
    │   └── utils.py           # Helper functions
    │
    ├── test/                  # Unit tests
    │   ├── __init__.py        # Enables pytest discovery
    │   ├── test_backtest.py
    │   ├── test_config.py
    │   ├── test_feature_engine.py
    │   ├── test_data_fetcher.py  
    │   └── test_strategy.py
    │
    ├── pytest.ini             # pytest configuration
    ├── requirements.txt       # Project dependencies
    ├── README.md              # This instruction file
    └── run_pipeline.py        # Pipeline execution script
    ```

**5. Run the project**

    > python run_pipeline.py

**6. Run tests**

    > pytest
