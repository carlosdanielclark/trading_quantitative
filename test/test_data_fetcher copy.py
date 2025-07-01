# trading_quant_project/test/test_data_fetcher.py

# import sys

#from pathlib import Path
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from src.utils import load_config

# Agregar src al path
# sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_data_fetcher():
    """Prueba de integración para el módulo DataFetcher"""
    config = load_config()['data_fetcher']
    
    # Usar un rango de fechas más pequeño para pruebas
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    fetcher = DataFetcher({
        **config,
        'start_date': start_date,
        'end_date': end_date
    })
    
    # Ejecutar la descarga
    df = fetcher.fetch_ohlcv(start_date, end_date)
    
    # Verificaciones básicas
    assert not df.empty, "El DataFrame está vacío"
    assert len(df) > 100, "No hay suficientes registros"
    assert 'close' in df.columns, "Falta columna 'close'"
    
    print("\n" + "="*50)
    print("✅ Prueba exitosa! Datos descargados:")
    print(f"Registros: {len(df)}")
    print(f"Primera fecha: {df.index[0]}")
    print(f"Última fecha: {df.index[-1]}")
    print(f"Columnas: {list(df.columns)}")
    print("="*50 + "\n")
    
    return df

if __name__ == "__main__":
    test_data_fetcher()