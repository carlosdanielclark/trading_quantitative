# test/test_data_fetcher.py

import requests
from datetime import datetime, timedelta

from src.data_fetcher import DataFetcher
from src.utils        import load_config

class DummyResponse:
    def __init__(self, status_code=200, data=None, url="mock://"):
        self.status_code = status_code
        self._data = data or []
        self.url = url

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"{self.status_code} Error for url: {self.url}")

    def json(self):
        # Devuelve directamente la lista de klines (12 columnas)
        return self._data

def test_data_fetcher(monkeypatch):
    # 1) Configuración y rango de fechas
    cfg   = load_config()['data_fetcher']
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    # 2) Simula 3 filas con 12 valores cada una:
    #    [timestamp, open, high, low, close, volume,
    #     close_time, quote_asset_volume, trades,
    #     taker_buy_base, taker_buy_quote, ignore]
    sample_segment = [
        [1622505600000, 1.0, 1.1, 0.9,  1.05, 100, 0, 0, 0, 0, 0, 0],
        [1622592000000, 1.05,1.2, 1.0,  1.15, 150, 0, 0, 0, 0, 0, 0],
        [1622678400000, 1.15,1.3, 1.1,  1.25, 200, 0, 0, 0, 0, 0, 0],
    ]

    # 3) Monkeypatch de requests.get en src.data_fetcher
    def fake_get(url, params, timeout):
        print(f"\n🎣 fake_get -> url={url}")
        print(f"          params={params}\n")
        return DummyResponse(status_code=200, data=sample_segment, url=url)

    monkeypatch.setattr("src.data_fetcher.requests.get", fake_get)

    # 4) Ejecuta la descarga
    fetcher = DataFetcher({**cfg, 'start_date': start, 'end_date': end})
    df = fetcher.fetch_ohlcv(start, end)

    # 5) Prints de depuración
    print("----- DataFrame HEAD -----")
    print(df.head(), "\n")
    print("----- DataFrame TAIL -----")
    print(df.tail(), "\n")
    print("Columns:", df.columns.tolist())
    print("Index range:", df.index[0], "->", df.index[-1])
    print("Total rows:", len(df))
    print("--------------------------\n")

    # 6) Asserts finales
    assert not df.empty,          "El DataFrame está vacío"
    assert 'close' in df.columns, "Falta columna 'close'"
    assert len(df) == len(sample_segment), "El número de filas no coincide"
