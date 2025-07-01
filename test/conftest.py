# test/conftest.py
import requests

class DummyResponse:
    def __init__(self, status_code=200, data=None, url="mock://"):
        self.status_code = status_code
        self._data = data or []
        self.url = url

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"{self.status_code} Error for url: {self.url}")

    def json(self):
        # <— Aquí devolvemos la lista directamente
        return self._data
    
#    def json(self):
#        return {"data": self._data}