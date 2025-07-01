from pathlib import Path
import yaml

def load_config() -> dict:
    # __file__ = .../test/test_config.py
    project_root = Path(__file__).parent.parent
    config_path  = project_root / "config" / "config.yaml"
    print("Leyendo configuración de:", config_path.resolve())

    if not config_path.is_file():
        raise FileNotFoundError(f"No hay config.yaml aquí: {config_path}")
    # fuerza UTF-8 y evita UnicodeDecodeError
    content = config_path.read_text(encoding="utf-8")
    return yaml.safe_load(content)

if __name__ == "__main__":
    cfg = load_config()
    # Ejemplo de acceso a valores
    print("\n" + "="*50)
    print("✅ Configuración cargada exitosamente!")
    print(f"Símbolo: {cfg['data_fetcher']['symbol']}")
    print(f"Timeframe: {cfg['data_fetcher']['timeframe']}")
    print(f"Estrategia: {cfg['strategy']['name']}")
    print("Indicadores:")
    for indicator in cfg['feature_engine']['indicators']:
        print(f" - {indicator['name']} (params: {indicator['params']})")
    print("="*50 + "\n")