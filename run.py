"""Ponto de entrada: python run.py"""
import logging
import os
import sys

# Evita um aviso inofensivo do Hugging Face no Windows (cache sem links simbólicos).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from backend.app import create_app  # noqa: E402
from backend.config import Config  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = Config.from_env()
    except ValueError as exc:
        print(f"\nConfiguração inválida: {exc}\nCorrija o arquivo .env e rode de novo.\n")
        sys.exit(1)

    app = create_app(config)
    print(
        f"\n  Vestígio em http://{config.host}:{config.port}\n"
        "  O modelo carrega em segundo plano; a página mostra quando o motor está pronto.\n"
        "  Para encerrar, pressione Ctrl+C.\n"
    )
    # debug=False e sem reloader: o reloader carregaria o modelo duas vezes.
    app.run(host=config.host, port=config.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
