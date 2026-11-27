"""Configuração do Vestígio, lida de variáveis de ambiente e do arquivo .env."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Modelo padrão: Smaug, classificador visual publicado pela Abacus.AI (Apache-2.0).
DEFAULT_MODEL_ID = "abacusai/Smaug-AI-Image-Detector"
MAX_MODELS = 3

# Nomes de formato como o Pillow os reporta (o MPO é um JPEG com várias imagens, comum em câmeras).
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
ALLOWED_MIME_TYPES = ("image/jpeg", "image/png", "image/webp")
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VALID_DEVICES = ("auto", "cpu", "cuda")

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _read_number(name: str, default, cast, minimum):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError:
        raise ValueError(f"{name} deve ser um número (valor atual: {raw!r}).") from None
    if value < minimum:
        raise ValueError(f"{name} deve ser pelo menos {minimum} (valor atual: {value}).")
    return value


@dataclass(frozen=True)
class Config:
    model_ids: tuple[str, ...] = (DEFAULT_MODEL_ID,)
    device: str = "auto"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 40_000_000
    min_side: int = 64
    host: str = "127.0.0.1"
    port: int = 5000

    @classmethod
    def from_env(cls) -> "Config":
        # Variáveis já definidas no sistema têm prioridade sobre o conteúdo do .env.
        load_dotenv(BASE_DIR / ".env")

        raw_models = os.getenv("MODEL_IDS", DEFAULT_MODEL_ID)
        model_ids = tuple(dict.fromkeys(m.strip() for m in raw_models.split(",") if m.strip()))
        if not model_ids:
            model_ids = (DEFAULT_MODEL_ID,)
        if len(model_ids) > MAX_MODELS:
            raise ValueError(f"MODEL_IDS aceita no máximo {MAX_MODELS} modelos.")
        for model_id in model_ids:
            if not _MODEL_ID_PATTERN.match(model_id) and not Path(model_id).is_dir():
                raise ValueError(
                    f"MODEL_IDS contém um valor inválido: {model_id!r}. "
                    "Use o formato dono/nome do Hugging Face ou o caminho de uma pasta local."
                )

        device = os.getenv("DETECTOR_DEVICE", "auto").strip().lower()
        if device not in VALID_DEVICES:
            raise ValueError(f"DETECTOR_DEVICE deve ser um destes valores: {', '.join(VALID_DEVICES)}.")

        max_mb = _read_number("MAX_UPLOAD_MB", 10.0, float, 0.1)
        max_pixels = _read_number("MAX_IMAGE_PIXELS", 40_000_000, int, 1_000)
        port = _read_number("APP_PORT", 5000, int, 1)
        if port > 65535:
            raise ValueError("APP_PORT deve estar entre 1 e 65535.")

        return cls(
            model_ids=model_ids,
            device=device,
            max_upload_bytes=int(max_mb * 1024 * 1024),
            max_pixels=max_pixels,
            host=os.getenv("APP_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=port,
        )
