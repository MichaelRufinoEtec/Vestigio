"""Validação e decodificação segura de imagens enviadas. Tudo acontece em memória."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ALLOWED_FORMATS

_FORMAT_ALIASES = {"MPO": "JPEG"}
_FRIENDLY_FORMATS = "JPEG, PNG ou WebP"


class ImageRejected(Exception):
    """Arquivo recusado, com código estável, mensagem para o usuário e status HTTP."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image  # sempre RGB
    format: str
    width: int
    height: int


def decode_upload(data: bytes, *, max_pixels: int, min_side: int) -> DecodedImage:
    """Confere se `data` é mesmo uma imagem suportada e a converte para RGB.

    Não confia em extensão nem em Content-Type: quem decide é o conteúdo do arquivo.
    """
    if not data:
        raise ImageRejected("EMPTY_FILE", "O arquivo enviado está vazio.", 400)

    try:
        probe = Image.open(BytesIO(data))
        detected = _FORMAT_ALIASES.get(probe.format or "", probe.format)
        width, height = probe.size
    except UnidentifiedImageError:
        raise ImageRejected(
            "UNSUPPORTED_TYPE",
            f"O arquivo não é uma imagem reconhecida. Use {_FRIENDLY_FORMATS}.",
            415,
        ) from None
    except Image.DecompressionBombError:
        raise ImageRejected(
            "IMAGE_DIMENSIONS_TOO_LARGE", "As dimensões da imagem são grandes demais para análise.", 422
        ) from None
    except (OSError, ValueError, SyntaxError):
        raise ImageRejected("INVALID_IMAGE", "A imagem está corrompida e não pôde ser lida.", 422) from None

    if detected not in ALLOWED_FORMATS:
        shown = (detected or "desconhecido").upper()
        raise ImageRejected(
            "UNSUPPORTED_TYPE", f"O formato {shown} não é suportado. Use {_FRIENDLY_FORMATS}.", 415
        )
    if width * height > max_pixels:
        raise ImageRejected(
            "IMAGE_DIMENSIONS_TOO_LARGE",
            f"A imagem tem {width} × {height} pixels, acima do limite de {max_pixels / 1_000_000:.0f} megapixels.",
            422,
        )
    if min(width, height) < min_side:
        raise ImageRejected(
            "IMAGE_TOO_SMALL",
            f"A imagem é pequena demais ({width} × {height}). O lado menor precisa ter ao menos {min_side} pixels.",
            422,
        )

    try:
        img = Image.open(BytesIO(data))
        img.load()  # decodifica de fato; arquivos truncados falham aqui
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError):
        raise ImageRejected(
            "INVALID_IMAGE", "A imagem está corrompida ou incompleta e não pôde ser lida.", 422
        ) from None

    try:
        img = ImageOps.exif_transpose(img)  # respeita a orientação gravada pela câmera
    except Exception:  # noqa: BLE001 - EXIF malformado não deve impedir a análise
        pass

    try:
        rgb = _to_rgb(img)
    except (OSError, ValueError):
        raise ImageRejected("INVALID_IMAGE", "Não foi possível converter a imagem para análise.", 422) from None

    return DecodedImage(image=rgb, format=detected, width=rgb.width, height=rgb.height)


def _to_rgb(img: Image.Image) -> Image.Image:
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")
