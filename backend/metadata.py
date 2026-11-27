"""Sinais complementares lidos dos metadados do arquivo (EXIF, texto PNG, XMP, C2PA).

São evidências do arquivo, separadas da pontuação do modelo. Metadados podem ser apagados ou
forjados; por isso a ausência deles nunca é tratada como indício de nada.
"""
from __future__ import annotations

import re
from io import BytesIO

from PIL import ExifTags, Image

_EXIF_IFD = 0x8769
_TEXT_LIMIT = 160
_SCAN_LIMIT = 20_000

# Termos padronizados (IPTC "Digital Source Type"), usados também em manifestos C2PA.
_TRAINED = b"trainedAlgorithmicMedia"
_COMPOSITE = b"compositeWithTrainedAlgorithmicMedia"

_GENERATOR = re.compile(
    r"\b(stable[ _-]?diffusion|midjourney|dall[-·_ ]?e|comfyui|novelai|invokeai|fooocus|"
    r"automatic1111|adobe firefly|sdxl)\b",
    re.IGNORECASE,
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_CAPTURE_FIELDS = (
    ("ExposureTime", "tempo de exposição"),
    ("FNumber", "abertura"),
    ("ISOSpeedRatings", "ISO"),
    ("PhotographicSensitivity", "ISO"),
    ("DateTimeOriginal", "data da captura"),
    ("LensModel", "lente"),
    ("FocalLength", "distância focal"),
)

_ORDER = {"ai_declared": 0, "ai_hint": 1, "camera": 2, "info": 3}


def _clean(value, limit: int = _TEXT_LIMIT) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = _CONTROL.sub(" ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _signal(kind: str, title: str, detail: str) -> dict:
    return {"kind": kind, "title": title, "detail": detail}


def extract_signals(data: bytes) -> dict:
    """Devolve {"signals": [...], "declares_ai": bool}. Nunca levanta exceção."""
    try:
        with Image.open(BytesIO(data)) as img:
            info = dict(img.info)
            exif = img.getexif()
            exif_main = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            try:
                exif_ifd = {ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(_EXIF_IFD).items()}
            except Exception:  # noqa: BLE001
                exif_ifd = {}
    except Exception:  # noqa: BLE001
        return {
            "signals": [_signal("info", "Metadados não lidos", "Não foi possível ler os metadados deste arquivo.")],
            "declares_ai": False,
        }

    texts: dict[str, str] = {}
    for key, value in info.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            texts[key] = value
        elif isinstance(value, bytes) and key.lower() in ("xmp", "xml:com.adobe.xmp"):
            texts[key] = value.decode("utf-8", "ignore")

    extra = "\n".join(texts.values()).encode("utf-8", "ignore")
    blob = data + b"\n" + extra if extra else data

    signals: list[dict] = []

    # 1) Declarações explícitas de IA nos metadados.
    has_trained = _TRAINED in blob
    has_composite = _COMPOSITE in blob
    if has_trained:
        signals.append(
            _signal(
                "ai_declared",
                "Metadados declaram geração por IA",
                "O arquivo contém o marcador padrão «trainedAlgorithmicMedia» (IPTC/C2PA), usado para indicar "
                "conteúdo gerado por um modelo de IA.",
            )
        )
    if has_composite:
        signals.append(
            _signal(
                "ai_declared",
                "Metadados declaram edição com IA generativa",
                "O arquivo contém o marcador «compositeWithTrainedAlgorithmicMedia»: parte do conteúdo foi "
                "gerada ou alterada por IA, por exemplo com preenchimento generativo.",
            )
        )

    lowered = {k.lower(): v for k, v in texts.items()}
    params = lowered.get("parameters", "")
    if "Steps:" in params and ("Sampler" in params or "CFG scale" in params or "Seed:" in params):
        signals.append(
            _signal(
                "ai_declared",
                "Parâmetros de geração gravados no arquivo",
                "Há um bloco de texto com prompt e configurações (steps, sampler, seed), padrão de interfaces "
                "de Stable Diffusion.",
            )
        )
    elif "prompt" in lowered and ("workflow" in lowered or "class_type" in lowered["prompt"]):
        signals.append(
            _signal(
                "ai_declared",
                "Fluxo de geração gravado no arquivo",
                "O arquivo traz o grafo de nós de um gerador de imagens (formato do ComfyUI).",
            )
        )

    # 2) Menções a ferramentas geradoras em campos descritivos.
    software = exif_main.get("Software")
    if not any(s["kind"] == "ai_declared" for s in signals):
        candidates: list[tuple[str, str]] = []
        if software:
            candidates.append(("Software", _clean(software, _SCAN_LIMIT)))
        if exif_main.get("ImageDescription"):
            candidates.append(("ImageDescription", _clean(exif_main["ImageDescription"], _SCAN_LIMIT)))
        for key, value in texts.items():
            if key.lower() not in ("parameters", "prompt", "workflow"):
                candidates.append((key, value[:_SCAN_LIMIT]))
        for source, text in candidates:
            match = _GENERATOR.search(text)
            if match:
                signals.append(
                    _signal(
                        "ai_hint",
                        "Menção a uma ferramenta de geração",
                        f"O campo «{_clean(source, 40)}» cita «{_clean(match.group(0), 40)}». "
                        "É um indício, não uma prova: textos podem ser editados.",
                    )
                )
                break

    # 3) Dados de câmera.
    make, model = exif_main.get("Make"), exif_main.get("Model")
    if make or model:
        label = " ".join(_clean(x, 60) for x in (make, model) if x)
        present = []
        for key, name in _CAPTURE_FIELDS:
            if exif_ifd.get(key) not in (None, "") and name not in present:
                present.append(name)
        capture = f" Também constam dados de captura ({', '.join(present)})." if present else ""
        signals.append(
            _signal(
                "camera",
                "Metadados de câmera presentes",
                f"Equipamento registrado: {label}.{capture} Esses dados são fáceis de copiar ou editar; "
                "contam como indício fraco a favor de uma foto, não como prova.",
            )
        )

    # 4) Informações neutras.
    if software and not any(s["kind"] == "ai_hint" for s in signals):
        signals.append(
            _signal(
                "info",
                "Software registrado no arquivo",
                f"«{_clean(software, 80)}». Mostra por onde o arquivo passou, mas não diz se a imagem "
                "original foi capturada ou gerada.",
            )
        )
    if b"c2pa" in blob and b"jumb" in blob:
        signals.append(
            _signal(
                "info",
                "Credenciais de Conteúdo (C2PA) detectadas",
                "O arquivo traz um manifesto C2PA. Este aplicativo não verifica a assinatura digital; use uma "
                "ferramenta de verificação de Content Credentials para ver o emissor e o histórico.",
            )
        )
    if not signals:
        signals.append(
            _signal(
                "info",
                "Sem metadados informativos",
                "Não há dados de câmera nem marcas de geradores neste arquivo. Isso é comum em imagens enviadas "
                "por aplicativos de mensagem e redes sociais, que removem metadados, e não indica nada por si só.",
            )
        )

    signals.sort(key=lambda s: _ORDER[s["kind"]])
    return {"signals": signals, "declares_ai": any(s["kind"] == "ai_declared" for s in signals)}
