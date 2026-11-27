"""Motor local de detecção de imagens geradas por IA.

Smaug é o modelo padrão. O classificador antigo baseado em Transformers continua
como opção explícita para comparação via MODEL_IDS, mas não é carregado por padrão.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from .scoring import short_name
from .smaug_model import MODEL_ID as SMAUG_MODEL_ID

log = logging.getLogger("vestigio.detector")

_AI_TOKENS = frozenset({"ai", "artificial", "fake", "generated", "synthetic", "aigenerated", "deepfake"})
_HUMAN_TOKENS = frozenset({"hum", "human", "real", "authentic", "natural", "genuine", "photo", "camera"})


class LabelMappingError(ValueError):
    """Não foi possível descobrir qual rótulo do modelo significa IA."""


class EngineNotReady(RuntimeError):
    pass


class EngineBusy(RuntimeError):
    pass


def find_ai_index(id2label: dict) -> int:
    """Descobre o índice de IA pelos rótulos; recusa adivinhar quando são ambíguos."""
    ai, human = [], []
    for index, label in id2label.items():
        tokens = set(re.findall(r"[a-z0-9]+", str(label).lower()))
        is_ai, is_human = bool(tokens & _AI_TOKENS), bool(tokens & _HUMAN_TOKENS)
        if is_ai and not is_human:
            ai.append(int(index))
        elif is_human and not is_ai:
            human.append(int(index))
    if len(id2label) != 2 or len(ai) != 1 or len(human) != 1:
        raise LabelMappingError(f"Não foi possível identificar o rótulo de IA em {dict(id2label)!r}.")
    return ai[0]


@dataclass
class _LoadedModel:
    model_id: str
    model: Any
    kind: str
    ai_index: int | None = None


class DetectionEngine:
    """Carrega modelos em segundo plano e serializa inferências para poupar memória."""

    def __init__(self, model_ids: tuple[str, ...], device: str = "auto"):
        self.model_ids = tuple(model_ids)
        self._device_pref = device
        self._device = None
        self._models: list[_LoadedModel] = []
        self._errors: dict[str, str] = {}
        self._state = "idle"
        self._state_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None:
                return
            self._state = "loading"
            self._thread = threading.Thread(target=self._load_all, name="model-loader", daemon=True)
            self._thread.start()

    def _finish(self, models: list[_LoadedModel], errors: dict[str, str], device) -> None:
        with self._state_lock:
            self._models, self._errors, self._device = models, errors, device
            if not models:
                self._state = "error"
            elif errors:
                self._state = "degraded"
            else:
                self._state = "ready"

    def _load_all(self) -> None:
        try:
            import torch
        except ImportError:
            log.exception("PyTorch não está instalado. Rode: pip install -r requirements.txt")
            self._finish([], {"*": "dependencies_missing"}, None)
            return

        device = self._resolve_device(torch)
        loaded: list[_LoadedModel] = []
        errors: dict[str, str] = {}
        for model_id in self.model_ids:
            try:
                log.info("Carregando %s; o primeiro download pode levar alguns minutos...", model_id)
                if model_id == SMAUG_MODEL_ID:
                    from .smaug_model import SmaugImageDetect

                    model = SmaugImageDetect.from_pretrained(model_id, device=device)
                    loaded.append(_LoadedModel(model_id, model, "smaug"))
                    log.info("Smaug pronto em %s.", device)
                    continue

                # Compatibilidade: modelos antigos continuam disponíveis se solicitados no .env.
                from transformers import AutoImageProcessor, AutoModelForImageClassification

                processor = AutoImageProcessor.from_pretrained(model_id)
                try:
                    classifier = AutoModelForImageClassification.from_pretrained(model_id)
                except (ValueError, KeyError):
                    from transformers import SiglipForImageClassification

                    classifier = SiglipForImageClassification.from_pretrained(model_id)
                classifier.to(device)
                classifier.eval()
                ai_index = find_ai_index(classifier.config.id2label)
                loaded.append(_LoadedModel(model_id, (processor, classifier), "transformers", ai_index))
                log.info("Modelo %s pronto em %s (rótulo IA: %s).", model_id, device, classifier.config.id2label[ai_index])
            except Exception:  # noqa: BLE001 - uma falha não derruba o servidor
                log.exception("Falha ao carregar o modelo %s", model_id)
                errors[model_id] = "load_failed"
        self._finish(loaded, errors, device)

    def _resolve_device(self, torch) -> str:
        wanted = self._device_pref
        if wanted == "cuda" and not torch.cuda.is_available():
            log.warning("DETECTOR_DEVICE=cuda, mas nenhuma GPU CUDA foi encontrada. Usando CPU.")
            return "cpu"
        if wanted == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return wanted

    def status(self) -> dict:
        with self._state_lock:
            loaded_ids = {model.model_id for model in self._models}
            return {
                "state": self._state,
                "device": self._device,
                "models": [
                    {
                        "id": model_id,
                        "name": short_name(model_id),
                        "loaded": model_id in loaded_ids,
                        "error": self._errors.get(model_id) or self._errors.get("*"),
                    }
                    for model_id in self.model_ids
                ],
            }

    def predict(self, image) -> list[dict]:
        with self._state_lock:
            models = list(self._models)
            device = self._device
        if not models:
            raise EngineNotReady("o motor de detecção ainda não está pronto")
        if not self._infer_lock.acquire(blocking=False):
            raise EngineBusy("já existe uma análise em andamento")
        try:
            import torch

            results = []
            for loaded in models:
                if loaded.kind == "smaug":
                    probability = loaded.model(image, crops=3)
                    labels = {"AI-generated": probability, "Not-AI-generated": 1.0 - probability}
                else:
                    processor, classifier = loaded.model
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        logits = classifier(**inputs).logits
                    probs = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
                    id2label = classifier.config.id2label
                    probability = float(probs[loaded.ai_index])
                    labels = {str(id2label[i]): float(p) for i, p in enumerate(probs)}

                results.append({
                    "model_id": loaded.model_id,
                    "labels": labels,
                    "ai_probability": float(probability),
                })
            return results
        finally:
            self._infer_lock.release()
