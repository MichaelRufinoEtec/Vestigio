"""Adaptador local para abacusai/Smaug-AI-Image-Detector.

A lógica de preparação de recortes e inferência segue a implementação pública
smaug_image_detect.py do repositório do modelo (Apache-2.0), mas fica versionada
neste projeto para não executar código Python baixado dinamicamente do Hub.
Modelo e pesos continuam sendo baixados do Hugging Face na primeira execução.
"""
from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageOps

MODEL_ID = "abacusai/Smaug-AI-Image-Detector"
CROP = 528
SIDE = 576
BUDGET = 140


def make_views(image: Image.Image, crops: int = 3) -> list[Image.Image]:
    """Cria os recortes de resolução nativa esperados pelo modelo Smaug."""
    if crops not in (1, 3):
        raise ValueError("crops deve ser 1 ou 3")

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    shorter = min(width, height)

    if shorter > SIDE:
        scale = SIDE / shorter
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    elif shorter < CROP:
        scale = CROP / shorter
        image = image.resize((math.ceil(width * scale), math.ceil(height * scale)), Image.Resampling.BICUBIC)

    width, height = image.size
    center_x, center_y = (width - CROP) // 2, (height - CROP) // 2
    if crops == 1:
        return [image.crop((center_x, center_y, center_x + CROP, center_y + CROP))]
    if width >= height:
        offsets = (0, center_x, width - CROP)
        return [image.crop((x, center_y, x + CROP, center_y + CROP)) for x in offsets]
    offsets = (0, center_y, height - CROP)
    return [image.crop((center_x, y, center_x + CROP, y + CROP)) for y in offsets]


class SmaugImageDetect:
    """Carrega o modelo oficial via SentenceTransformers e retorna score IA [0, 1]."""

    def __init__(self, sentence_model: Any):
        self.sentence_model = sentence_model
        image_processor = sentence_model[0].processor.image_processor
        image_processor.max_soft_tokens = BUDGET
        image_processor.image_seq_length = BUDGET

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID, device: str = "cpu") -> "SmaugImageDetect":
        import torch
        from sentence_transformers import SentenceTransformer

        # O autor recomenda BF16 em CUDA e FP32 em CPU/MPS; FP16 é instável neste modelo.
        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        sentence_model = SentenceTransformer(
            model_id,
            device=device,
            config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": dtype},
        )
        return cls(sentence_model)

    def __call__(self, image: Image.Image, crops: int = 3) -> float:
        import torch

        views = make_views(image, crops=crops)
        batch = [{"image": view} for view in views]
        with torch.inference_mode():
            logits = self.sentence_model.encode(batch, batch_size=len(batch), convert_to_tensor=True)
            score = torch.sigmoid(logits.float().reshape(1, len(batch)).mean(dim=1))[0].item()
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Smaug retornou um score inválido")
        return float(score)
