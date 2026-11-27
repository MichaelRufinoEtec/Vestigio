"""Normalização da resposta do detector e redação do veredito.

Princípio: a pontuação exibida é o score retornado pelo modelo para a classe «IA». Ela NÃO é uma
probabilidade calibrada nem a confiança de que o veredito esteja correto.
"""
from __future__ import annotations

import math

# Faixas usadas só para escrever o veredito. São convenção de leitura, não foram calibradas.
AI_HIGH = 0.80
AI_LOW = 0.20

DISCLAIMER = (
    "Este resultado é uma estimativa. Nenhum detector identifica corretamente todas as imagens geradas "
    "por IA, e nenhum garante que uma fotografia seja autêntica."
)

READING_NOTES = (
    "A pontuação é o score produzido pelo modelo para a classe «IA». Não é uma probabilidade calibrada nem a "
    "chance de o veredito estar correto. Modelos podem errar com mais convicção do que o número sugere.",
    "A classe oposta («humana») reúne, nos dados de treino, fotografias reais e obras feitas por humanos. "
    "Uma pontuação baixa não prova que a imagem seja uma foto original.",
    "Capturas de tela, redução de tamanho, recompressão e edições alteram o resultado. Imagens de geradores "
    "mais novos que os dados de treino do modelo podem passar despercebidas.",
    "Os limites de 20% e 80% usados para escrever o veredito são uma convenção conservadora deste aplicativo; não foram "
    "calibrados com dados de validação.",
)


class BadDetectorResponse(ValueError):
    """O motor devolveu algo fora do formato esperado."""


def short_name(model_id: str) -> str:
    return model_id.replace("\\", "/").rstrip("/").split("/")[-1] or model_id


def fmt_pct(p: float) -> str:
    value = p * 100
    if value >= 99.95:
        return "mais de 99,9%"
    if value < 0.05:
        return "menos de 0,1%"
    return f"{value:.1f}".replace(".", ",") + "%"


def classify(p: float) -> str:
    if p >= AI_HIGH:
        return "ai"
    if p <= AI_LOW:
        return "human"
    return "uncertain"


def validate_model_results(raw) -> list[dict]:
    """Confere o formato devolvido pelo motor e o converte para a estrutura da API."""
    if not isinstance(raw, list) or not raw:
        raise BadDetectorResponse("lista de resultados vazia ou inválida")

    models = []
    for item in raw:
        if not isinstance(item, dict):
            raise BadDetectorResponse("item de resultado inválido")
        model_id = item.get("model_id")
        labels = item.get("labels")
        p = item.get("ai_probability")
        if not isinstance(model_id, str) or not model_id:
            raise BadDetectorResponse("model_id ausente")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise BadDetectorResponse("ai_probability fora do intervalo [0, 1]")
        if not isinstance(labels, dict) or not labels:
            raise BadDetectorResponse("rótulos ausentes")
        clean_labels = {}
        for name, value in labels.items():
            if (
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise BadDetectorResponse("rótulo com valor inválido")
            clean_labels[name] = round(float(value), 6)
        p = float(p)
        models.append(
            {
                "id": model_id,
                "name": short_name(model_id),
                "ai_probability": round(p, 6),
                "human_probability": round(1.0 - p, 6),
                "raw_labels": clean_labels,
                "classification": classify(p),
            }
        )
    return models


def _join_pt(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " e " + items[-1]


def build_verdict(models: list[dict], declares_ai: bool) -> dict:
    classes = [m["classification"] for m in models]
    unique = set(classes)
    single = len(models) == 1
    scores = _join_pt([fmt_pct(m["ai_probability"]) for m in models])

    if single:
        agreement = "single"
    elif len(unique) == 1:
        agreement = "agree"
    else:
        agreement = "disagree"

    if declares_ai:
        code = "declared_ai"
    elif unique == {"ai"}:
        code = "ai"
    elif unique == {"human"}:
        code = "camera"
    else:
        code = "inconclusive"

    subject = "O modelo atribuiu" if single else "Os modelos atribuíram"

    if code == "declared_ai":
        headline = "Os metadados declaram o uso de IA"
        summary = (
            "O próprio arquivo registra o uso de IA generativa (veja «Metadados do arquivo»). Metadados podem "
            "ser apagados ou forjados, mas, quando existem, são uma declaração explícita. "
            f"{subject} um score de {scores} para a classe «IA»."
        )
    elif code == "ai":
        headline = "Mais semelhante à classe IA"
        if single:
            summary = (
                f"O modelo atribuiu um score de {scores} para a classe «IA»: a imagem se parece com as imagens "
                "geradas que ele viu no treinamento. É um indício, não uma prova de origem."
            )
        else:
            summary = (
                f"Os {len(models)} modelos atribuíram um score de {scores} para a classe «IA»: a imagem se parece "
                "com as imagens geradas que eles viram no treinamento. É um indício, não uma prova de origem."
            )
    elif code == "camera":
        headline = "Mais semelhante à classe humana"
        who = "O modelo atribuiu" if single else f"Os {len(models)} modelos atribuíram"
        summary = (
            f"{who} um score de {scores} para a classe «IA»: a imagem se parece mais com exemplos classificados "
            "como humanos nos dados de treino, que incluem fotografias e obras humanas. Isso não comprova que "
            "tenha sido capturada por uma câmera nem confirma sua origem."
        )
    elif agreement == "disagree":
        headline = "Resultado inconclusivo"
        summary = (
            f"Os modelos divergem entre si (scores para IA: {scores}). Quando as opiniões se dividem, "
            "o resultado mais honesto é não decidir."
        )
    else:
        headline = "Resultado inconclusivo"
        summary = (
            f"O score para IA ({scores}) ficou na faixa intermediária, em que o modelo não separa bem as "
            "duas classes."
        )

    return {"code": code, "headline": headline, "summary": summary, "agreement": agreement}
