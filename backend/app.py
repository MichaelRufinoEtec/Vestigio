"""Aplicação Flask do Vestígio: serve o frontend e expõe a API de análise."""
from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, FRONTEND_DIR, Config
from .detector import DetectionEngine, EngineBusy, EngineNotReady
from .imaging import ImageRejected, decode_upload
from .metadata import extract_signals
from .scoring import (
    DISCLAIMER,
    READING_NOTES,
    BadDetectorResponse,
    build_verdict,
    validate_model_results,
)

log = logging.getLogger("vestigio.app")

_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' blob: data:; "
    "connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def _human_size(n: int) -> str:
    if n >= 1024 * 1024:
        mb = n / (1024 * 1024)
        return f"{mb:.0f} MB" if mb == int(mb) else f"{mb:.1f} MB".replace(".", ",")
    return f"{max(1, round(n / 1024))} KB"


def _error(status: int, code: str, message: str):
    response = jsonify({"error": {"code": code, "message": message}})
    response.status_code = status
    return response


def create_app(config: Config | None = None, engine=None) -> Flask:
    config = config or Config.from_env()
    if engine is None:
        engine = DetectionEngine(config.model_ids, config.device)
        engine.start()

    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")
    app.json.ensure_ascii = False
    # Margem para o "envelope" multipart; o limite exato do arquivo é conferido na rota.
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes + 64 * 1024
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.extensions["engine"] = engine

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "engine": engine.status(),
                "limits": {
                    "max_upload_bytes": config.max_upload_bytes,
                    "mime_types": list(ALLOWED_MIME_TYPES),
                    "extensions": list(ALLOWED_EXTENSIONS),
                    "min_side": config.min_side,
                },
            }
        )

    @app.post("/api/analyze")
    def analyze():
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]

        state = engine.status()["state"]
        if state in ("idle", "loading"):
            return _error(
                503,
                "ENGINE_LOADING",
                "O modelo de detecção ainda está sendo carregado. Na primeira execução ele é baixado; aguarde e tente de novo.",
            )
        if state == "error":
            return _error(
                503,
                "ENGINE_UNAVAILABLE",
                "O modelo de detecção não pôde ser carregado. Veja a mensagem de erro no terminal onde o servidor está rodando.",
            )

        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return _error(400, "NO_FILE", "Nenhuma imagem foi enviada. Envie um arquivo no campo «image».")

        data = upload.stream.read(config.max_upload_bytes + 1)
        if len(data) > config.max_upload_bytes:
            return _error(
                413, "FILE_TOO_LARGE", f"O arquivo excede o limite de {_human_size(config.max_upload_bytes)}."
            )

        try:
            decoded = decode_upload(data, max_pixels=config.max_pixels, min_side=config.min_side)
        except ImageRejected as exc:
            return _error(exc.status, exc.code, exc.message)

        meta = extract_signals(data)

        try:
            raw = engine.predict(decoded.image)
        except EngineBusy:
            return _error(429, "BUSY", "Já existe uma análise em andamento. Aguarde alguns segundos e tente de novo.")
        except EngineNotReady:
            return _error(503, "ENGINE_LOADING", "O modelo de detecção ainda não está pronto.")
        except Exception:  # noqa: BLE001 - detalhes ficam no log, não na resposta
            log.exception("Falha na inferência (request %s)", request_id)
            return _error(
                500, "INFERENCE_FAILED", f"A análise falhou por um erro interno (referência {request_id})."
            )

        try:
            models = validate_model_results(raw)
        except BadDetectorResponse as exc:
            log.error("Resposta inválida do detector (request %s): %s", request_id, exc)
            return _error(
                502, "DETECTOR_BAD_RESPONSE", f"O detector devolveu uma resposta inesperada (referência {request_id})."
            )

        verdict = build_verdict(models, meta["declares_ai"])
        return jsonify(
            {
                "request_id": request_id,
                "verdict": verdict,
                "models": models,
                "metadata": {"signals": meta["signals"]},
                "image": {
                    "format": decoded.format,
                    "width": decoded.width,
                    "height": decoded.height,
                    "size_bytes": len(data),
                },
                "notes": list(READING_NOTES),
                "disclaimer": DISCLAIMER,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return _error(413, "FILE_TOO_LARGE", f"O arquivo excede o limite de {_human_size(config.max_upload_bytes)}.")

    @app.errorhandler(404)
    def not_found(_exc):
        if request.path.startswith("/api/"):
            return _error(404, "NOT_FOUND", "Endereço não encontrado.")
        return "Página não encontrada.", 404

    @app.errorhandler(405)
    def method_not_allowed(_exc):
        return _error(405, "METHOD_NOT_ALLOWED", "Método não permitido para este endereço.")

    @app.errorhandler(Exception)
    def unexpected(exc):
        if isinstance(exc, HTTPException):
            return exc
        log.exception("Erro não tratado")
        return _error(500, "INTERNAL_ERROR", "Ocorreu um erro interno.")

    return app
