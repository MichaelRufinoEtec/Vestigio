"""Testes automatizados do backend. Usam um motor FALSO no lugar do modelo real.

Eles verificam validações, formato da API, tratamento de erros e lógica do veredito.
NÃO verificam a qualidade do modelo nem o download dele: para isso, veja o roteiro no README.

Rodar:  python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import io
import os
import unittest

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from backend.app import create_app
from backend.config import Config, DEFAULT_MODEL_ID
from backend.detector import EngineBusy, LabelMappingError, find_ai_index
from backend.scoring import (
    BadDetectorResponse,
    build_verdict,
    classify,
    fmt_pct,
    validate_model_results,
)


def make_image(fmt="PNG", size=(128, 96), color=(120, 90, 60), **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def raw_result(p: float, model_id="org/modelo-a"):
    return {"model_id": model_id, "labels": {"ai": p, "hum": 1 - p}, "ai_probability": p}


class FakeEngine:
    """Imita a interface de DetectionEngine (status e predict)."""

    def __init__(self, state="ready", result=None, error=None):
        self.state = state
        self.result = result if result is not None else [raw_result(0.97)]
        self.error = error
        self.calls = 0

    def status(self):
        return {
            "state": self.state,
            "device": "cpu",
            "models": [{"id": "org/modelo-a", "name": "modelo-a", "loaded": self.state == "ready", "error": None}],
        }

    def predict(self, image):
        self.calls += 1
        assert image.mode == "RGB"
        if self.error:
            raise self.error
        return self.result


class ApiTests(unittest.TestCase):
    def client(self, engine=None, **config):
        engine = engine or FakeEngine()
        app = create_app(Config(**config), engine=engine)
        return app.test_client(), engine

    def post(self, client, data, name="foto.png", mime="image/png", field="image"):
        return client.post("/api/analyze", data={field: (io.BytesIO(data), name, mime)}, content_type="multipart/form-data")

    # ---- caminho feliz --------------------------------------------------------
    def test_valid_image_returns_full_payload(self):
        client, engine = self.client()
        res = self.post(client, make_image())
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["verdict"]["code"], "ai")
        self.assertEqual(body["verdict"]["headline"], "Mais semelhante à classe IA")
        self.assertEqual(body["verdict"]["agreement"], "single")
        self.assertIn("score de", body["verdict"]["summary"])
        self.assertNotIn("probabilidade", body["verdict"]["summary"])
        self.assertAlmostEqual(body["models"][0]["ai_probability"], 0.97)
        self.assertEqual(body["models"][0]["classification"], "ai")
        self.assertEqual(body["image"]["format"], "PNG")
        self.assertEqual((body["image"]["width"], body["image"]["height"]), (128, 96))
        self.assertTrue(body["metadata"]["signals"])
        self.assertIn("estimativa", body["disclaimer"])
        self.assertEqual(engine.calls, 1)

    def test_low_score_is_camera_and_middle_is_inconclusive(self):
        client, _ = self.client(FakeEngine(result=[raw_result(0.03)]))
        verdict = self.post(client, make_image()).get_json()["verdict"]
        self.assertEqual(verdict["code"], "camera")  # Mantido para compatibilidade da API.
        self.assertEqual(verdict["headline"], "Mais semelhante à classe humana")
        self.assertIn("não comprova que tenha sido capturada por uma câmera", verdict["summary"])
        self.assertIn("score de", verdict["summary"])
        client, _ = self.client(FakeEngine(result=[raw_result(0.55)]))
        self.assertEqual(self.post(client, make_image()).get_json()["verdict"]["code"], "inconclusive")

    def test_models_that_disagree_are_inconclusive(self):
        result = [raw_result(0.97, "org/modelo-a"), raw_result(0.05, "org/modelo-b")]
        client, _ = self.client(FakeEngine(result=result))
        body = self.post(client, make_image()).get_json()
        self.assertEqual(body["verdict"]["code"], "inconclusive")
        self.assertEqual(body["verdict"]["agreement"], "disagree")
        self.assertEqual(len(body["models"]), 2)

    def test_jpeg_and_webp_and_rgba_png_are_accepted(self):
        client, _ = self.client()
        for fmt, name in (("JPEG", "a.jpg"), ("WEBP", "a.webp")):
            self.assertEqual(self.post(client, make_image(fmt), name).status_code, 200, fmt)
        rgba = io.BytesIO()
        Image.new("RGBA", (100, 100), (255, 0, 0, 0)).save(rgba, "PNG")
        self.assertEqual(self.post(client, rgba.getvalue()).status_code, 200)

    # ---- validação de arquivo ----------------------------------------------------
    def test_text_file_renamed_as_png_is_rejected_by_content(self):
        client, engine = self.client()
        res = self.post(client, b"isto nao e uma imagem", name="falsa.png", mime="image/png")
        self.assertEqual(res.status_code, 415)
        self.assertEqual(res.get_json()["error"]["code"], "UNSUPPORTED_TYPE")
        self.assertEqual(engine.calls, 0)

    def test_gif_is_rejected_even_though_it_is_an_image(self):
        buf = io.BytesIO()
        Image.new("P", (100, 100)).save(buf, "GIF")
        client, _ = self.client()
        res = self.post(client, buf.getvalue(), name="a.gif", mime="image/gif")
        self.assertEqual(res.status_code, 415)
        self.assertIn("GIF", res.get_json()["error"]["message"])

    def test_file_over_limit_is_rejected(self):
        client, engine = self.client(max_upload_bytes=20_000)
        noisy = io.BytesIO()
        Image.frombytes("RGB", (200, 200), os.urandom(200 * 200 * 3)).save(noisy, "PNG")  # ruído não comprime
        payload = noisy.getvalue()
        self.assertGreater(len(payload), 20_000)
        res = self.post(client, payload)
        self.assertEqual(res.status_code, 413)
        self.assertEqual(res.get_json()["error"]["code"], "FILE_TOO_LARGE")
        self.assertEqual(engine.calls, 0)

    def test_request_far_over_limit_is_cut_off_by_the_server(self):
        client, engine = self.client(max_upload_bytes=20_000)
        res = self.post(client, os.urandom(400_000))
        self.assertEqual(res.status_code, 413)
        self.assertEqual(res.get_json()["error"]["code"], "FILE_TOO_LARGE")
        self.assertEqual(engine.calls, 0)

    def test_missing_field_empty_file_and_tiny_image(self):
        client, _ = self.client()
        res = client.post("/api/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual((res.status_code, res.get_json()["error"]["code"]), (400, "NO_FILE"))
        res = self.post(client, b"")
        self.assertEqual(res.status_code, 400)
        res = self.post(client, make_image(size=(20, 20)))
        self.assertEqual((res.status_code, res.get_json()["error"]["code"]), (422, "IMAGE_TOO_SMALL"))

    def test_truncated_png_is_rejected_as_invalid(self):
        data = make_image(size=(300, 300))
        client, _ = self.client()
        res = self.post(client, data[: len(data) // 2])
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.get_json()["error"]["code"], "INVALID_IMAGE")

    def test_pixel_limit(self):
        client, _ = self.client(max_pixels=10_000)
        res = self.post(client, make_image(size=(200, 200)))
        self.assertEqual((res.status_code, res.get_json()["error"]["code"]), (422, "IMAGE_DIMENSIONS_TOO_LARGE"))

    # ---- motor / detector -----------------------------------------------------------
    def test_engine_loading_and_error_states(self):
        client, engine = self.client(FakeEngine(state="loading"))
        res = self.post(client, make_image())
        self.assertEqual((res.status_code, res.get_json()["error"]["code"]), (503, "ENGINE_LOADING"))
        client, engine = self.client(FakeEngine(state="error"))
        res = self.post(client, make_image())
        self.assertEqual((res.status_code, res.get_json()["error"]["code"]), (503, "ENGINE_UNAVAILABLE"))
        self.assertEqual(engine.calls, 0)

    def test_busy_engine_returns_429(self):
        client, _ = self.client(FakeEngine(error=EngineBusy("x")))
        res = self.post(client, make_image())
        self.assertEqual((res.status_code, res.get_json()["error"]["code"]), (429, "BUSY"))

    def test_unexpected_exception_does_not_leak_internals(self):
        client, _ = self.client(FakeEngine(error=RuntimeError("SEGREDO C:\\caminho\\interno traceback")))
        res = self.post(client, make_image())
        self.assertEqual(res.status_code, 500)
        text = res.get_data(as_text=True)
        self.assertEqual(res.get_json()["error"]["code"], "INFERENCE_FAILED")
        self.assertNotIn("SEGREDO", text)
        self.assertNotIn("caminho", text)

    def test_malformed_detector_output_is_rejected(self):
        bad_outputs = [
            [],
            "texto",
            [{"model_id": "m", "labels": {"ai": 0.5}, "ai_probability": float("nan")}],
            [{"model_id": "m", "labels": {"ai": 0.5}, "ai_probability": 1.7}],
            [{"model_id": "m", "labels": {}, "ai_probability": 0.5}],
            [{"labels": {"ai": 0.5}, "ai_probability": 0.5}],
            [{"model_id": "m", "labels": {"ai": 0.5}, "ai_probability": True}],
        ]
        for bad in bad_outputs:
            client, _ = self.client(FakeEngine(result=bad))
            res = self.post(client, make_image())
            self.assertEqual((res.status_code, res.get_json()["error"]["code"]), (502, "DETECTOR_BAD_RESPONSE"), bad)

    # ---- metadados --------------------------------------------------------------
    def test_stable_diffusion_parameters_in_png_are_reported_as_declared(self):
        info = PngInfo()
        info.add_text("parameters", "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1")
        buf = io.BytesIO()
        Image.new("RGB", (128, 128), (9, 9, 9)).save(buf, "PNG", pnginfo=info)
        client, _ = self.client(FakeEngine(result=[raw_result(0.02)]))  # o modelo diz "câmera"
        body = self.post(client, buf.getvalue()).get_json()
        self.assertEqual(body["verdict"]["code"], "declared_ai")
        self.assertEqual(body["metadata"]["signals"][0]["kind"], "ai_declared")
        self.assertEqual(body["verdict"]["agreement"], "single")

    def test_iptc_marker_in_compressed_png_text_is_found(self):
        info = PngInfo()
        info.add_itxt("XML:com.adobe.xmp", "<x>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</x>", zip=True)
        buf = io.BytesIO()
        Image.new("RGB", (128, 128)).save(buf, "PNG", pnginfo=info)
        client, _ = self.client()
        body = self.post(client, buf.getvalue()).get_json()
        self.assertEqual(body["verdict"]["code"], "declared_ai")

    def test_camera_exif_is_a_weak_signal_and_does_not_change_verdict(self):
        exif = Image.Exif()
        exif[0x010F] = "Canon"
        exif[0x0110] = "EOS R5"
        buf = io.BytesIO()
        Image.new("RGB", (128, 128)).save(buf, "JPEG", exif=exif)
        client, _ = self.client(FakeEngine(result=[raw_result(0.5)]))
        body = self.post(client, buf.getvalue(), "a.jpg", "image/jpeg").get_json()
        kinds = [s["kind"] for s in body["metadata"]["signals"]]
        self.assertIn("camera", kinds)
        self.assertEqual(body["verdict"]["code"], "inconclusive")

    def test_plain_image_gets_neutral_no_metadata_note(self):
        client, _ = self.client()
        signals = self.post(client, make_image()).get_json()["metadata"]["signals"]
        self.assertEqual([s["kind"] for s in signals], ["info"])

    # ---- demais rotas e cabeçalhos ----------------------------------------------------
    def test_health_reports_engine_and_limits(self):
        client, _ = self.client(max_upload_bytes=5 * 1024 * 1024)
        body = client.get("/api/health").get_json()
        self.assertEqual(body["engine"]["state"], "ready")
        self.assertEqual(body["limits"]["max_upload_bytes"], 5 * 1024 * 1024)
        self.assertIn("image/png", body["limits"]["mime_types"])

    def test_index_and_static_files_are_served_with_security_headers(self):
        client, _ = self.client()
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        self.assertIn("script-src 'self'", res.headers["Content-Security-Policy"])
        self.assertEqual(res.headers["X-Content-Type-Options"], "nosniff")
        for path in ("/static/css/styles.css", "/static/js/app.js", "/static/favicon.svg"):
            self.assertEqual(client.get(path).status_code, 200, path)

    def test_unknown_api_route_and_wrong_method_return_json(self):
        client, _ = self.client()
        self.assertEqual(client.get("/api/nada").get_json()["error"]["code"], "NOT_FOUND")
        self.assertEqual(client.get("/api/analyze").status_code, 405)

    def test_api_responses_are_not_cached(self):
        client, _ = self.client()
        self.assertEqual(client.get("/api/health").headers["Cache-Control"], "no-store")


class UnitTests(unittest.TestCase):
    def test_smaug_is_the_default_model(self):
        self.assertEqual(DEFAULT_MODEL_ID, "abacusai/Smaug-AI-Image-Detector")
        self.assertEqual(Config().model_ids, (DEFAULT_MODEL_ID,))


    def test_find_ai_index_with_common_label_schemes(self):
        self.assertEqual(find_ai_index({0: "ai", 1: "hum"}), 0)
        self.assertEqual(find_ai_index({0: "hum", 1: "ai"}), 1)
        self.assertEqual(find_ai_index({0: "artificial", 1: "human"}), 0)
        self.assertEqual(find_ai_index({0: "REAL", 1: "FAKE"}), 1)
        self.assertEqual(find_ai_index({0: "human", 1: "AI-generated"}), 1)
        self.assertEqual(find_ai_index({"0": "AI", "1": "HUMAN"}), 0)

    def test_find_ai_index_refuses_to_guess(self):
        for labels in ({0: "cat", 1: "dog"}, {0: "a", 1: "b", 2: "c"}, {0: "ai", 1: "fake"}, {0: "real", 1: "human"}):
            with self.assertRaises(LabelMappingError, msg=str(labels)):
                find_ai_index(labels)

    def test_classify_thresholds(self):
        self.assertEqual(classify(0.80), "ai")
        self.assertEqual(classify(0.79), "uncertain")
        self.assertEqual(classify(0.20), "human")
        self.assertEqual(classify(0.21), "uncertain")

    def test_percent_formatting_never_claims_certainty(self):
        self.assertEqual(fmt_pct(0.973), "97,3%")
        self.assertEqual(fmt_pct(0.99999), "mais de 99,9%")
        self.assertEqual(fmt_pct(0.00001), "menos de 0,1%")

    def test_validate_results_normalizes(self):
        out = validate_model_results([raw_result(0.9, "dono/modelo-x")])
        self.assertEqual(out[0]["name"], "modelo-x")
        self.assertAlmostEqual(out[0]["human_probability"], 0.1)
        with self.assertRaises(BadDetectorResponse):
            validate_model_results(None)

    def test_declared_ai_overrides_camera_score_but_mentions_the_model(self):
        models = validate_model_results([raw_result(0.02)])
        verdict = build_verdict(models, declares_ai=True)
        self.assertEqual(verdict["code"], "declared_ai")
        self.assertIn("2,0%", verdict["summary"])


if __name__ == "__main__":
    unittest.main()
