"""Regression and local HTTP contract tests; no model downloads or GPU required."""
import base64
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from atelier import benchmark, downloader, fileio, registry, settings
from atelier.engine import sdcpp, sdserver


class PlacementCompatibilityTests(unittest.TestCase):
    def args(self, enabled, help_text, options):
        with patch.object(sdcpp, "supported_options", return_value=frozenset(options)), \
             patch.object(sdcpp, "binary_help", return_value=help_text):
            return sdcpp.auto_fit_args(Path("engine"), enabled)

    def test_current_engine_receives_explicit_on_and_off(self):
        for enabled, value in [(True, "on"), (False, "off")]:
            self.assertEqual(self.args(enabled, "--auto-fit on|off (default: on)",
                                       {"--auto-fit"}), ["--auto-fit", value])

    def test_release_841_uses_boolean_syntax(self):
        self.assertEqual(self.args(True, "--auto-fit pick the device placements",
                                   {"--auto-fit"}), ["--auto-fit"])
        self.assertEqual(self.args(False, "--auto-fit pick the device placements",
                                   {"--auto-fit"}), [])

    def test_unsupported_requested_auto_fit_is_explained(self):
        with self.assertRaisesRegex(sdcpp.EngineError, "newer engine"):
            self.args(True, "", set())

    def test_cli_and_server_have_the_same_memory_policy(self):
        req = sdcpp.GenRequest(diffusion_model=Path("m.gguf"), max_vram="-1",
                              params_backend="diffusion=cpu,te=cpu,vae=cuda0",
                              stream_layers=True, flags={"clip_on_cpu": True,
                                                         "offload_to_cpu": True})
        opts = {"--auto-fit", "--params-backend", "--max-vram",
                "--disable-segmented-compute", "--disable-prefetch"}
        with patch.object(sdcpp, "supported_options", return_value=frozenset(opts)), \
             patch.object(sdcpp, "binary_help", return_value="--auto-fit on|off"), \
             patch.object(sdcpp, "_require"):
            memory = sdcpp.memory_args(Path("engine"), req)
            cli = sdcpp.build_gen_cmd(Path("engine"), req, Path("out.png"))
            server = sdserver.server_args(Path("engine"), req)
        self.assertNotIn("--stream-layers", memory)
        self.assertNotIn("--offload-to-cpu", memory)
        self.assertIn("--clip-on-cpu", memory)
        self.assertEqual(memory[-2:], ["--auto-fit", "off"])
        self.assertIn(" ".join(memory), " ".join(cli))
        self.assertIn(" ".join(memory), " ".join(server))

    def test_unsupported_cache_does_not_silently_change_the_image(self):
        with patch.object(sdcpp, "supported_options", return_value=frozenset()), \
             patch.object(sdcpp, "_require"):
            with self.assertRaisesRegex(sdcpp.EngineError, "cache setting"):
                sdcpp.build_gen_cmd(Path("engine"), sdcpp.GenRequest(cache_mode="easycache"), Path("x"))

    def test_transient_help_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "engine"
            binary.write_text("test")
            good = MagicMock(stdout="--auto-fit on|off", stderr="")
            with patch.object(sdcpp.subprocess, "run", side_effect=[OSError("busy"), good]) as run:
                self.assertEqual(sdcpp.supported_options(binary), frozenset())
                self.assertIn("--auto-fit", sdcpp.supported_options(binary))
                sdcpp.supported_options(binary)
                self.assertEqual(run.call_count, 2)


class CatalogAndDownloadTests(unittest.TestCase):
    def test_catalog_parses_once_but_reloads_edits_and_isolates_mutation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "CONFIG_DIR", Path(tmp)):
            p = Path(tmp) / "models.yaml"
            p.write_text("base_models: [{id: first}]\n")
            with patch.object(registry.yaml, "safe_load", wraps=registry.yaml.safe_load) as parse:
                registry._catalog()["base_models"][0]["id"] = "mutated"
                self.assertEqual(registry._catalog()["base_models"][0]["id"], "first")
                self.assertEqual(parse.call_count, 1)
                p.write_text("base_models: [{id: changed_on_disk}]\n")
                self.assertEqual(registry._catalog()["base_models"][0]["id"], "changed_on_disk")
                self.assertEqual(parse.call_count, 2)

    def test_installed_requested_quant_works_offline(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "MODELS_DIR", Path(tmp)):
            dest = settings.model_repo_dir("owner/repo") / "model-Q5_K_M.gguf"
            dest.parent.mkdir()
            dest.write_bytes(b"weights")
            comp = registry.Component("diffusion", "owner/repo", "model-{quant}.gguf", "Q5_K_M")
            with patch("huggingface_hub.list_repo_files", side_effect=AssertionError("network")), \
                 patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network")):
                self.assertEqual(downloader.download_component(comp), dest)

    def test_installed_lower_quant_does_not_block_requested_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch.object(settings, "configure_hf_env"):
            repo = settings.model_repo_dir("owner/repo")
            repo.mkdir()
            (repo / "model-Q4_K_M.gguf").write_bytes(b"old")
            target = repo / "model-Q5_K_M.gguf"
            comp = registry.Component("diffusion", "owner/repo", "model-{quant}.gguf", "Q5_K_M")
            with patch("huggingface_hub.list_repo_files", return_value=[target.name]), \
                 patch("huggingface_hub.hf_hub_download", return_value=str(target)) as download:
                self.assertEqual(downloader.download_component(comp), target)
                download.assert_called_once()

    def test_repository_listing_is_reused_within_one_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "MODELS_DIR", Path(tmp)), \
             patch.object(settings, "configure_hf_env"), \
             patch("huggingface_hub.list_repo_files", return_value=["a.gguf", "b.gguf"]) as listing, \
             patch("huggingface_hub.hf_hub_download", return_value=str(Path(tmp) / "got")):
            cache = {}
            for name in ["a.gguf", "b.gguf"]:
                downloader.download_component(registry.Component("diffusion", "owner/repo", name, None),
                                              _repo_files=cache)
            self.assertEqual(listing.call_count, 1)


class PersistenceAndBenchmarkTests(unittest.TestCase):
    def test_failed_preference_replace_preserves_original_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "preferences.json"
            dest.write_text('{"quant":"Q5_K_M"}')
            with patch.object(fileio.os, "replace", side_effect=OSError("locked")):
                with self.assertRaises(OSError):
                    fileio.atomic_write_text(dest, "new")
            self.assertEqual(json.loads(dest.read_text())["quant"], "Q5_K_M")
            self.assertEqual(list(Path(tmp).iterdir()), [dest])

    def test_benchmark_isolated_from_resident_and_cache_preferences(self):
        base = {"resident_engine": True, "cache_mode": "easycache", "flags": {}}
        measured = benchmark._benchmark_prefs(base, {})
        self.assertFalse(measured["resident_engine"])
        self.assertEqual(measured["cache_mode"], "")
        self.assertTrue(base["resident_engine"])

    def test_benchmark_uses_krea_native_eight_steps(self):
        with patch.object(benchmark.hardware, "used_vram_gb", return_value={}), \
             patch.object(benchmark.generate, "generate", return_value=[]) as gen, \
             patch("atelier.engine.release_resident_engine"):
            benchmark._one_run("krea2-turbo", {}, {"steps": 8}, None)
        self.assertEqual(gen.call_args.kwargs["steps"], 8)


class ResidentReliabilityTests(unittest.TestCase):
    def tearDown(self):
        sdserver._CANCEL.clear()

    def test_replacing_model_at_same_path_changes_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "weights.gguf"
            p.write_bytes(b"old")
            before = sdserver._file_signature([str(p)])
            p.write_bytes(b"updated-weights")
            self.assertNotEqual(before, sdserver._file_signature([str(p)]))

    def test_cancel_during_loading_terminates_and_waits(self):
        proc = MagicMock()
        with patch.object(sdserver, "_LIVE", sdserver._Live(proc, 7777, "key")):
            self.assertIn("cancelled", sdserver.cancel_active())
            proc.terminate.assert_called_once()
            proc.wait.assert_called_once()
            self.assertTrue(sdserver._CANCEL.is_set())

    def test_cancelled_connection_never_falls_back_to_cli(self):
        def cancelled(*args, **kwargs):
            sdserver._CANCEL.set()
            raise sdserver.ServerUnavailable("connection closed")
        with patch.object(sdserver, "_generate", side_effect=cancelled):
            with self.assertRaisesRegex(sdcpp.EngineError, "Interrupted") as ctx:
                sdserver.generate(Path("engine"), sdcpp.GenRequest(), Path("out"))
            self.assertNotIsInstance(ctx.exception, sdserver.ServerUnavailable)

    def test_timeout_releases_gpu_without_restarting_generation(self):
        with patch.object(sdserver, "ensure", return_value=7777), \
             patch.object(sdserver, "_request", return_value={"id": "job"}), \
             patch.object(sdserver, "JOB_TIMEOUT_S", -1), \
             patch.object(sdserver, "stop") as stop:
            with self.assertRaisesRegex(sdcpp.EngineError, "timed out"):
                sdserver.generate(Path("engine"), sdcpp.GenRequest(model_path=Path("m")), Path("out"))
            stop.assert_called_once()

    def test_second_request_cannot_replace_running_model(self):
        with sdserver._GENERATION_LOCK:
            with self.assertRaisesRegex(sdcpp.EngineError, "already generating"):
                sdserver.generate(Path("engine"), sdcpp.GenRequest(), Path("out"))


class ResidentHttpContractTests(unittest.TestCase):
    """Exercise the real local HTTP client and image I/O against native schema."""
    def test_reference_editing_round_trip_preserves_cfg_and_image(self):
        png = io.BytesIO()
        Image.new("RGB", (16, 16), "navy").save(png, format="PNG")
        image_bytes = png.getvalue()
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, obj):
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path.endswith("/capabilities"):
                    self.respond({"features_by_mode": {"img_gen": {"ref_images": True}}})
                else:
                    self.respond({"status": "completed", "result": {"images": [
                        {"b64_json": base64.b64encode(image_bytes).decode()}]}})

            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.respond({"id": "job-1", "status": "queued"})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ref, out = Path(tmp) / "ref.png", Path(tmp) / "out.png"
                ref.write_bytes(image_bytes)
                req = sdcpp.GenRequest(diffusion_model=Path("model.gguf"), ref_image=[ref],
                                      cfg_scale=4.5, negative="blur", seed=42)
                with patch.object(sdserver, "ensure", return_value=server.server_port), \
                     patch.object(sdserver, "_POLL_S", 0):
                    result = sdserver.generate(Path("engine"), req, out)
                self.assertEqual(result, [out])
                self.assertEqual(out.read_bytes(), image_bytes)
                self.assertEqual(seen[0]["sample_params"]["guidance"], {"txt_cfg": 4.5})
                self.assertEqual(seen[0]["negative_prompt"], "blur")
                self.assertEqual(base64.b64decode(seen[0]["ref_images"][0]), image_bytes)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
