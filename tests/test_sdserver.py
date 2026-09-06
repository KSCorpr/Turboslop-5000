"""Moteur résident : ce qui part au démarrage, ce qui part par requête.

Le serveur ne peut pas tourner ici (il lui faut le binaire et 8 Go de poids),
mais tout ce qui décide À SA PLACE se teste : la frontière entre les arguments
de démarrage et ceux de la requête — c'est elle qui permet au modèle de rester
chargé —, le périmètre qu'il accepte, et le repli sur sd-cli, qui doit rester
la voie normale dès que quelque chose cloche.
"""
import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier.engine import generate, sdcpp, sdserver

ROOT = Path(__file__).resolve().parent.parent


def _req(**kwargs):
    base = dict(diffusion_model=Path("krea2.gguf"), vae=Path("ae.sft"),
                text_encoder=Path("qwen.gguf"), prompt="un chat",
                steps=8, cfg_scale=1.0, sampler="euler", width=1024,
                height=768, seed=42, batch_count=1)
    base.update(kwargs)
    return sdcpp.GenRequest(**base)


class ScopeTests(unittest.TestCase):
    """Ce qu'on refuse de servir, on le refuse EXPLICITEMENT."""

    def test_a_plain_generation_is_served(self):
        self.assertTrue(sdserver.can_serve(_req()))

    def test_img2img_is_served(self):
        self.assertTrue(sdserver.can_serve(_req(init_image=Path("a.png"))))

    def test_loras_go_back_to_the_command_line(self):
        # L'API ignore délibérément les balises <lora:…> du prompt : servir
        # cette demande produirait une image SANS le LoRA, sans rien dire.
        self.assertFalse(sdserver.can_serve(_req(lora_dir=Path("loras"))))

    def test_hd_pass_falls_back_but_reference_editing_can_reuse_the_model(self):
        self.assertFalse(sdserver.can_serve(_req(hires=object())))
        self.assertTrue(sdserver.can_serve(_req(ref_image=Path("r.png"))))

    def test_auto_fit_and_step_cache_go_back_too(self):
        self.assertFalse(sdserver.can_serve(_req(auto_fit=True)))
        self.assertFalse(sdserver.can_serve(_req(cache_mode="easycache")))


class StartupArgsTests(unittest.TestCase):
    def _args(self, req, options=("--params-backend", "--max-vram")):
        with patch.object(sdcpp, "supported_options",
                          return_value=frozenset(options)):
            return sdserver.server_args(Path("sd-server"), req)

    def test_the_model_is_described_at_startup(self):
        args = self._args(_req())
        self.assertEqual(args[args.index("--diffusion-model") + 1], "krea2.gguf")
        self.assertEqual(args[args.index("--llm") + 1], "qwen.gguf")
        self.assertEqual(args[args.index("--vae") + 1], "ae.sft")

    def test_nothing_that_changes_per_image_is_baked_in(self):
        # C'est TOUTE la raison d'être du serveur : si le prompt ou la graine
        # partaient au démarrage, il faudrait le relancer à chaque image.
        args = self._args(_req())
        for leaked in ("-p", "un chat", "-s", "42", "--steps", "-W", "-H"):
            self.assertNotIn(leaked, args)

    def test_memory_residency_is_a_startup_decision(self):
        args = self._args(_req(params_backend="diffusion=cuda0,vae=cuda0,te=cpu",
                               max_vram="10"))
        self.assertEqual(args[args.index("--params-backend") + 1],
                         "diffusion=cuda0,vae=cuda0,te=cpu")
        self.assertEqual(args[args.index("--max-vram") + 1], "10")

    def test_an_old_binary_gets_neither_option(self):
        args = self._args(_req(params_backend="diffusion=cuda0", max_vram="10"),
                          options=())
        self.assertNotIn("--params-backend", args)
        self.assertNotIn("--max-vram", args)


class PayloadTests(unittest.TestCase):
    def test_sampling_goes_into_sample_params(self):
        payload = sdserver.request_payload(
            _req(schedule="karras", flow_shift=3.0))
        self.assertEqual(payload["sample_params"], {
            "sample_method": "euler", "sample_steps": 8, "guidance": {"txt_cfg": 1.0},
            "scheduler": "karras", "flow_shift": 3.0})
        self.assertEqual(payload["width"], 1024)
        self.assertEqual(payload["batch_count"], 1)

    def test_the_negative_prompt_follows_the_same_rule_as_the_cli(self):
        # Sous CFG 1, sd.cpp ne calcule pas la branche non conditionnée : un
        # négatif envoyé quand même donnerait une image différente du mode CLI.
        self.assertNotIn("negative_prompt",
                         sdserver.request_payload(_req(negative="flou")))
        payload = sdserver.request_payload(_req(negative="flou", cfg_scale=5.0))
        self.assertEqual(payload["negative_prompt"], "flou")

    def test_an_init_image_travels_as_base64(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.png"
            Image.new("RGB", (8, 8), "navy").save(src)
            payload = sdserver.request_payload(
                _req(init_image=src, strength=0.4))
        self.assertEqual(payload["strength"], 0.4)
        self.assertTrue(base64.b64decode(payload["init_image"])
                        .startswith(b"\x89PNG"))


class ResultTests(unittest.TestCase):
    def _images(self, count):
        return [{"index": i, "b64_json": base64.b64encode(
            f"img{i}".encode()).decode()} for i in range(count)]

    def test_a_single_image_takes_the_expected_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "krea2-1.png"
            written = sdserver._write_images(self._images(1), out, 1)
        self.assertEqual(written, [out])

    def test_a_batch_is_named_so_collect_outputs_finds_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "krea2-1.png"
            sdserver._write_images(self._images(3), out, 3)
            found = sdcpp.collect_outputs(out, 3)
        self.assertEqual([p.name for p in found],
                         ["krea2-1_1.png", "krea2-1_2.png", "krea2-1_3.png"])


class JobLoopTests(unittest.TestCase):
    """Le suivi de tâche : trois issues, trois comportements distincts."""

    def _run(self, states):
        seen = iter(states)
        calls = []

        def fake_request(url, payload=None, timeout=30.0):
            calls.append(url)
            if url.endswith("/img_gen"):
                return {"id": "job-1", "status": "queued"}
            return next(seen)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "krea2-1.png"
            with patch.object(sdserver, "ensure", return_value=7777), \
                 patch.object(sdserver, "_request", side_effect=fake_request), \
                 patch.object(sdserver, "_POLL_S", 0):
                result = sdserver.generate(Path("sd-server"), _req(), out)
            # Le dossier temporaire disparaît à la sortie du bloc : on relit
            # ici, sinon l'assertion porterait sur un fichier déjà effacé.
            return result, out, (out.read_bytes() if out.is_file() else b"")

    def test_polls_until_the_image_is_there(self):
        payload = base64.b64encode(b"png").decode()
        result, out, data = self._run([
            {"status": "queued"}, {"status": "generating"},
            {"status": "completed",
             "result": {"images": [{"index": 0, "b64_json": payload}]}},
        ])
        self.assertEqual(result, [out])
        self.assertEqual(data, b"png")

    def test_a_failed_job_is_an_engine_error_not_a_fallback(self):
        # Relancer en ligne de commande donnerait la même erreur, avec 80 s de
        # chargement en plus : ça ne se replie pas.
        with self.assertRaises(sdcpp.EngineError) as caught:
            self._run([{"status": "failed",
                        "error": {"message": "mémoire insuffisante"}}])
        self.assertNotIsInstance(caught.exception, sdserver.ServerUnavailable)
        self.assertIn("mémoire insuffisante", str(caught.exception))

    def test_a_cancelled_job_reads_as_a_user_interruption(self):
        with self.assertRaises(sdcpp.EngineError) as caught:
            self._run([{"status": "cancelled"}])
        self.assertIn("Interrupted", str(caught.exception))


class FallbackTests(unittest.TestCase):
    """Le serveur est un raccourci ; sd-cli reste le chemin qui doit marcher."""

    def _generate(self, prefs, server=None, serve=None):
        from atelier import registry
        used = {"cli": 0, "server": 0}
        model = registry.BaseModel(
            id="krea2-turbo", name="Krea 2", family="krea2", tags=[],
            description="", components=[
                registry.Component("diffusion", "repo", "m.gguf", None)],
            defaults={"sampler": "euler"}, vram_min_gb=12, presets=[])

        def cli_run(cmd, log=None, gpu_index=None, all_gpus=False):
            used["cli"] += 1

        def server_generate(*a, **k):
            used["server"] += 1
            if serve is not None:
                raise serve
            return [Path("image.png")]

        with tempfile.TemporaryDirectory() as tmp:
            diffusion = Path(tmp) / "m.gguf"
            diffusion.touch()
            with patch.object(generate.settings, "find_sd_cli",
                              return_value=Path("sd-cli")), \
                 patch.object(generate.registry, "get_base_model",
                              return_value=model), \
                 patch.object(generate, "_component", return_value=diffusion), \
                 patch.object(sdserver, "find_server", return_value=server), \
                 patch.object(sdserver, "generate", side_effect=server_generate), \
                 patch.object(sdserver, "stop"), \
                 patch.object(sdcpp, "supported_options",
                              return_value=frozenset()), \
                 patch.object(sdcpp, "build_gen_cmd", return_value=["sd-cli"]), \
                 patch.object(sdcpp, "run", side_effect=cli_run), \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate("krea2-turbo", "p", "", 4, 1.0, 512, 512,
                                  42, 1, prefs_override=prefs,
                                  save_prompt=False)
        return used

    def test_off_by_default(self):
        used = self._generate({}, server=Path("sd-server"))
        self.assertEqual(used, {"cli": 1, "server": 0})

    def test_used_when_asked_for_and_present(self):
        used = self._generate({"resident_engine": True},
                              server=Path("sd-server"))
        self.assertEqual(used, {"cli": 0, "server": 1})

    def test_asked_for_but_missing_binary_falls_back(self):
        used = self._generate({"resident_engine": True}, server=None)
        self.assertEqual(used, {"cli": 1, "server": 0})

    def test_an_unavailable_server_falls_back_to_the_command_line(self):
        used = self._generate({"resident_engine": True},
                              server=Path("sd-server"),
                              serve=sdserver.ServerUnavailable("mort"))
        self.assertEqual(used, {"cli": 1, "server": 1})


class MissingModuleTests(unittest.TestCase):
    """Une option absente retire l'option, elle n'empêche pas de démarrer.

    Vécu : `from . import sdcpp, sdserver` en tête de generate.py, et une mise à
    jour où sdserver.py manquait — l'application entière refusait de démarrer
    sur un ImportError, pour une fonctionnalité facultative et désactivée par
    défaut. Le test simule l'absence du fichier dans un interpréteur neuf.
    """

    def _import_with_sdserver_missing(self, modules: str) -> str:
        import subprocess
        import sys
        code = (
            "import importlib.abc, sys\n"
            "class Blocker(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'atelier.engine.sdserver':\n"
            "            raise ImportError('fichier absent (simulation)')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            f"import {modules}\n"
            "print('DEMARRE')\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True)
        return done.stdout + done.stderr

    def test_the_generation_engine_still_imports(self):
        self.assertIn("DEMARRE",
                      self._import_with_sdserver_missing("atelier.engine.generate"))

    def test_the_settings_tab_still_imports(self):
        self.assertIn("DEMARRE",
                      self._import_with_sdserver_missing("atelier.ui.settings_tab"))

    def test_the_application_itself_still_starts(self):
        # Le chemin exact du plantage rapporté : app.py -> convert_tab ->
        # generate -> sdserver.
        self.assertIn("DEMARRE", self._import_with_sdserver_missing("app"))

    def test_generation_falls_back_to_the_command_line(self):
        from atelier.engine import generate as gen
        with patch.object(gen, "resident_engine", return_value=None):
            self.assertIsNone(gen._resident_server({"resident_engine": True},
                                                   _req(), None))

    def test_cancelling_still_works(self):
        from atelier.engine import generate as gen
        with patch.object(gen, "resident_engine", return_value=None), \
             patch.object(sdcpp, "cancel_active", return_value="⏹️ Annulé."):
            self.assertEqual(gen.cancel(), "⏹️ Annulé.")

    def test_freeing_the_gpu_is_a_no_op(self):
        from atelier import engine
        with patch.object(engine, "resident_engine", return_value=None):
            engine.release_resident_engine("test")   # ne doit rien lever


class UnavailableReasonTests(unittest.TestCase):
    """Une option qu'on ne peut pas activer doit dire POURQUOI.

    Première version : la case disparaissait. On lisait sa description dans le
    dépôt puis on la cherchait en vain dans les Réglages, sans jamais savoir
    laquelle des deux pièces manquait.
    """

    def _reason(self, server=..., source=""):
        from atelier.ui import settings_tab
        with patch.object(settings_tab, "resident_engine",
                          return_value=server), \
             patch.object(settings_tab, "engine_build_source",
                          return_value=source):
            return settings_tab._resident_reason()

    def test_nothing_to_say_when_it_works(self):
        with patch.object(sdserver, "available", return_value=True):
            self.assertEqual(self._reason(server=sdserver), "")

    def test_a_missing_module_names_the_file_and_the_gesture(self):
        reason = self._reason(server=None)
        self.assertIn("sdserver.py", reason)
        self.assertIn("re-extract", reason)

    def test_a_missing_binary_from_our_own_build_names_the_workflow(self):
        with patch.object(sdserver, "available", return_value=False):
            reason = self._reason(server=sdserver, source="custom-ci")
        self.assertIn("in-house build", reason)
        self.assertIn("update-engine.bat", reason)

    def test_a_missing_binary_otherwise_points_at_the_updater(self):
        with patch.object(sdserver, "available", return_value=False):
            reason = self._reason(server=sdserver, source="official")
        self.assertIn("update-engine.bat", reason)
        self.assertNotIn("build maison", reason)


class VramHandoverTests(unittest.TestCase):
    """Un modèle résident occupe la carte : il doit céder la place tout seul."""

    def test_a_command_line_run_frees_the_resident_engine_first(self):
        order = []
        with patch.object(sdserver, "is_running", return_value=True), \
             patch.object(sdserver, "stop",
                          side_effect=lambda *a, **k: order.append("stop")), \
             patch("subprocess.Popen") as popen:
            popen.return_value.stdout = iter(())
            popen.return_value.wait.return_value = 0
            popen.return_value.poll.return_value = 0
            order.append("run")
            sdcpp.run(["sd-cli"])
        # `stop` doit être appelé PENDANT run, donc après le marqueur « run ».
        self.assertEqual(order, ["run", "stop"])

    def test_a_toolkit_tool_frees_it_too(self):
        from atelier.engine import tools
        stopped = []
        with patch.object(sdserver, "is_running", return_value=True), \
             patch.object(sdserver, "stop",
                          side_effect=lambda *a, **k: stopped.append(1)), \
             patch("subprocess.Popen") as popen:
            popen.return_value.stdout = iter(())
            popen.return_value.wait.return_value = 0
            tools._run_tool(["outil"], None, "échec")
        self.assertEqual(len(stopped), 1)


if __name__ == "__main__":
    unittest.main()
