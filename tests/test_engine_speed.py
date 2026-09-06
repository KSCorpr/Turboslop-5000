"""Ce que le journal de sd-cli dit du matériel, et ce qu'on en fait.

Les deux corrections testées ici viennent d'un log réel : 250 s pour une image
Krea 2, dont 80 s à lire le modèle (105 Mo/s, une vitesse de disque mécanique)
et 38 s à encoder le prompt sur une GTX 1080 Ti. Aucun des deux chiffres n'est
un problème de moteur de diffusion, et aucun des deux ne se voyait sans faire
une division de tête.
"""
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import hardware
from atelier.engine import generate, sdcpp


def _gpu(index, name, arch, tensor_cores, vram=12.0):
    return hardware.Gpu(index=index, name=name, vram_gb=vram, arch=arch,
                        tensor_cores=tensor_cores)


# Extrait VERBATIM d'un journal de génération (Krea 2 Turbo, Q5_K_M).
REAL_LOG = [
    "[INFO ] model_loader.cpp:983  - loading 430/432 tensors from "
    "D:\\AI\\models\\krea2_turbo-Q5_K_M.gguf",
    "[DEBUG] model_loader.cpp:1243 - loading tensors completed, taking 80.54s "
    "(read: 79.86s, memcpy: 0.00s, convert: 0.00s, copy_to_backend: 0.31s)",
    "[INFO ] model_manager.cpp:409  - model manager prepared params backend "
    "buffer (8410.71 MB, 430 tensors, VRAM)",
]
# La VAE, juste après : petite et vite lue. Rien à signaler.
VAE_LOG = [
    "[DEBUG] model_loader.cpp:1243 - loading tensors completed, taking 1.37s "
    "(read: 1.22s, memcpy: 0.00s, convert: 0.02s, copy_to_backend: 0.00s)",
    "[INFO ] model_manager.cpp:409  - model manager prepared params backend "
    "buffer (136.46 MB, 104 tensors, VRAM)",
]


class DiskWatchTests(unittest.TestCase):
    def _notes(self, lines):
        watch = sdcpp.DiskWatch()
        return [n for n in (watch.note(line) for line in lines) if n]

    def test_reports_the_measured_speed_of_a_slow_disk(self):
        notes = self._notes(REAL_LOG)
        self.assertEqual(len(notes), 1)
        # 8410.71 Mo / 79.86 s = 105 Mo/s.
        self.assertIn("105 MB/s", notes[0])
        self.assertIn("8.2 GB", notes[0])

    def test_says_how_much_an_ssd_would_save(self):
        note = self._notes(REAL_LOG)[0]
        # 8410.71 / 550 ≈ 15 s, donc ~65 s économisées.
        self.assertIn("~15 s", note)
        self.assertIn("~65 s", note)

    def test_stays_quiet_on_a_fast_disk(self):
        fast = [REAL_LOG[0],
                REAL_LOG[1].replace("read: 79.86s", "read: 6.86s"),
                REAL_LOG[2]]
        self.assertEqual(self._notes(fast), [])

    def test_stays_quiet_on_small_files(self):
        # 136 Mo en 1,22 s = 111 Mo/s : lent sur le papier, mais une seconde de
        # lecture ne prouve rien et ne vaut pas un avertissement.
        self.assertEqual(self._notes(VAE_LOG), [])

    def test_says_it_once_not_for_every_model_of_the_run(self):
        self.assertEqual(len(self._notes(REAL_LOG + REAL_LOG)), 1)

    def test_ignores_a_size_that_arrives_without_a_read_time(self):
        self.assertEqual(self._notes([REAL_LOG[2]]), [])


class EncoderPlacementTests(unittest.TestCase):
    """Le fp16 d'une Pascal tourne à 1/64 : elle stocke, elle ne calcule pas."""

    def _with(self, gpus):
        return patch.object(hardware, "detect_gpus", return_value=tuple(gpus))

    def test_pascal_second_card_is_refused_for_the_encoder(self):
        gpus = [_gpu(0, "NVIDIA GeForce RTX 3060", "ampere", True),
                _gpu(1, "NVIDIA GeForce GTX 1080 Ti", "pascal", False, 11.0)]
        with self._with(gpus):
            self.assertTrue(generate.encoder_gpu_too_slow(1, 0))

    def test_two_capable_cards_keep_the_split(self):
        gpus = [_gpu(0, "RTX 3060", "ampere", True),
                _gpu(1, "RTX 2080 Ti", "turing", True, 11.0)]
        with self._with(gpus):
            self.assertFalse(generate.encoder_gpu_too_slow(1, 0))

    def test_a_pascal_main_card_does_not_refuse_its_own_kind(self):
        # Si la carte de génération n'a pas de tensor cores non plus, déplacer
        # l'encodeur ne gagne rien : on ne touche à rien.
        gpus = [_gpu(0, "GTX 1080 Ti", "pascal", False, 11.0),
                _gpu(1, "GTX 1070", "pascal", False, 8.0)]
        with self._with(gpus):
            self.assertFalse(generate.encoder_gpu_too_slow(1, 0))

    def test_same_card_or_unknown_index_is_never_a_problem(self):
        gpus = [_gpu(0, "RTX 3060", "ampere", True)]
        with self._with(gpus):
            self.assertFalse(generate.encoder_gpu_too_slow(0, 0))
            self.assertFalse(generate.encoder_gpu_too_slow(None, 0))
            self.assertFalse(generate.encoder_gpu_too_slow(7, 0))


class EncoderResidencyTests(unittest.TestCase):
    """Sortir l'encodeur du split ne suffit pas : la résidence l'y renverrait.

    `params_backend` est enregistré dans les préférences de l'utilisateur. Tant
    qu'il contient « te=cuda<Pascal> », changer l'index du GPU d'encodeur ne
    déplace rien du tout — c'est la chaîne de résidence qui décide.
    """

    def _request(self, prefs, gpus):
        import tempfile
        from atelier import registry
        captured = {}
        model = registry.BaseModel(
            id="krea2-turbo", name="Krea 2", family="krea2", tags=[],
            description="", components=[
                registry.Component("diffusion", "repo", "m.gguf", None)],
            defaults={"sampler": "euler"}, vram_min_gb=12, presets=[])
        with tempfile.TemporaryDirectory() as tmp:
            diffusion = Path(tmp) / "m.gguf"
            diffusion.touch()

            def build(_cli, req, _out):
                captured["request"] = req
                return ["sd-cli"]

            with patch.object(generate.settings, "find_sd_cli",
                              return_value=Path("sd-cli")), \
                 patch.object(generate.registry, "get_base_model",
                              return_value=model), \
                 patch.object(generate, "_component", return_value=diffusion), \
                 patch.object(hardware, "detect_gpus", return_value=tuple(gpus)), \
                 patch.object(sdcpp, "supported_options",
                              return_value=frozenset({"--params-backend"})), \
                 patch.object(sdcpp, "build_gen_cmd", side_effect=build), \
                 patch.object(sdcpp, "run"), \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate("krea2-turbo", "p", "", 4, 1.0, 512, 512,
                                  42, 1, prefs_override=prefs,
                                  save_prompt=False)
        return captured["request"]

    DUO = [_gpu(0, "NVIDIA GeForce RTX 3060", "ampere", True),
           _gpu(1, "NVIDIA GeForce GTX 1080 Ti", "pascal", False, 11.0)]

    def _prefs(self, **extra):
        return {"auto_optimize": False, "gpu_index": 0,
                "encoder_gpu_index": 1, "auto_fit": False,
                "params_backend": "diffusion=cuda0,vae=cuda0,te=cuda1",
                "flags": {"diffusion_fa": True}, **extra}

    def test_saved_pascal_residency_is_rewritten(self):
        req = self._request(self._prefs(), self.DUO)
        self.assertEqual(req.params_backend, "diffusion=cuda0,vae=cuda0,te=cpu")
        self.assertIsNone(req.encoder_gpu_index)

    def test_a_capable_second_card_keeps_its_encoder(self):
        duo = [_gpu(0, "RTX 3060", "ampere", True),
               _gpu(1, "RTX 2080 Ti", "turing", True, 11.0)]
        req = self._request(self._prefs(), duo)
        self.assertEqual(req.params_backend,
                         "diffusion=cuda0,vae=cuda0,te=cuda1")
        self.assertEqual(req.encoder_gpu_index, 1)

    def test_the_benchmark_may_still_measure_the_slow_placement(self):
        req = self._request(self._prefs(encoder_placement_forced=True),
                            self.DUO)
        self.assertEqual(req.params_backend,
                         "diffusion=cuda0,vae=cuda0,te=cuda1")

    def test_without_a_saved_residency_the_encoder_still_leaves_the_pascal(self):
        prefs = self._prefs()
        prefs.pop("params_backend")
        req = self._request(prefs, self.DUO)
        self.assertEqual(req.params_backend, "diffusion=cuda0,vae=cuda0,te=cpu")


class VramRetryTests(unittest.TestCase):
    """Ramener l'encodeur sur la carte principale peut faire déborder 12 Go."""

    def _generate(self, failures):
        import tempfile
        from atelier import registry
        calls = []
        model = registry.BaseModel(
            id="krea2-turbo", name="Krea 2", family="krea2", tags=[],
            description="", components=[
                registry.Component("diffusion", "repo", "m.gguf", None)],
            defaults={"sampler": "euler"}, vram_min_gb=12, presets=[])

        def run(cmd, log=None, gpu_index=None, all_gpus=False):
            calls.append(cmd)
            if len(calls) <= failures:
                raise sdcpp.VramError("plus de VRAM")

        with tempfile.TemporaryDirectory() as tmp:
            diffusion = Path(tmp) / "m.gguf"
            diffusion.touch()
            with patch.object(generate.settings, "find_sd_cli",
                              return_value=Path("sd-cli")), \
                 patch.object(generate.registry, "get_base_model",
                              return_value=model), \
                 patch.object(generate, "_component", return_value=diffusion), \
                 patch.object(hardware, "detect_gpus", return_value=()), \
                 patch.object(sdcpp, "supported_options",
                              return_value=frozenset()), \
                 patch.object(sdcpp, "build_gen_cmd",
                              side_effect=lambda c, req, o: list(
                                  ["--clip-on-cpu"] if req.flags.get("clip_on_cpu")
                                  else [])), \
                 patch.object(sdcpp, "run", side_effect=run), \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate("krea2-turbo", "p", "", 4, 1.0, 512, 512,
                                  42, 1, save_prompt=False,
                                  prefs_override={"auto_optimize": False,
                                                  "gpu_index": 0,
                                                  "flags": {}})
        return calls

    def test_an_oom_is_retried_once_with_the_encoder_in_ram(self):
        calls = self._generate(failures=1)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--clip-on-cpu", calls[0])
        self.assertIn("--clip-on-cpu", calls[1])

    def test_a_successful_run_is_never_retried(self):
        self.assertEqual(len(self._generate(failures=0)), 1)

    def test_the_retry_is_not_repeated_forever(self):
        with self.assertRaises(sdcpp.VramError):
            self._generate(failures=2)


class PresetTests(unittest.TestCase):
    def test_the_dual_gpu_preset_no_longer_encodes_on_the_pascal(self):
        gpus = (_gpu(0, "NVIDIA GeForce RTX 3060", "ampere", True),
                _gpu(1, "NVIDIA GeForce GTX 1080 Ti", "pascal", False, 11.0))
        with patch.object(hardware, "detect_gpus", return_value=gpus):
            prefs = hardware.rtx3060_1080ti_prefs()
        self.assertEqual(prefs["gpu_index"], 0)
        self.assertEqual(prefs["encoder_gpu_index"], 0)
        # Le LLM de prompt, lui, reste sur la 1080 Ti : il tourne seul.
        self.assertEqual(prefs["text_gpu_index"], 1)


if __name__ == "__main__":
    unittest.main()
