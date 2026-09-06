import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import benchmark, hardware, registry
from atelier.engine import generate, sdcpp


class ParamsBackendTests(unittest.TestCase):
    def _cmd(self, req, options):
        with patch.object(sdcpp, "_require", lambda *a, **k: None), \
             patch.object(sdcpp, "supported_options",
                          return_value=frozenset(options)):
            return sdcpp.build_gen_cmd(Path("sd-cli"), req, Path("out.png"))

    def test_explicit_residency_wins_over_legacy_offload(self):
        mapping = "diffusion=cuda0,vae=cuda0,te=cuda1"
        req = sdcpp.GenRequest(
            diffusion_model=Path("model.gguf"), params_backend=mapping,
            flags={"offload_to_cpu": True, "clip_on_cpu": True})
        cmd = self._cmd(req, {"--params-backend"})
        self.assertEqual(cmd[cmd.index("--params-backend") + 1], mapping)
        self.assertNotIn("--offload-to-cpu", cmd)
        # Parameter residency must not erase an explicit CPU-compute fallback.
        self.assertIn("--clip-on-cpu", cmd)

    def test_old_engine_keeps_the_compatible_offload(self):
        req = sdcpp.GenRequest(
            diffusion_model=Path("model.gguf"),
            params_backend="diffusion=cuda0,vae=cuda0,te=cuda1",
            flags={"offload_to_cpu": True})
        cmd = self._cmd(req, set())
        self.assertNotIn("--params-backend", cmd)
        self.assertIn("--offload-to-cpu", cmd)


class BenchmarkPlanTests(unittest.TestCase):
    def test_exact_combo_compares_resident_and_staged_encoder(self):
        gpus = (
            hardware.Gpu(0, "NVIDIA GeForce RTX 3060", 12.0, "ampere", True),
            hardware.Gpu(1, "NVIDIA GeForce GTX 1080 Ti", 11.0, "pascal", False),
        )
        with patch.object(hardware, "auto_profile") as auto:
            auto.return_value.flags.return_value = {
                "diffusion_fa": True, "offload_to_cpu": True,
                "vae_tiling": True, "clip_on_cpu": False,
                "vae_on_cpu": False,
            }
            modes = benchmark.placement_candidates({"gpu_index": 0}, gpus)
        self.assertEqual([m.key for m in modes],
                         ["single-staged", "dual-resident", "dual-staged"])
        resident = modes[1].prefs_patch
        self.assertEqual(resident["params_backend"],
                         "diffusion=cuda0,vae=cuda0,te=cuda1")
        self.assertFalse(resident["flags"]["offload_to_cpu"])
        self.assertEqual(modes[2].prefs_patch["params_backend"], "*=cpu")


class Int8PlacementTests(unittest.TestCase):
    def _run(self, gpus):
        """Le placement INT8 obtenu avec ces cartes-là.

        Les cartes sont FOURNIES : depuis que la génération refuse de faire
        calculer l'encodeur sur une puce sans tensor cores, ce test dépendrait
        sinon du matériel de la machine qui l'exécute.
        """
        captured = {}
        model = registry.BaseModel(
            id="krea2-turbo-int8", name="INT8", family="krea2", tags=[],
            description="", components=[
                registry.Component("diffusion", "repo", "model.safetensors", None)
            ], defaults={"memory_preset": "int8_stream", "sampler": "euler"},
            vram_min_gb=12, presets=[])
        prefs = {
            "auto_optimize": False, "gpu_index": 0, "encoder_gpu_index": 1,
            "auto_fit": True, "flags": {"diffusion_fa": True,
                                          "offload_to_cpu": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            diffusion = Path(tmp) / "model.safetensors"
            diffusion.touch()

            def build(_cli, req, _out):
                captured["request"] = req
                return ["sd-cli"]

            with patch.object(generate.settings, "find_sd_cli",
                              return_value=Path("sd-cli")), \
                 patch.object(generate.registry, "get_base_model",
                              return_value=model), \
                 patch.object(generate, "_component", return_value=diffusion), \
                 patch.object(hardware, "detect_gpus", return_value=gpus), \
                 patch.object(sdcpp, "supported_options",
                              return_value=frozenset({"--params-backend"})), \
                 patch.object(sdcpp, "build_gen_cmd", side_effect=build), \
                 patch.object(sdcpp, "run"), \
                 patch.object(sdcpp, "collect_outputs", return_value=[]):
                generate.generate(
                    "krea2-turbo-int8", "prompt", "", 4, 1.0, 512, 512,
                    42, 1, prefs_override=prefs, save_prompt=False)
        return captured["request"]

    def test_int8_streaming_disables_conflicting_auto_fit(self):
        req = self._run((
            hardware.Gpu(0, "RTX 3060", 12.0, "ampere", True),
            hardware.Gpu(1, "RTX 2080 Ti", 11.0, "turing", True),
        ))
        self.assertFalse(req.auto_fit)
        self.assertEqual(req.params_backend,
                         "diffusion=cpu,vae=cuda0,te=cuda1")
        self.assertTrue(req.stream_layers)

    def test_int8_streaming_keeps_the_encoder_off_a_pascal_card(self):
        req = self._run((
            hardware.Gpu(0, "RTX 3060", 12.0, "ampere", True),
            hardware.Gpu(1, "GTX 1080 Ti", 11.0, "pascal", False),
        ))
        # Le streaming INT8 garde ses poids de diffusion en RAM ; seul le
        # placement de l'encodeur change.
        self.assertEqual(req.params_backend, "diffusion=cpu,vae=cuda0,te=cpu")
        self.assertIsNone(req.encoder_gpu_index)


class PcieDetectionTests(unittest.TestCase):
    def tearDown(self):
        hardware.detect_gpus.cache_clear()
        hardware._apple_gpu.cache_clear()

    def test_link_width_is_reported_without_breaking_old_driver_fallback(self):
        responses = {
            "index,name,memory.total,compute_cap,driver_version":
                "0, NVIDIA GeForce RTX 3060, 12288, 8.6, 610.0\n",
            "index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current":
                "0, 00000000:01:00.0, 3, 16\n",
        }
        hardware.detect_gpus.cache_clear()
        with patch.object(hardware, "_apple_gpu", return_value=None), \
             patch.object(hardware, "_nvidia_smi",
                          side_effect=lambda fields: responses.get(fields)):
            gpu = hardware.detect_gpus()[0]
        self.assertEqual(gpu.pcie_label, "Gen3 x16")
        self.assertEqual(gpu.bus_id, "00000000:01:00.0")


if __name__ == "__main__":
    unittest.main()
