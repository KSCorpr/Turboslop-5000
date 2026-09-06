import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier.engine import tools


class SeedVr2CommandTests(unittest.TestCase):
    def test_secondary_gpu_is_mapped_as_cuda_one(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            fake_python = base / "python.exe"
            fake_python.touch()
            image = Image.new("RGB", (32, 24), "navy")

            def fake_run(cmd, log, err_msg, gpu_index=None, cwd=None, env=None):
                captured.update(cmd=cmd, cwd=cwd, env=env)
                output = Path(cmd[cmd.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 48), "navy").save(output)

            with (
                patch.object(tools, "seedvr2_is_installed", return_value=True),
                patch.object(tools, "_seedvr2_python", return_value=fake_python),
                patch.object(tools, "SEEDVR2_SOURCE_DIR", source),
                patch.object(tools, "SEEDVR2_MODEL_DIR", base / "models"),
                patch.object(tools.settings, "OUTPUT_DIR", base / "outputs"),
                patch.object(tools.settings, "load_prefs", return_value={
                    "encoder_gpu_index": 5, "text_gpu_index": 5,
                }),
                patch.object(tools.settings, "child_env", return_value=os.environ.copy()),
                patch.object(tools, "_gen_gpu_index", return_value=2),
                patch.object(tools, "_run_tool", side_effect=fake_run),
            ):
                output = tools.seedvr2_upscale(
                    image, resolution=2048, blocks_to_swap=16,
                    offload="secondary")

        self.assertTrue(output.name.startswith("seedvr2-"))
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "2,5")
        self.assertEqual(captured["cwd"], source)
        for option in ("--dit_offload_device", "--vae_offload_device",
                       "--tensor_offload_device"):
            self.assertEqual(captured["cmd"][captured["cmd"].index(option) + 1], "1")
        self.assertIn("--swap_io_components", captured["cmd"])

    def test_directory_batch_uses_one_process_and_model_caches(self):
        captured = []
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            fake_python = base / "python.exe"
            fake_python.touch()
            inputs = base / "originals"
            inputs.mkdir()
            for name, color in (("a.png", "navy"), ("b.png", "blue")):
                Image.new("RGB", (32, 24), color).save(inputs / name)

            def fake_run(cmd, log, err_msg, gpu_index=None, cwd=None, env=None):
                captured.append(cmd)
                out_dir = Path(cmd[cmd.index("--output") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 48), "navy").save(out_dir / "a.png")
                Image.new("RGB", (64, 48), "blue").save(out_dir / "b.png")

            with (
                patch.object(tools, "seedvr2_is_installed", return_value=True),
                patch.object(tools, "_seedvr2_python", return_value=fake_python),
                patch.object(tools, "SEEDVR2_SOURCE_DIR", source),
                patch.object(tools, "SEEDVR2_MODEL_DIR", base / "models"),
                patch.object(tools.settings, "TMP_DIR", base / "tmp"),
                patch.object(tools.settings, "OUTPUT_DIR", base / "outputs"),
                patch.object(tools.settings, "ensure_dirs", return_value=None),
                patch.object(tools.settings, "load_prefs", return_value={
                    "encoder_gpu_index": 5, "text_gpu_index": 5,
                }),
                patch.object(tools.settings, "child_env",
                             return_value=os.environ.copy()),
                patch.object(tools, "_gen_gpu_index", return_value=2),
                patch.object(tools, "_run_tool", side_effect=fake_run),
            ):
                outputs = tools.seedvr2_batch(
                    [inputs / "a.png", inputs / "b.png"], offload="secondary")

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(outputs), 2)
        self.assertIn("--cache_dit", captured[0])
        self.assertIn("--cache_vae", captured[0])
        self.assertTrue(Path(captured[0][2]).is_dir() or "seedvr2-batch" in captured[0][2])


class SeedVr2ModelTests(unittest.TestCase):
    """Le menu de l'interface EST la liste blanche : ils ne peuvent pas diverger."""

    def _run_with(self, model, blocks=36):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            fake_python = base / "python.exe"
            fake_python.touch()
            image = Image.new("RGB", (32, 24), "navy")

            def fake_run(cmd, log, err_msg, gpu_index=None, cwd=None, env=None):
                captured.update(cmd=cmd)
                output = Path(cmd[cmd.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 48), "navy").save(output)

            with (
                patch.object(tools, "seedvr2_is_installed", return_value=True),
                patch.object(tools, "_seedvr2_python", return_value=fake_python),
                patch.object(tools, "SEEDVR2_SOURCE_DIR", source),
                patch.object(tools, "SEEDVR2_MODEL_DIR", base / "models"),
                patch.object(tools.settings, "OUTPUT_DIR", base / "outputs"),
                patch.object(tools.settings, "load_prefs", return_value={}),
                patch.object(tools.settings, "child_env",
                             return_value=os.environ.copy()),
                patch.object(tools, "_gen_gpu_index", return_value=0),
                patch.object(tools, "_run_tool", side_effect=fake_run),
            ):
                tools.seedvr2_upscale(image, model=model, blocks_to_swap=blocks,
                                      offload="cpu")
        return captured["cmd"]

    def test_every_model_of_the_menu_is_accepted(self):
        for _, filename in tools.SEEDVR2_MODELS:
            cmd = self._run_with(filename)
            self.assertEqual(cmd[cmd.index("--dit_model") + 1], filename)

    def test_unknown_weights_are_refused(self):
        with self.assertRaises(tools.ToolError):
            self._run_with("../../etc/passwd.gguf")

    def test_seven_b_may_swap_its_36_blocks_the_three_b_only_32(self):
        cmd7 = self._run_with("seedvr2_ema_7b-Q4_K_M.gguf", blocks=36)
        cmd3 = self._run_with("seedvr2_ema_3b-Q8_0.gguf", blocks=36)
        self.assertEqual(cmd7[cmd7.index("--blocks_to_swap") + 1], "36")
        self.assertEqual(cmd3[cmd3.index("--blocks_to_swap") + 1], "32")


class SeedVr2AttentionTests(unittest.TestCase):
    """Flash attention seulement si la carte ET le paquet suivent."""

    def _mode(self, compute_cap, packages, arch="turing"):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            for name in packages:
                (site / name).mkdir()
            gpu = tools.hardware.Gpu(index=0, name="carte", vram_gb=12.0,
                                     arch=arch, tensor_cores=True,
                                     compute_cap=compute_cap)
            with (
                patch.object(tools, "_seedvr2_site_packages", return_value=site),
                patch.object(tools, "_gen_gpu_index", return_value=0),
                patch.object(tools.hardware, "detect_gpus", return_value=(gpu,)),
            ):
                return tools.seedvr2_attention_mode()

    def test_ampere_with_flash_attn_uses_it(self):
        self.assertEqual(self._mode("8.6", ["flash_attn"]), "flash_attn_2")

    def test_turing_stays_on_sdpa_even_with_the_package(self):
        self.assertEqual(self._mode("7.5", ["flash_attn"]), "sdpa")

    def test_pascal_stays_on_sdpa(self):
        self.assertEqual(self._mode("6.1", ["flash_attn", "sageattention"]),
                         "sdpa")

    def test_without_the_package_sdpa(self):
        self.assertEqual(self._mode("8.6", []), "sdpa")

    def test_sage_is_used_when_flash_is_absent(self):
        self.assertEqual(self._mode("8.6", ["sageattention"]), "sageattn_2")

    def test_old_driver_falls_back_on_the_architecture(self):
        # Pas de capacité rapportée : c'est l'architecture qui tranche.
        self.assertEqual(self._mode("", ["flash_attn"], arch="turing"), "sdpa")
        self.assertEqual(self._mode("", ["flash_attn"], arch="ampere"),
                         "flash_attn_2")


if __name__ == "__main__":
    unittest.main()
