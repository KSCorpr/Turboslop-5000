"""Pixel geometry, metadata, and retry regressions without a GPU."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from atelier.imaging.upscale import read_source, save_result, tiled_x4
from atelier.engine import generate, sdcpp


class PixelTests(unittest.TestCase):
    def test_overlap_has_no_gaps_or_seams_on_odd_rectangle(self):
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, (71, 103, 3), dtype=np.uint8)
        result = tiled_x4(Image.fromarray(pixels),
                         lambda a: a.repeat(4, 0).repeat(4, 1),
                         tile=32, overlap=8, halo=4)
        np.testing.assert_array_equal(np.asarray(result), pixels.repeat(4, 0).repeat(4, 1))

    def test_tiny_image(self):
        source = Image.new('RGB', (1, 3), (12, 41, 198))
        result = tiled_x4(source, lambda a: a.repeat(4, 0).repeat(4, 1))
        self.assertEqual(result.size, (4, 12))
        self.assertEqual(result.getpixel((0, 0)), (12, 41, 198))

    def test_wrong_model_and_nonfinite_are_rejected(self):
        source = Image.new('RGB', (4, 3))
        with self.assertRaises(ValueError):
            tiled_x4(source, lambda a: a)
        with self.assertRaises(ValueError):
            tiled_x4(source, lambda a: np.full((12, 16, 3), np.nan))

    def test_alpha_icc_and_exact_size(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Image.new('RGBA', (7, 5), (30, 60, 90, 100))
            source.info['icc_profile'] = b'test-profile'
            path = Path(folder) / 'out.png'
            save_result(Image.new('RGB', (28, 20)), source, path, (28, 20))
            with Image.open(path) as result:
                self.assertEqual(result.mode, 'RGBA')
                self.assertEqual(result.getchannel('A').getextrema(), (100, 100))
                self.assertEqual(result.info['icc_profile'], b'test-profile')
            with self.assertRaises(ValueError):
                save_result(Image.new('RGB', (14, 10)), source, path, (28, 20))
            with Image.open(path) as result:
                self.assertEqual(result.size, (28, 20))

    def test_orientation_and_palette_transparency(self):
        source = Image.new('RGB', (2, 3))
        source.getexif()[274] = 6
        self.assertEqual(read_source(source).size, (3, 2))
        with tempfile.TemporaryDirectory() as folder:
            source = Image.new('P', (2, 2))
            source.info['transparency'] = 0
            path = Path(folder) / 'out.png'
            save_result(Image.new('RGB', (8, 8)), source, path, (8, 8))
            with Image.open(path) as result:
                self.assertEqual(result.getchannel('A').getextrema(), (0, 0))


class NativeTests(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        def patched(obj, name, **kw):
            return self.stack.enter_context(patch.object(obj, name, **kw))
        patched(generate.settings, 'TMP_DIR', new=folder)
        patched(generate.settings, 'OUTPUT_DIR', new=folder)
        patched(generate.settings, 'ensure_dirs', return_value=None)
        patched(generate.settings, 'find_sd_cli', return_value=Path('sd-cli'))
        patched(generate.settings, 'load_prefs', return_value={})
        patched(generate.registry, 'upscaler_path', return_value=Path('4x.gguf'))
        patched(generate, '_resolved_flags', return_value=({}, 0))
        patched(generate.hardware, 'auto_profile', return_value=SimpleNamespace(gpu=None))
        patched(generate.hardware, 'free_vram_gb', return_value=4)
        import atelier.engine
        patched(atelier.engine, 'release_resident_engine', return_value=None)
        patched(sdcpp, 'supported_options', return_value={'--upscale-tile-size'})
        patched(sdcpp, 'was_cancelled', return_value=False)
        patched(sdcpp, 'build_upscale_cmd', side_effect=lambda cli, src, model, out, **kw:
                [str(src), str(out), str(kw['tile_size'])])
        self.run = patched(sdcpp, 'run')

    def produce(self, cmd, **kw):
        with Image.open(cmd[0]) as source:
            source.resize((source.width*4, source.height*4)).save(cmd[1])

    def test_native_preserves_alpha_and_cleans_temporary_files(self):
        self.run.side_effect = self.produce
        out = generate.upscale_image(Image.new('RGBA', (7, 5), (1, 2, 3, 42)), '4x.gguf')
        with Image.open(out) as result:
            self.assertEqual(result.size, (28, 20))
            self.assertEqual(result.getchannel('A').getextrema(), (42, 42))
        self.assertEqual(list(out.parent.iterdir()), [out])

    def test_generic_error_is_not_retried(self):
        self.run.side_effect = sdcpp.EngineError('invalid model')
        with self.assertRaises(sdcpp.EngineError):
            generate.upscale_image(Image.new('RGB', (256, 256)), '4x.gguf')
        self.assertEqual(self.run.call_count, 1)

    def test_only_oom_retries_and_reduces_tile(self):
        def run(cmd, **kw):
            if self.run.call_count == 1:
                raise sdcpp.VramError('out of memory')
            self.produce(cmd)
        self.run.side_effect = run
        generate.upscale_image(Image.new('RGB', (256, 256)), '4x.gguf')
        calls = self.run.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertLess(int(calls[1].args[0][2]), int(calls[0].args[0][2]))

    def test_wrong_native_scale_is_rejected(self):
        def bad(cmd, **kw):
            Image.new('RGB', (7, 5)).save(cmd[1])
        self.run.side_effect = bad
        with self.assertRaisesRegex(sdcpp.EngineError, 'expected'):
            generate.upscale_image(Image.new('RGB', (7, 5)), '4x.gguf')


class PortableRunnerTests(unittest.TestCase):
    def test_runner_imports_without_script_directory_on_sys_path(self):
        # -I removes cwd/PYTHONPATH, reproducing the embedded Python path issue.
        # run_path does not add the runner directory for a plain Python file.
        import subprocess
        import sys
        runner = Path(__file__).resolve().parents[1] / "scripts/tools/run_spandrel.py"
        code = (
            "import runpy, sys; from pathlib import Path; "
            "runner = Path(sys.argv[1]); "
            "assert str(runner.parent) not in sys.path; "
            "scope = runpy.run_path(str(runner)); "
            "assert Path(scope['pick_device'].__code__.co_filename).resolve() "
            "== runner.with_name('_device.py').resolve()"
        )
        with tempfile.TemporaryDirectory() as unrelated_cwd:
            result = subprocess.run([sys.executable, "-I", "-c", code, str(runner)],
                                    cwd=unrelated_cwd, capture_output=True, text=True,
                                    timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
