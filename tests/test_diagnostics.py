import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atelier import diagnostics, settings


class ShareableReportTests(unittest.TestCase):
    def test_engine_path_does_not_export_the_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "bin" / "sd-cli"
            cli.parent.mkdir()
            cli.write_bytes(b"engine")
            with patch.object(settings, "ROOT", root), \
                 patch.object(settings, "find_sd_cli", return_value=cli), \
                 patch.object(diagnostics, "read_engine_manifest", return_value={}), \
                 patch.object(diagnostics.sdcpp, "supported_options",
                              return_value=frozenset({"--mode"})):
                report = diagnostics.engine_report()
        self.assertEqual(report["path"], str(Path("bin") / "sd-cli"))
        self.assertNotIn(tmp, report["path"])

    def test_system_report_never_exports_download_credentials(self):
        prefs = {
            "gpu_index": 0,
            "civitai_token": "TOP-SECRET",
            "hf_endpoint": "https://private.example/token",
        }
        with patch.object(settings, "load_prefs", return_value=prefs), \
             patch.object(diagnostics.hardware, "detect_gpus", return_value=()), \
             patch.object(diagnostics.hardware, "used_vram_gb", return_value={}), \
             patch.object(diagnostics.hardware, "detect_ram_gb", return_value=64), \
             patch.object(diagnostics.hardware, "cpu_name", return_value="CPU"), \
             patch.object(diagnostics, "engine_report", return_value={}):
            report = diagnostics.system_report()
        exported = str(report)
        self.assertNotIn("TOP-SECRET", exported)
        self.assertNotIn("private.example", exported)


if __name__ == "__main__":
    unittest.main()
