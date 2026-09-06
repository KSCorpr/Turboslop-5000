"""Choix du paquet trellis : quelle archive pour quelle carte.

Longtemps la réponse était « Vulkan, sauf RTX 30xx/50xx » : le CMakeLists amont
ÉCRASAIT la liste d'architectures de sa propre CI et ne compilait ses noyaux
que pour 86 et 120. v0.6.0 (19 août 2026) en a fait un simple défaut, et publie
deux archives CUDA — vérifié dans .github/workflows/release.yml au tag :

    cuda   (CUDA 13.1) -> 75;80;86;89;90;120   Turing et plus récent
    cuda12 (CUDA 12.9) -> 60;61;70             Pascal et Volta

Ces tests verrouillent la correspondance carte → archive, et surtout les deux
façons de se tromper : livrer « cuda » à une Pascal, ou « cuda12 » à une
Turing. Les deux se téléchargent, s'installent, et échouent au premier noyau.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import get_trellis  # noqa: E402


class BackendChoiceTests(unittest.TestCase):
    def test_turing_and_newer_get_the_cuda_package(self):
        for cap, card in (("7.5", "RTX 2080 Ti"), ("8.0", "A100"),
                          ("8.6", "RTX 3060"), ("8.9", "RTX 4070"),
                          ("9.0", "H100"), ("12.0", "RTX 50xx")):
            self.assertEqual(get_trellis.preferred_backend(cap), "cuda", card)

    def test_pascal_and_volta_get_the_legacy_package(self):
        for cap, card in (("6.0", "Tesla P100"), ("6.1", "GTX 1080 Ti"),
                          ("7.0", "Tesla V100")):
            self.assertEqual(get_trellis.preferred_backend(cap), "cuda12", card)

    def test_anything_else_gets_vulkan(self):
        # Sans information, on ne parie pas : Vulkan ne compile rien par
        # architecture, il marche partout.
        for unknown in ("", None, "   ", "bogus", "5.2", "10.0"):
            self.assertEqual(get_trellis.preferred_backend(unknown), "vulkan")


class AssetPickTests(unittest.TestCase):
    # Les noms réels de la release v0.6.0.
    ASSETS = [
        {"name": "trellis-cuda-linux-x64.tar.gz"},
        {"name": "trellis-cuda-windows-x64.zip"},
        {"name": "trellis-cuda12-linux-x64.tar.gz"},
        {"name": "trellis-cuda12-windows-x64.zip"},
        {"name": "trellis-rocm-windows-x64.zip"},
        {"name": "trellis-studio-windows-x64-portable.zip"},
        {"name": "trellis-vulkan-windows-x64.zip"},
        {"name": "trellis-vulkan-linux-x64.tar.gz"},
    ]

    def _pick(self, backend, assets=None):
        return get_trellis._pick_asset(assets or self.ASSETS, backend)["name"]

    def test_cuda_is_not_confused_with_cuda12(self):
        # « cuda » est un préfixe de « cuda12 » : un simple `in` donnerait
        # l'archive Pascal à une carte Turing.
        self.assertEqual(self._pick("cuda"), "trellis-cuda-windows-x64.zip")
        self.assertEqual(self._pick("cuda12"), "trellis-cuda12-windows-x64.zip")

    def test_it_never_picks_rocm_the_studio_app_or_linux(self):
        for backend in ("cuda", "cuda12", "vulkan"):
            name = self._pick(backend)
            self.assertNotIn("rocm", name)
            self.assertNotIn("studio", name)
            self.assertIn("windows", name)

    def test_the_only_fallback_is_vulkan(self):
        # Un job de CI peut échouer. Mais remplacer « cuda » par « cuda12 »
        # (ou l'inverse) livrerait un binaire compilé pour d'autres
        # architectures que celles de la carte.
        without_cuda = [a for a in self.ASSETS if "cuda-" not in a["name"]]
        self.assertEqual(get_trellis._pick_asset(without_cuda, "cuda")["name"],
                         "trellis-vulkan-windows-x64.zip")
        without_legacy = [a for a in self.ASSETS if "cuda12" not in a["name"]]
        self.assertEqual(get_trellis._pick_asset(without_legacy, "cuda12")["name"],
                         "trellis-vulkan-windows-x64.zip")

    def test_no_windows_asset_at_all(self):
        self.assertIsNone(get_trellis._pick_asset(
            [{"name": "trellis-vulkan-linux-x64.tar.gz"}], "vulkan"))

    def test_the_studio_app_is_never_mistaken_for_the_engine(self):
        only_studio = [{"name": "trellis-studio-windows-x64-portable.zip"}]
        self.assertIsNone(get_trellis._pick_asset(only_studio, "cuda"))


class DiagnosisTests(unittest.TestCase):
    def test_kernel_image_error_stays_actionable(self):
        """L'ancien message conseillait une mise à jour — qui ne pouvait rien.

        Depuis v0.6.0 elle le peut, justement : c'est le conseil à donner.
        """
        from atelier.engine import trellis
        trellis._CRASH.clear()
        trellis._CRASH.append(
            "CUDA error: no kernel image is available for execution on the device")
        msg = trellis._diagnose_crash(None)
        self.assertTrue("update-trellis" in msg.lower()
                        or "vulkan" in msg.lower(),
                        "le message n'indique aucune action")


if __name__ == "__main__":
    unittest.main()
