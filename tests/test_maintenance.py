"""La maintenance ne doit pas supprimer du code vivant.

Ce test existe à cause d'un incident précis : `atelier/engine/sdserver.py`
était déclaré dans REMOVED_FEATURES (ancien backend serveur, retiré), un
nouveau module du même nom est arrivé des mois plus tard, et chaque passage de
la maintenance l'effaçait. L'application ne démarrait plus, avec un ImportError
que rien ne rattachait à la maintenance.

Une table de noms de fichiers ne peut pas savoir qu'un nom a été repris. Le
code actuel, lui, le sait.
"""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "maintenance.py"
_SPEC = importlib.util.spec_from_file_location("maintenance_test", _PATH)
M = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(M)


def _fake_project(root: Path, engine_init: str) -> None:
    """Un projet minuscule mais de la même FORME que le vrai."""
    (root / "atelier" / "engine").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "app.py").write_text("from atelier import engine\n", encoding="utf-8")
    (root / "atelier" / "__init__.py").write_text("", encoding="utf-8")
    (root / "atelier" / "engine" / "__init__.py").write_text(
        engine_init, encoding="utf-8")
    (root / "atelier" / "engine" / "sdserver.py").write_text(
        "SERVER = 1\n", encoding="utf-8")


class RemovedFeatureGuardTests(unittest.TestCase):
    def setUp(self):
        M._REACHABLE = None
        self._problems = M._problems

    def tearDown(self):
        M._REACHABLE = None
        M._problems = self._problems

    def _clean(self, root, engine_init):
        _fake_project(root, engine_init)
        table = [{"name": "Ancien backend serveur",
                  "files": ["atelier/engine/sdserver.py"], "dirs": []}]
        with patch.object(M, "ROOT", root), \
             patch.object(M, "REMOVED_FEATURES", table):
            M.clean_removed_features(purge=False)
        return (root / "atelier" / "engine" / "sdserver.py").exists()

    def test_a_module_the_code_imports_survives_the_table(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # L'import est PARESSEUX, à l'intérieur d'une fonction : c'est
            # exactement la forme qu'a le vrai code, et celle qu'une analyse
            # trop naïve rate.
            init = ("def resident_engine():\n"
                    "    from . import sdserver\n"
                    "    return sdserver\n")
            self.assertTrue(self._clean(Path(tmp), init),
                            "un module importé a été supprimé")

    def test_a_module_nobody_imports_is_still_removed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self._clean(Path(tmp), "# rien\n"),
                             "le nettoyage ne fait plus son travail")


class RelativeImportTests(unittest.TestCase):
    """« from . import x » dans un __init__.py désigne le paquet lui-même."""

    def setUp(self):
        M._REACHABLE = None

    def tearDown(self):
        M._REACHABLE = None

    def test_a_package_init_resolves_its_own_package(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_project(root, "def f():\n    from . import sdserver\n")
            with patch.object(M, "ROOT", root):
                found = M._imports_of(root / "atelier" / "engine" / "__init__.py")
        self.assertIn("atelier.engine.sdserver", found)
        # Le paquet du dessus n'a rien à voir : c'était le bug.
        self.assertNotIn("atelier.sdserver", found)

    def test_a_plain_module_resolves_its_parent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_project(root, "")
            (root / "atelier" / "engine" / "generate.py").write_text(
                "from . import sdcpp\n", encoding="utf-8")
            with patch.object(M, "ROOT", root):
                found = M._imports_of(root / "atelier" / "engine" / "generate.py")
        self.assertIn("atelier.engine.sdcpp", found)


class RealProjectTests(unittest.TestCase):
    """Sur le vrai dépôt, aucune entrée de la table ne vise du code vivant."""

    def setUp(self):
        M._REACHABLE = None

    def tearDown(self):
        M._REACHABLE = None

    def test_no_declared_removal_targets_a_living_module(self):
        living = [rel for feat in M.REMOVED_FEATURES for rel in feat["files"]
                  if M._still_in_service(rel)]
        self.assertEqual(living, [], "REMOVED_FEATURES vise du code vivant")

    def test_the_resident_engine_is_reachable(self):
        # Il n'est importé que paresseusement, depuis un __init__.py : c'est
        # le cas limite qui a tout déclenché.
        self.assertIn("atelier.engine.sdserver", M._reachable_modules())


if __name__ == "__main__":
    unittest.main()
