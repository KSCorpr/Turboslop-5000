"""Service des fichiers image par le serveur web.

Le symptôme quand ça casse est une **icône d'image cassée** : le navigateur a
bien reçu une réponse, mais pas une image. Deux causes, toutes deux vérifiées
ici parce qu'aucune des deux ne se voit à la lecture du code.
"""
import os
import unittest
from pathlib import Path

from atelier import settings


class ServedPathsTests(unittest.TestCase):
    """Sans `allowed_paths`, Gradio répond 403 sur un chemin d'origine."""

    def test_covers_every_directory_handed_to_the_interface(self):
        served = {Path(p).resolve() for p in settings.served_paths()}
        # Ce sont les deux seuls dossiers dont l'application passe des chemins
        # à des composants : images finales, et aperçus/masques de travail.
        self.assertIn(settings.OUTPUT_DIR.resolve(), served)
        self.assertIn(settings.TMP_DIR.resolve(), served)

    def test_stays_narrow(self):
        """En partage réseau, cette liste est ce que la machine expose."""
        served = {Path(p).resolve() for p in settings.served_paths()}
        for forbidden in (settings.MODELS_DIR, settings.LORA_DIR,
                          settings.BIN_DIR, settings.ROOT):
            self.assertNotIn(forbidden.resolve(), served,
                             f"{forbidden} ne doit pas être exposé")

    def test_paths_are_strings(self):
        # Gradio attend des chaînes ; un Path passe silencieusement à côté sur
        # certaines versions.
        for p in settings.served_paths():
            self.assertIsInstance(p, str)


class GradioCacheLocationTests(unittest.TestCase):
    """Le cache de Gradio doit vivre DANS le projet.

    Dans %TEMP%, il est à la merci du nettoyage de disque de Windows : la copie
    servie au navigateur disparaît, la requête passe en 404, et l'image casse —
    par intermittence, ce qui rend le diagnostic pénible.
    """

    def test_app_pins_the_cache_inside_the_project(self):
        import ast
        src = (settings.ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        # `os.environ["GRADIO_TEMP_DIR"] = …` : une AFFECTATION, pas un
        # setdefault.
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 for tgt in n.targets
                 if isinstance(tgt, ast.Subscript)
                 and isinstance(tgt.slice, ast.Constant)
                 and tgt.slice.value == "GRADIO_TEMP_DIR"]
        self.assertTrue(found, "app.py ne fixe pas GRADIO_TEMP_DIR")

    def test_the_environment_cannot_silently_win(self):
        """Avec `setdefault`, une variable GRADIO_TEMP_DIR déjà posée dans
        l'environnement (autre application Gradio, ancienne installation)
        reprend la main sans bruit et remet le cache dans %TEMP% — soit
        exactement le bug que cette ligne existe pour empêcher."""
        import ast
        src = (settings.ROOT / "app.py").read_text(encoding="utf-8")
        setdefaults = [n for n in ast.walk(ast.parse(src))
                       if isinstance(n, ast.Call)
                       and getattr(n.func, "attr", "") == "setdefault"
                       and n.args and isinstance(n.args[0], ast.Constant)
                       and n.args[0].value == "GRADIO_TEMP_DIR"]
        self.assertEqual(setdefaults, [],
                         "GRADIO_TEMP_DIR doit être imposé, pas suggéré")

    def test_it_is_set_before_gradio_is_imported(self):
        """L'ordre est tout : la variable est lue au chargement du module."""
        src = (settings.ROOT / "app.py").read_text(encoding="utf-8")
        env_at = src.index("GRADIO_TEMP_DIR")
        import re
        m = re.search(r"^import gradio", src, re.MULTILINE)
        self.assertIsNotNone(m, "app.py n'importe pas gradio")
        self.assertLess(env_at, m.start(),
                        "GRADIO_TEMP_DIR doit être posé AVANT `import gradio`")

    def test_cache_is_under_tmp(self):
        want = (settings.ROOT / "tmp" / "gradio").resolve()
        src = (settings.ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"tmp", "gradio"', src,
                      f"le cache devrait être {want}")


if __name__ == "__main__":
    unittest.main()
