"""Ce que le passage à Gradio 6 impose, et qu'un oubli casserait en silence.

Deux pièges, tous deux invisibles à la lecture :

1. **Le bouton « share ».** Gradio 6 a remplacé les `show_*_button` par une
   liste, et son défaut pour une image est `["download", "share",
   "fullscreen"]`. Le bouton « share » partage vers les Discussions Hugging
   Face Spaces — sans objet dans une application locale, où il n'apparaissait
   d'ailleurs pas sous Gradio 5. Un composant ajouté sans `buttons=` le
   reprendrait sans que rien ne le signale.

2. **Thème, CSS et `<head>`.** Ils ont quitté le constructeur `Blocks(...)`
   pour `launch(...)`. Oubliés, l'interface se construit et s'affiche — sans
   son thème. Rien n'échoue, personne n'est prévenu.
"""
import ast
import unittest
from pathlib import Path

import gradio as gr

import app
from atelier.ui import widgets

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "atelier" / "ui"


class ButtonListsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = app.build_app()
        cls.visuals = [c for c in cls.demo.blocks.values()
                       if isinstance(c, (gr.Image, gr.Gallery))]

    def test_there_are_visual_components_to_check(self):
        self.assertGreater(len(self.visuals), 20)

    def test_no_component_offers_to_share_to_hugging_face(self):
        guilty = [getattr(c, "label", "?") for c in self.visuals
                  if "share" in (getattr(c, "buttons", None) or [])]
        self.assertEqual(guilty, [], f"bouton « share » sur : {guilty}")

    def test_every_image_declares_its_buttons_explicitly(self):
        """Sans `buttons=`, c'est le défaut de Gradio qui décide — « share »
        compris. Le contrôle se fait sur la SOURCE : un composant construit
        avec le défaut est indiscernable une fois instancié."""
        bare = []
        for path in sorted(UI.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("Image", "Gallery")
                        and not any(k.arg == "buttons" for k in n.keywords)):
                    bare.append(f"{path.name}:{n.lineno}")
        self.assertEqual(bare, [], "sans buttons= : " + ", ".join(bare))

    def test_download_stays_available_where_there_is_something_to_keep(self):
        with_dl = [c for c in self.visuals
                   if "download" in (getattr(c, "buttons", None) or [])]
        self.assertGreater(len(with_dl), 5,
                           "plus rien ne se télécharge : les sorties ont perdu "
                           "leur bouton")

    def test_the_named_lists_are_the_ones_used(self):
        used = {tuple(getattr(c, "buttons", None) or ()) for c in self.visuals}
        known = {tuple(widgets.IMAGE_BUTTONS), tuple(widgets.IMAGE_VIEW_ONLY),
                 tuple(widgets.GALLERY_BUTTONS)}
        self.assertTrue(used <= known, f"jeux de boutons hors liste : {used - known}")


class PresentationMovedToLaunchTests(unittest.TestCase):
    def test_presentation_carries_the_three_parameters(self):
        p = app.presentation()
        self.assertEqual(set(p), {"theme", "css", "head"})
        self.assertTrue(p["css"])
        self.assertTrue(p["head"])

    def test_blocks_no_longer_receives_them(self):
        """Gradio 6 les refuse ; sous 5.x ils déclenchaient un avertissement."""
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "Blocks"):
                names = {k.arg for k in n.keywords}
                self.assertEqual(names & {"theme", "css", "head"}, set(),
                                 f"Blocks() reçoit encore {names}")

    def test_launch_receives_them(self):
        """Le piège du déménagement : sans ça l'interface s'affiche nue."""
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        launches = [n for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "launch"]
        self.assertTrue(launches, "app.py n'appelle plus launch()")
        starred = [n for n in launches
                   if any(isinstance(k.value, ast.Call)
                          and getattr(k.value.func, "id", "") == "presentation"
                          for k in n.keywords if k.arg is None)]
        self.assertTrue(starred, "launch() ne reçoit pas **presentation()")

    def test_show_api_is_gone(self):
        """Retiré de launch() en 6.0 : le laisser lèverait un TypeError au
        démarrage, c'est-à-dire après l'installation, chez l'utilisateur."""
        import inspect
        self.assertNotIn("show_api",
                         inspect.signature(gr.Blocks.launch).parameters)
        # Sur l'AST, pas sur le texte : le commentaire qui explique le retrait
        # contient le mot, et ce n'est pas un appel.
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        passed = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and any(k.arg == "show_api" for k in n.keywords)]
        self.assertEqual(passed, [], "show_api est encore passé à un appel")


class DependencyPinTests(unittest.TestCase):
    def test_requirements_demand_gradio_6(self):
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("gradio>=6.0,<7", req)

    def test_the_installed_version_matches(self):
        major = int(gr.__version__.split(".")[0])
        self.assertEqual(major, 6, f"gradio {gr.__version__} installé")

    def test_maintenance_checks_the_major(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "maintenance", ROOT / "scripts" / "maintenance.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.GRADIO_MAJOR, 6)
        self.assertTrue(callable(mod.check_gradio_major))


if __name__ == "__main__":
    unittest.main()
