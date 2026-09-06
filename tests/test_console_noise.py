"""La console doit rester lisible — sans devenir muette.

Deux exigences opposées, et c'est tout l'enjeu :

· Gradio appelle une constante Starlette dépréciée à CHAQUE requête mise en
  file. Pendant une génération, la console se remplit de dizaines de lignes
  identiques que personne ne peut corriger — c'est du code de Gradio.

· Mais le même mécanisme a servi à trouver un vrai défaut chez nous :
  « A function returned too many output values » signalait que les boutons
  Stop calculaient un message d'annulation puis le jetaient. Masquer largement
  les UserWarning de Gradio aurait caché ça aussi.

Le filtre doit donc être assez précis pour taire l'un sans taire l'autre.
"""
import ast
import unittest
import warnings
from pathlib import Path

import app  # installe les filtres à l'import

ROOT = Path(__file__).resolve().parent.parent


def _warn_as(module_name: str, category, message="ceci est un essai"):
    """Émet un avertissement EN SE FAISANT PASSER pour `module_name`.

    Le filtre `module=` de `warnings` compare le nom du module d'où part
    l'avertissement : le tester depuis ce fichier ne prouverait rien.
    """
    import importlib
    mod = importlib.import_module(module_name)
    glb = dict(vars(mod))
    glb["_C"] = category
    glb["_M"] = message
    exec("import warnings as _w; _w.warn(_M, _C)", glb)  # noqa: S102


class StarletteFloodTests(unittest.TestCase):
    def setUp(self):
        try:
            from starlette.exceptions import StarletteDeprecationWarning
        except ImportError:  # pragma: no cover
            self.skipTest("starlette absent")
        self.sdw = StarletteDeprecationWarning

    def test_the_class_is_a_user_warning_not_a_deprecation_one(self):
        """La raison pour laquelle le premier filtre ne mordait pas. Si
        Starlette change d'avis, ce test le dira avant l'utilisateur."""
        self.assertTrue(issubclass(self.sdw, UserWarning))
        self.assertFalse(issubclass(self.sdw, DeprecationWarning))

    def test_it_is_silenced_when_it_comes_from_gradio(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            for f in app.warnings.filters:
                pass
            warnings.resetwarnings()
            app.warnings.filterwarnings(
                "ignore", category=self.sdw, module="gradio")
            _warn_as("gradio.routes", self.sdw,
                     "'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated")
        self.assertEqual([str(w.message) for w in seen], [])

    def test_it_still_speaks_up_from_anywhere_else(self):
        """Un filtre trop large masquerait le même avertissement venu de
        notre propre code, où il vaudrait la peine d'être corrigé."""
        with warnings.catch_warnings(record=True) as seen:
            warnings.resetwarnings()
            app.warnings.filterwarnings(
                "ignore", category=self.sdw, module="gradio")
            warnings.warn("depuis nos modules", self.sdw)
        self.assertEqual(len(seen), 1)


class AppActuallyInstallsTheFilterTests(unittest.TestCase):
    """Les tests ci-dessus posent le filtre eux-mêmes : ils prouvent qu'il
    fonctionne, pas qu'`app.py` le pose. On vérifie donc dans un interpréteur
    NEUF, où seul l'import d'app.py a eu lieu."""

    def test_importing_app_is_enough(self):
        import subprocess
        import sys
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import warnings, app\n"
            "from starlette.exceptions import StarletteDeprecationWarning as S\n"
            "import gradio.routes as r\n"
            "with warnings.catch_warnings(record=True) as seen:\n"
            "    g = dict(vars(r)); g['_S'] = S\n"
            "    exec(\"import warnings as w; w.warn('x', _S)\", g)\n"
            "print('SILENCE' if not seen else 'BRUIT')\n"
        ) % str(ROOT)
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        self.assertIn("SILENCE", proc.stdout, proc.stdout + proc.stderr[-800:])


class RealProblemsStayVisibleTests(unittest.TestCase):
    def test_a_plain_user_warning_from_gradio_is_not_silenced(self):
        """« A function returned too many output values » arrive par cette
        voie : c'est un UserWarning émis par gradio/blocks.py. Il doit
        continuer de passer."""
        with warnings.catch_warnings(record=True) as seen:
            warnings.resetwarnings()
            try:
                from starlette.exceptions import StarletteDeprecationWarning
                app.warnings.filterwarnings(
                    "ignore", category=StarletteDeprecationWarning,
                    module="gradio")
            except ImportError:  # pragma: no cover
                pass
            app.warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module="gradio")
            _warn_as("gradio.blocks", UserWarning,
                     "A function returned too many output values")
        self.assertEqual(len(seen), 1, "l'avertissement utile a été masqué")


class StopButtonsTests(unittest.TestCase):
    """Ce que l'avertissement avait révélé : neuf boutons Stop appelaient une
    fonction qui RENVOIE un message, avec `outputs=None`. Le message était
    calculé puis jeté — appuyer sur Stop ne confirmait rien à l'écran."""

    def test_no_handler_is_wired_without_outputs(self):
        offenders = []
        for path in sorted((ROOT / "atelier" / "ui").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                if getattr(n.func, "attr", "") not in ("click", "change",
                                                       "submit", "select"):
                    continue
                for kw in n.keywords:
                    if kw.arg == "outputs" and isinstance(kw.value, ast.Constant) \
                            and kw.value.value is None:
                        offenders.append(f"{path.name}:{n.lineno}")
        self.assertEqual(offenders, [], "sans sortie : " + ", ".join(offenders))

    def test_the_cancel_message_reaches_a_component(self):
        src = "\n".join(
            p.read_text(encoding="utf-8")
            for p in (ROOT / "atelier" / "ui").glob("*.py"))
        self.assertIn("widgets.stop_into_status", src)
        self.assertIn("widgets.stop_into_log", src)
        self.assertNotIn("cancel(), outputs=None", src)

    def test_appending_to_a_log_keeps_what_was_there(self):
        """Écrire dans un journal, c'est le remplacer : la trace de ce qu'on
        vient d'interrompre est justement ce qu'on veut relire."""
        from atelier.ui import widgets

        captured = {}

        class FakeButton:
            def click(self, fn, inputs=None, outputs=None, cancels=None):
                captured["fn"] = fn

        widgets.stop_into_log(FakeButton(), lambda: "⏹️ annulé", None, [])
        self.assertEqual(captured["fn"]("ligne 1\nligne 2"),
                         "ligne 1\nligne 2\n⏹️ annulé")
        self.assertEqual(captured["fn"](""), "⏹️ annulé")


if __name__ == "__main__":
    unittest.main()
