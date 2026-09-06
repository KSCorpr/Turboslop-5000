"""The interface is written in English, and stays that way.

The application used to carry French source strings plus a translation table.
The table is gone: the strings themselves are English now. What can regress is
not a missing translation any more — it is a French string sneaking back into a
new tab, a new error message, a new log line. That is what these tests watch.

Two detectors, because one is not enough:

* **accents** — imparable, and it catches most of what would slip in;
* **a French word list** — for strings like `"12 fichier(s)"` where not a
  single accent appears. The words are chosen so they collide with no ordinary
  English word (hence no `on`, no `si`, no `plus`, no bare `To`/`Go`);
* **the metric units** — `Go` / `Mo` / `Ko` / `To` are French, and only after a
  number, which is what separates them from `To use` and `Go high resolution`.

A handful of accented words are English (`café`, `fête`, `naïve`): they are
listed as exceptions rather than allowed to weaken the accent rule.

Docstrings and comments are deliberately out of scope: they are the source
code's own language, not the interface's, and they are still French throughout.
"""
import ast
import re
import unittest
from pathlib import Path

from atelier import i18n

_ROOT = Path(__file__).resolve().parents[1]

_FRENCH_WORDS = """
le les un une des du et ou pas est sont dans pour par avec sur vous votre
vos ton tes cette ces qui que quoi moins tout tous toute toutes
aucun aucune chaque entre selon puis donc mais ses leur leurs nous
notre nos ils elles aux lui avant vers chez dont fichier fichiers dossier
dossiers calque calques modele modeles taille tailles choisir choisissez
cliquez lancez prenez mettez placez ajoutez utilisez telechargez installez
supprimez enregistrer annuler termine echec alors quand comme aussi tres
bien mieux pire trop beaucoup encore deja jamais toujours rien
quelque quelques autres meme memes seule seulement octets
lignes boutons onglet onglets reglage reglages defaut
cartes disque disques memoire vitesse lecture ecriture gestion aide
outils apercu reglage etape etapes couleur couleurs profondeur
moteur moteurs catalogue detourage restauration amelioration parametres
""".split()

# English words that legitimately carry an accent — the accent rule must not
# turn them into false positives.
_LOANWORDS = ("café", "cafés", "fête", "fêtes", "naïve", "résumé", "cliché",
              "façade", "crème", "déjà vu")

_FRENCH = re.compile(
    r"[éèêëàâçùûôîïÉÈÊÀÂÇÙÛÔÎÏœ]"
    r"|\b(" + "|".join(_FRENCH_WORDS) + r")\b", re.IGNORECASE)
# Units stay CASE-SENSITIVE: `2 Go` is French, `0 to 1` is not.
_UNITS = re.compile(r"\d\s*(Go|Mo|Ko|To)\b")


def _strip_loanwords(text: str) -> str:
    for word in _LOANWORDS:
        text = re.sub(re.escape(word), "", text, flags=re.IGNORECASE)
    return text


def _is_french(text: str) -> bool:
    clean = _strip_loanwords(text)
    return bool(_FRENCH.search(clean) or _UNITS.search(clean))


def _files() -> list[Path]:
    """The interface layer: every module that can put text on screen."""
    return sorted(_ROOT.glob("atelier/**/*.py")) + [_ROOT / "app.py"]


def _docstring_ids(tree: ast.Module) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                out.add(id(node.body[0].value))
    return out


def _french_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_ids(tree)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in skip
                and _is_french(node.value)):
            out.append((node.lineno, node.value))
    return out


class EnglishOnlyTests(unittest.TestCase):
    def test_no_french_string_reaches_the_interface(self):
        offenders = []
        for path in _files():
            for lineno, text in _french_literals(path):
                rel = path.relative_to(_ROOT)
                offenders.append(f"{rel}:{lineno}  {text[:70]!r}")
        self.assertEqual(offenders, [], "\n".join(offenders))


class IdentityLayerTests(unittest.TestCase):
    """`t()` and friends survive as a seam, and must not alter anything."""

    def test_t_returns_its_argument_unchanged(self):
        for value in ("🧰 Toolkit", "", "Done — {n} image(s).", None, 3):
            self.assertIs(i18n.t(value), value)

    def test_to_source_is_the_identity_too(self):
        # Menu choices are their own key now: a round trip must not move.
        label = "🧰 Toolkit"
        self.assertIs(i18n.to_source(i18n.t(label)), label)

    def test_the_language_is_english_and_setting_it_is_harmless(self):
        i18n.set_lang("fr")          # accepted, and deliberately ignored
        self.assertEqual(i18n.get_lang(), "en")
        self.assertEqual(i18n.init_from_prefs(), "en")


if __name__ == "__main__":
    unittest.main()
