"""Onglet Xanax : l'entrée est une phrase de la vie courante, pas un prompt.

Ce que ces tests protègent n'est pas du code compliqué — c'est une INTENTION,
et une intention se perd sans bruit. Le style « xanax » de l'améliorateur
existe parce que les deux autres réclament un objectif, un éclairage et une
composition : appliqués ici, ils produiraient une belle photo, c'est-à-dire
l'échec exact de cet onglet. Le jour où quelqu'un rebranche l'améliorateur sur
`generic` « parce que c'est le défaut », rien ne casse — les images deviennent
simplement jolies, et personne ne relie ça à un commit.
"""
import ast
import unittest
from pathlib import Path

from atelier.engine import tools
from atelier.ui import xanax_tab

_RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "tools" / "run_enhance.py"


def _runner_consts() -> dict:
    """Constantes de run_enhance.py SANS importer torch/transformers.

    `STYLES` pointe vers d'autres constantes (`SYSTEM_XANAX`…) : literal_eval
    seul cale dessus, on résout donc les noms déjà rencontrés.
    """
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    out: dict = {}

    def value(node):
        if isinstance(node, ast.Name):
            return out[node.id]
        # SYSTEM_GENERIC = "…" + _CORE : une concaténation, pas un littéral.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        if isinstance(node, ast.Dict):
            return {ast.literal_eval(k): value(v)
                    for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", "")
            try:
                out[name] = value(node.value)
            except (ValueError, KeyError, TypeError):
                pass
    return out


class EnhancerStyleTests(unittest.TestCase):
    def setUp(self):
        self.consts = _runner_consts()

    def test_the_xanax_style_exists_on_both_sides(self):
        """Un style connu de l'application mais pas du runner échoue à
        l'exécution, avec un `choices` d'argparse pour seul message."""
        self.assertIn("xanax", tools.ENHANCE_STYLES)
        self.assertIn("xanax", self.consts["STYLES"])

    def test_the_tab_asks_for_the_xanax_style(self):
        src = Path(xanax_tab.__file__).read_text(encoding="utf-8")
        self.assertIn('style="xanax"', src)
        for other in ('style="generic"', 'style="krea2"'):
            self.assertNotIn(other, src)

    def test_the_xanax_prompt_forbids_photographic_craft(self):
        """Le cœur de la consigne : pas d'objectif, pas d'éclairage, pas de
        composition. Sans ça, l'améliorateur retombe dans ses habitudes."""
        sysmsg = self.consts["SYSTEM_XANAX"].lower()
        for banned in ("lens", "aperture", "depth of field", "lighting setup",
                       "golden hour", "bokeh", "cinematic"):
            self.assertIn(banned, sysmsg,
                          f"« {banned} » n'est plus explicitement interdit")
        self.assertIn("never add photographic craft", sysmsg)

    def test_the_xanax_prompt_explains_the_translation_job(self):
        # Le modèle d'image ne connaît pas les enseignes françaises : c'est
        # l'améliorateur qui doit décrire la CHOSE, pas la marque.
        sysmsg = self.consts["SYSTEM_XANAX"]
        self.assertIn("Flunch", sysmsg)
        self.assertIn("cafeteria", sysmsg)

    def test_the_other_styles_are_untouched(self):
        """Le style Xanax ne devait rien changer aux onglets normaux."""
        for name in ("generic", "krea2"):
            self.assertIn(name, self.consts["STYLES"])
        self.assertIn("85mm", self.consts["SYSTEM_GENERIC"])


class AnecdoteBankTests(unittest.TestCase):
    def test_the_bank_is_not_a_list_of_image_descriptions(self):
        """Une anecdote parle de SOI. « a man waiting for the bus » est une
        description : c'est précisément ce qu'on ne veut plus proposer."""
        for line in xanax_tab.ANECDOTES:
            self.assertFalse(line.lower().startswith(("a ", "an ", "the ")),
                             f"« {line} » est formulé comme une description")

    def test_every_anecdote_is_a_short_first_person_sentence(self):
        """Ton de carnet, pas de légende : minuscule à l'attaque.

        Sauf quand le premier mot est un nom propre — « Grandad », « New
        Year's Eve » — qui garde sa majuscule en anglais quoi qu'il arrive.
        Les lister vaut mieux que de relâcher la règle pour tout le monde.
        """
        proper = ("Grandad", "New Year")
        for line in xanax_tab.ANECDOTES:
            self.assertLess(len(line), 80, line)
            self.assertEqual(line, line.strip())
            self.assertTrue(line[0].islower() or line.startswith(proper),
                            f"« {line} » : ton de carnet")

    def test_no_duplicate(self):
        self.assertEqual(len(set(xanax_tab.ANECDOTES)),
                         len(xanax_tab.ANECDOTES))


class PromptAssemblyTests(unittest.TestCase):
    def test_the_fixed_style_still_leads(self):
        got = xanax_tab.build_prompt("j'ai mangé chez Flunch avec Mamie")
        self.assertTrue(got.startswith(xanax_tab.XANAX_STYLE))
        self.assertTrue(got.endswith("j'ai mangé chez Flunch avec Mamie"))

    def test_an_empty_subject_leaves_no_dangling_comma(self):
        self.assertEqual(xanax_tab.build_prompt(""), xanax_tab.XANAX_STYLE)
        self.assertEqual(xanax_tab.build_prompt("  ,  "), xanax_tab.XANAX_STYLE)

    def test_the_style_stays_mundane(self):
        # Garde-fou sur le style figé lui-même : il décrit une photo ratée.
        for word in ("amateur snapshot", "mundane", "no filter",
                     "careless framing"):
            self.assertIn(word, xanax_tab.XANAX_STYLE)


if __name__ == "__main__":
    unittest.main()
