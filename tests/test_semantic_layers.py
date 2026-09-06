"""Étiquetage sémantique des calques (CLIP) : vocabulaire et fusion.

CLIP lui-même n'est pas testé ici — il faudrait 600 Mo de poids. Ce qui EST
testé, c'est tout ce qui l'entoure et qui décide de la qualité du résultat : la
forme du vocabulaire, la présence des catégories poubelle sans lesquelles un
classifieur zéro-shot étiquette « voiture » un bout de bitume flou, et la
règle de fusion qui recolle les morceaux d'un même objet.
"""
import importlib.util
import unittest

import numpy as np

from atelier import settings
from atelier.engine import vocab


def _run_layers():
    spec = importlib.util.spec_from_file_location(
        "run_layers", settings.ROOT / "scripts" / "tools" / "run_layers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VocabTests(unittest.TestCase):
    def test_every_entry_is_well_formed(self):
        for name, variants, depth in vocab.entries():
            self.assertTrue(name, "étiquette vide")
            self.assertTrue(variants, f"« {name} » n'a aucune formulation")
            self.assertTrue(all(isinstance(v, str) and v for v in variants))
            self.assertGreaterEqual(depth, 0.0)
            self.assertLessEqual(depth, 1.0)

    def test_labels_are_unique(self):
        names = [n for n, _v, _d in vocab.entries()]
        self.assertEqual(len(names), len(set(names)))

    def test_junk_categories_exist(self):
        """Sans elles, CLIP est FORCÉ de choisir un objet réel pour du vide."""
        self.assertTrue(vocab.JUNK_LABELS)
        entry_names = {n for n, _v, _d in vocab.entries()}
        for junk in vocab.JUNK_LABELS:
            self.assertIn(junk, entry_names)

    def test_prompts_use_the_clip_template_and_map_back(self):
        texts, owner = vocab.prompts()
        self.assertEqual(len(texts), len(owner))
        self.assertTrue(all(t.startswith("a photo of ") for t in texts))
        # Chaque entrée doit être représentée par au moins une formulation.
        self.assertEqual(set(owner), set(range(len(vocab.entries()))))

    def test_depth_priors_are_sane(self):
        """Le ciel derrière, les sujets devant — c'est le but même du champ."""
        self.assertLess(vocab.typical_depth("ciel"),
                        vocab.typical_depth("route"))
        self.assertLess(vocab.typical_depth("route"),
                        vocab.typical_depth("véhicule"))
        self.assertLess(vocab.typical_depth("bâtiment"),
                        vocab.typical_depth("personne"))

    def test_unknown_label_gets_a_neutral_depth(self):
        self.assertEqual(vocab.typical_depth("n'existe pas"), 0.5)


class SemanticMergeTests(unittest.TestCase):
    """La fusion est ce qui fait passer l'outil de « formes » à « objets »."""

    H = W = 200

    def _box(self, y0, y1, x0, x1):
        m = np.zeros((self.H, self.W), bool)
        m[y0:y1, x0:x1] = True
        return m

    def test_adjacent_pieces_of_one_object_are_merged(self):
        body = self._box(100, 140, 20, 80)
        door = self._box(105, 135, 78, 100)
        wheel = self._box(135, 150, 30, 50)
        masks = [body, door, wheel]
        labels = [("véhicule", 0.05)] * 3
        out_m, out_l = _run_layers()._merge_by_label(masks, labels,
                                                     lambda _m: None)
        self.assertEqual(len(out_m), 1)
        self.assertEqual(out_l[0][0], "véhicule")
        self.assertEqual(int(out_m[0].sum()), int((body | door | wheel).sum()))

    def test_distant_objects_of_the_same_kind_stay_separate(self):
        """Deux voitures aux extrémités ne sont pas la même voiture."""
        left = self._box(100, 140, 10, 60)
        right = self._box(100, 140, 160, 195)
        out_m, _l = _run_layers()._merge_by_label(
            [left, right], [("véhicule", 0.05)] * 2, lambda _m: None)
        self.assertEqual(len(out_m), 2)

    def test_different_labels_are_never_merged(self):
        a = self._box(100, 140, 20, 80)
        b = self._box(100, 140, 79, 120)          # collés, mais nature ≠
        out_m, _l = _run_layers()._merge_by_label(
            [a, b], [("véhicule", 0.05), ("personne", 0.05)], lambda _m: None)
        self.assertEqual(len(out_m), 2)

    def test_merged_group_keeps_the_most_confident_margin(self):
        a = self._box(100, 140, 20, 80)
        b = self._box(105, 135, 78, 100)
        _m, out_l = _run_layers()._merge_by_label(
            [a, b], [("véhicule", 0.02), ("véhicule", 0.09)], lambda _m: None)
        self.assertAlmostEqual(out_l[0][1], 0.09)

    def test_nothing_to_merge_leaves_the_input_untouched(self):
        a = self._box(0, 20, 0, 20)
        b = self._box(150, 180, 150, 180)
        masks = [a, b]
        out_m, _l = _run_layers()._merge_by_label(
            masks, [("ciel", 0.1), ("sol", 0.1)], lambda _m: None)
        self.assertIs(out_m, masks)


if __name__ == "__main__":
    unittest.main()
