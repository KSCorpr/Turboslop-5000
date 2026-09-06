"""Fusion sémantique des calques : ne coller QUE ce qui se touche.

Ces tests viennent d'un PSD ouvert dans Photoshop : le calque « véhicule »
faisait 25,7 % de l'image et contenait la voiture, la fumée à l'autre bout du
cadre et le grillage du fond. Trois choses sans rapport, un seul calque.

La cause n'était pas l'étiquetage mais la mesure du voisinage : on comparait
les BOÎTES ENGLOBANTES. Sur une photo large, la boîte d'une voiture couvre la
moitié du cadre, donc tout ce qui porte la même étiquette « tombe à côté »
d'elle, où que ce soit dans l'image.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from atelier.engine import masks as M

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_layers", ROOT / "scripts" / "tools" / "run_layers.py")
run_layers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_layers)


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


class TouchesTests(unittest.TestCase):
    def test_two_boxes_that_share_an_edge_touch(self):
        a = box(200, 400, 50, 100, 10, 100)
        b = box(200, 400, 50, 100, 100, 190)
        self.assertTrue(M.touches(a, b))

    def test_two_boxes_at_opposite_ends_do_not(self):
        a = box(200, 400, 50, 100, 0, 40)
        b = box(200, 400, 50, 100, 360, 400)
        self.assertFalse(M.touches(a, b))

    def test_overlapping_bounding_boxes_are_not_enough(self):
        """LE cas du bug : deux zones en L, boîtes qui se croisent, aucun
        pixel en commun ni voisin."""
        a = box(400, 400, 0, 40, 0, 400)        # bande tout en haut
        b = box(400, 400, 0, 400, 360, 400)     # bande à droite
        b[:60] = False                          # ... qui commence PLUS BAS
        # Les boîtes se recouvrent largement…
        self.assertTrue(a[:, 360:].shape == b[:, 360:].shape)
        # … et pourtant les pixels ne se touchent pas.
        self.assertFalse(M.touches(a, b))

    def test_an_empty_mask_touches_nothing(self):
        self.assertFalse(M.touches(np.zeros((50, 50), bool),
                                   box(50, 50, 0, 10, 0, 10)))

    def test_it_survives_a_very_large_image(self):
        # 4096×2137, la taille de l'image du rapport : l'adjacence se juge sur
        # une grille réduite, ça doit rester instantané.
        a = box(2137, 4096, 100, 400, 100, 900)
        b = box(2137, 4096, 100, 400, 900, 1600)
        self.assertTrue(M.touches(a, b))
        self.assertFalse(M.touches(a, box(2137, 4096, 1800, 2000,
                                          3500, 4000)))


class MergeByLabelTests(unittest.TestCase):
    def setUp(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)

    # Le voisinage tel qu'il était mesuré AVANT — gardé ici comme référence,
    # pour que le test montre ce qu'il empêche de revenir.
    @staticmethod
    def _bbox_near(a, b, gap=12):
        ays, axs = np.nonzero(a)
        bys, bxs = np.nonzero(b)
        return not (axs.max() + gap < bxs.min() or bxs.max() + gap < axs.min()
                    or ays.max() + gap < bys.min()
                    or bys.max() + gap < ays.min())

    def test_the_photoshop_case_no_longer_merges(self):
        """Voiture, fumée, grillage : une CHAÎNE de boîtes qui se chevauchent
        sans qu'un seul pixel se touche.

        C'est le mécanisme exact du PSD raté. La voiture et le grillage sont
        aux deux bouts et leurs boîtes ne se croisent même pas ; c'est la
        fumée, au milieu, qui chevauche les deux et fait le pont. Un seul
        calque de 25 % de l'image en sortait.
        """
        H, W = 400, 800
        car = box(H, W, 200, 340, 200, 600)
        smoke = box(H, W, 150, 190, 590, 780)
        fence = box(H, W, 20, 140, 700, 790)

        # L'ancienne règle : chaîne complète, donc un seul groupe.
        self.assertTrue(self._bbox_near(car, smoke))
        self.assertTrue(self._bbox_near(smoke, fence))
        self.assertFalse(self._bbox_near(car, fence))

        # La réalité des pixels : rien ne se touche.
        self.assertFalse(M.touches(car, smoke))
        self.assertFalse(M.touches(smoke, fence))

        out_m, out_l = run_layers._merge_by_label(
            [car, smoke, fence], [("véhicule", 0.2)] * 3, self.log)
        self.assertEqual(len(out_m), 3, "\n".join(self.lines))
        self.assertEqual(len(out_l), 3)
        biggest = max(int(m.sum()) for m in out_m) / (H * W)
        self.assertLess(biggest, 0.20, "un calque avale encore la scène")

    def test_pieces_of_one_object_still_merge(self):
        """Ce que la fusion doit continuer de faire : recoller carrosserie,
        portière et roue en une voiture."""
        H, W = 400, 800
        body = box(H, W, 200, 300, 200, 500)
        door = box(H, W, 200, 300, 500, 560)
        wheel = box(H, W, 290, 340, 240, 300)
        out_m, out_l = run_layers._merge_by_label(
            [body, door, wheel], [("véhicule", 0.2)] * 3, self.log)
        self.assertEqual(len(out_m), 1, "\n".join(self.lines))
        self.assertEqual(int(out_m[0].sum()),
                         int((body | door | wheel).sum()))

    def test_different_labels_never_merge_even_when_touching(self):
        H, W = 200, 400
        a = box(H, W, 50, 150, 10, 200)
        b = box(H, W, 50, 150, 200, 390)
        out_m, _ = run_layers._merge_by_label(
            [a, b], [("véhicule", 0.2), ("route", 0.2)], self.log)
        self.assertEqual(len(out_m), 2)

    def test_an_uncertain_label_does_not_merge(self):
        """Fusionner sur un mot tiré au sort colle deux objets sans rapport."""
        H, W = 200, 400
        a = box(H, W, 50, 150, 10, 200)
        b = box(H, W, 50, 150, 200, 390)
        out_m, _ = run_layers._merge_by_label(
            [a, b], [("véhicule", 0.001), ("véhicule", 0.2)], self.log,
            min_margin=0.012)
        self.assertEqual(len(out_m), 2)
        self.assertTrue(any("marge" in l for l in self.lines), self.lines)

    def test_a_group_cannot_swallow_the_image(self):
        """Même contiguë, une chaîne de même nom ne devient pas un fond."""
        H, W = 100, 100
        a = box(H, W, 0, 50, 0, 100)      # 50 %
        b = box(H, W, 50, 90, 0, 100)     # 40 % — collée à la précédente
        out_m, _ = run_layers._merge_by_label(
            [a, b], [("mur", 0.2)] * 2, self.log, max_share=0.35)
        self.assertEqual(len(out_m), 2)
        self.assertTrue(any("taille" in l for l in self.lines), self.lines)

    def test_merging_is_transitive_through_contact(self):
        """A touche B, B touche C, A ne touche pas C : un seul objet quand
        même — c'est une chaîne continue de matière."""
        H, W = 200, 600
        a = box(H, W, 80, 120, 10, 200)
        b = box(H, W, 80, 120, 200, 400)
        c = box(H, W, 80, 120, 400, 590)
        self.assertFalse(M.touches(a, c))
        out_m, _ = run_layers._merge_by_label(
            [a, b, c], [("véhicule", 0.2)] * 3, self.log, max_share=0.9)
        self.assertEqual(len(out_m), 1)


class PartitionKeepsLabelsAlignedTests(unittest.TestCase):
    """Le second défaut du même PSD, plus sournois : la découpe supprime des
    calques AU MILIEU de la liste, et l'appelant tronquait les étiquettes par
    la FIN. Tous les noms situés après le premier calque disparu glissaient
    d'un cran — la voiture prenait le nom du mur."""

    def log(self, _msg):
        pass

    def test_indices_report_which_layers_survived(self):
        H, W = 100, 100
        back = box(H, W, 0, 100, 0, 100)
        hidden = box(H, W, 10, 20, 10, 20)     # sera entièrement recouvert
        front = box(H, W, 0, 60, 0, 100)
        kept, alive = run_layers._partition([back, hidden, front], self.log)
        self.assertEqual(len(kept), len(alive))
        self.assertNotIn(1, alive, "le calque caché doit disparaître")
        self.assertIn(0, alive)
        self.assertIn(2, alive)

    def test_labels_follow_the_surviving_layers(self):
        H, W = 100, 100
        layers = [box(H, W, 0, 100, 0, 100),
                  box(H, W, 10, 20, 10, 20),
                  box(H, W, 0, 60, 0, 100)]
        labels = [("route", 0.2), ("mur", 0.2), ("véhicule", 0.2)]
        _kept, alive = run_layers._partition(layers, self.log)
        got = [labels[i][0] for i in alive]
        self.assertEqual(got, ["route", "véhicule"])

    def test_a_layer_reduced_to_a_fringe_is_dropped(self):
        H, W = 100, 100
        back = box(H, W, 0, 100, 0, 100)
        almost = box(H, W, 0, 99, 0, 100)      # ne laisse qu'une ligne
        kept, alive = run_layers._partition([back, almost], self.log,
                                            min_area=500)
        self.assertEqual(alive, [1])
        self.assertEqual(len(kept), 1)

    def test_the_stack_still_rebuilds_the_image_exactly(self):
        """Propriété qui ne doit pas être perdue en route : les calques sont
        disjoints et leur union couvre ce que couvraient les masques."""
        H, W = 120, 200
        a = box(H, W, 0, 120, 0, 200)
        b = box(H, W, 20, 80, 20, 120)
        c = box(H, W, 40, 100, 60, 180)
        kept, _alive = run_layers._partition([a, b, c], self.log)
        stack = np.zeros((H, W), int)
        for m in kept:
            stack += m.astype(int)
        self.assertTrue((stack <= 1).all(), "un pixel peint deux fois")
        self.assertTrue(((stack > 0) == (a | b | c)).all())


if __name__ == "__main__":
    unittest.main()


class GridComparisonTests(unittest.TestCase):
    """Comparer sur grilles plutôt qu'à pleine résolution.

    Sans ça, l'outil est inutilisable au-delà du petit format : mesuré sur une
    image 4096×4096, un seul IoU coûte 77 ms, soit 14 minutes pour 150 zones,
    plus 70 s d'adjacence. Ces tests vérifient que le raccourci répond la même
    chose que le calcul exact — sinon il ne raccourcit rien, il change le
    résultat.
    """

    def test_identical_masks_score_one(self):
        m = box(400, 400, 50, 300, 50, 300)
        g = M.coverage_grid(m)
        self.assertAlmostEqual(M.grid_iou(g, g), 1.0, places=5)

    def test_disjoint_masks_score_zero(self):
        a = M.coverage_grid(box(400, 400, 0, 100, 0, 100))
        b = M.coverage_grid(box(400, 400, 300, 400, 300, 400))
        self.assertEqual(M.grid_iou(a, b), 0.0)

    def test_it_tracks_the_exact_iou(self):
        """Le seuil de doublon est à 0,75 : l'approximation doit rester bien
        en deçà de l'erreur qui ferait basculer une décision."""
        H = W = 512
        base = box(H, W, 100, 400, 100, 400)
        for shift in (0, 20, 60, 120, 200):
            other = box(H, W, 100, 400, 100 + shift, 400 + shift)
            inter = float((base & other).sum())
            exact = inter / float((base | other).sum()) if inter else 0.0
            approx = M.grid_iou(M.coverage_grid(base),
                                M.coverage_grid(other))
            self.assertAlmostEqual(exact, approx, delta=0.02,
                                   msg=f"décalage {shift}")

    def test_a_near_duplicate_is_still_recognised(self):
        H = W = 2048
        a = box(H, W, 200, 1200, 200, 1200)
        b = box(H, W, 208, 1208, 196, 1196)      # même zone, 8 px de décalage
        self.assertGreater(M.grid_iou(M.coverage_grid(a),
                                      M.coverage_grid(b)), 0.75)

    def test_two_distinct_neighbours_are_not_duplicates(self):
        H = W = 2048
        a = box(H, W, 200, 1200, 200, 700)
        b = box(H, W, 200, 1200, 700, 1200)      # collée, mais AUTRE zone
        self.assertLess(M.grid_iou(M.coverage_grid(a),
                                   M.coverage_grid(b)), 0.75)

    def test_coverage_keeps_areas_not_just_presence(self):
        """Un « ou » par blocs ferait grossir les petites zones et les ferait
        passer pour identiques à leurs voisines."""
        H = W = 1024
        speck = box(H, W, 500, 505, 500, 505)
        big = box(H, W, 400, 600, 400, 600)
        self.assertLess(M.grid_iou(M.coverage_grid(speck),
                                   M.coverage_grid(big)), 0.1)

    def test_the_grid_is_small_whatever_the_image(self):
        for side in (512, 4096):
            g = M.coverage_grid(box(side, side, 0, side // 2, 0, side // 2))
            self.assertLessEqual(max(g.shape), M.COMPARE_GRID)
            self.assertLess(g.nbytes, 400_000)


class RawDuplicateRejectionTests(unittest.TestCase):
    """Dédoublonner AVANT de nettoyer : le nettoyage coûte 0,7 à 1,8 s par
    masque en 4096×4096, et SAM en rend des dizaines d'identiques."""

    def test_near_identical_raw_masks_are_dropped_before_cleaning(self):
        H = W = 256
        base = box(H, W, 40, 200, 40, 200)
        raws = [base] + [box(H, W, 40 + d, 200 + d, 40 - d, 200 - d)
                         for d in (1, 2, 3, 4)]
        lines = []
        kept = run_layers._filter_masks(raws, 0.004 * H * W, 0.85 * H * W,
                                        0.75, lines.append)
        self.assertEqual(len(kept), 1, "\n".join(lines))
        self.assertTrue(any("avant nettoyage" in l for l in lines), lines)

    def test_genuinely_different_masks_all_survive(self):
        H = W = 256
        raws = [box(H, W, 10, 100, 10, 100),
                box(H, W, 140, 240, 10, 100),
                box(H, W, 10, 100, 140, 240)]
        kept = run_layers._filter_masks(raws, 0.004 * H * W, 0.85 * H * W,
                                        0.75, lambda _m: None)
        self.assertEqual(len(kept), 3)
