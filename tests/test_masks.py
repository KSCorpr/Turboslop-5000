"""Nettoyage des masques : ce qui sépare une soupe de zones d'un vrai calque.

Les cas testés ici viennent tous d'un résultat réel jugé inexploitable : des
« calques » faits de taches éparpillées, des découpes en gruyère, des objets de
premier plan supprimés parce qu'une zone plus grande les contenait, et des noms
du genre « Zone 9 — 0,48 % » qui n'apprennent rien.
"""
import unittest

import numpy as np

from atelier.engine import masks as M


class ConnectedComponentTests(unittest.TestCase):
    def test_separates_disjoint_blobs(self):
        m = np.zeros((40, 60), bool)
        m[5:15, 5:20] = True
        m[25:30, 40:50] = True
        _labels, n = M.label_components(m)
        self.assertEqual(n, 2)

    def test_diagonal_touch_is_not_connected(self):
        # Voisinage 4 : deux blocs qui ne se touchent qu'en coin restent deux
        # objets. C'est le comportement voulu — sinon une image bruitée finit
        # en une seule composante géante.
        m = np.zeros((10, 10), bool)
        m[2:4, 2:4] = True
        m[4:6, 4:6] = True
        _labels, n = M.label_components(m)
        self.assertEqual(n, 2)

    def test_empty_mask(self):
        _labels, n = M.label_components(np.zeros((10, 10), bool))
        self.assertEqual(n, 0)

    def test_full_mask_is_one_component(self):
        _labels, n = M.label_components(np.ones((10, 10), bool))
        self.assertEqual(n, 1)

    def test_scattered_specks_are_dropped(self):
        """Le défaut le plus visible : une « zone » faite de 20 miettes."""
        m = np.zeros((200, 200), bool)
        rng = np.random.default_rng(0)
        for _ in range(20):
            y, x = rng.integers(0, 190), rng.integers(0, 190)
            m[y:y + 5, x:x + 5] = True
        self.assertEqual(M.largest_components(m, min_area=200), [])


class FillHolesTests(unittest.TestCase):
    def test_interior_hole_is_filled(self):
        m = np.zeros((30, 30), bool)
        m[5:25, 5:25] = True
        m[10:14, 10:14] = False
        self.assertEqual(int(M.fill_holes(m).sum()), 20 * 20)

    def test_notch_open_to_the_edge_is_preserved(self):
        # Une échancrure ouverte sur l'extérieur n'est pas un trou : la boucher
        # déformerait la silhouette.
        m = np.zeros((30, 30), bool)
        m[5:25, 5:25] = True
        m[10:20, 20:25] = False
        self.assertEqual(int(M.fill_holes(m).sum()), int(m.sum()))


class FeatherTests(unittest.TestCase):
    def test_edge_gets_intermediate_values(self):
        m = np.zeros((40, 40), bool)
        m[10:30, 10:30] = True
        a = M.feather_alpha(m, radius=1)
        self.assertEqual(a.max(), 255)
        self.assertEqual(a.min(), 0)
        self.assertTrue(((a > 0) & (a < 255)).any(), "aucun pixel de transition")

    def test_centre_stays_opaque(self):
        m = np.zeros((40, 40), bool)
        m[10:30, 10:30] = True
        self.assertEqual(int(M.feather_alpha(m, radius=1)[20, 20]), 255)


class NamingTests(unittest.TestCase):
    def _solid(self, rgb_value):
        img = np.zeros((20, 20, 3), "uint8")
        img[:, :] = rgb_value
        return img, np.ones((20, 20), bool)

    def test_dark_grey_is_not_called_green(self):
        """Le cas qui a motivé la pondération par luminosité."""
        img, m = self._solid((70, 70, 75))
        self.assertEqual(M.dominant_color_name(img, m), "dark grey")

    def test_common_colors(self):
        for value, want in (((150, 190, 230), "light blue"),
                            ((230, 120, 40), "orange"),
                            ((20, 20, 22), "black"),
                            ((250, 250, 250), "white")):
            img, m = self._solid(value)
            self.assertEqual(M.dominant_color_name(img, m), want, value)

    def test_position(self):
        m = np.zeros((100, 100), bool)
        m[:20, :20] = True
        self.assertEqual(M.position_name(m), "top left")
        m2 = np.zeros((100, 100), bool)
        m2[40:60, 40:60] = True
        self.assertEqual(M.position_name(m2), "centre")

    def test_describe_is_informative(self):
        img = np.zeros((100, 100, 3), "uint8")
        img[:, :] = (230, 120, 40)
        m = np.zeros((100, 100), bool)
        m[:20, :20] = True
        name = M.describe(m, img, 0, 3)
        for bit in ("background", "top left", "orange", "%"):
            self.assertIn(bit, name)


class PartitionTests(unittest.TestCase):
    """Les calques doivent être disjoints ET reconstituer l'image."""

    def _partition(self, ordered):
        import importlib.util
        from atelier import settings
        spec = importlib.util.spec_from_file_location(
            "run_layers", settings.ROOT / "scripts" / "tools" / "run_layers.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # _partition rend aussi les INDICES survivants : la découpe
        # supprime des calques au milieu de la liste, et l'appelant doit
        # pouvoir réaligner les étiquettes (cf. test_layer_merge.py).
        masks, _alive = mod._partition(ordered, lambda _m: None)
        return masks

    def test_layers_become_disjoint_and_cover_everything(self):
        H = W = 60
        back = np.ones((H, W), bool)              # fond plein
        front = np.zeros((H, W), bool)
        front[20:40, 20:40] = True                # objet devant
        out = self._partition([back, front])
        self.assertFalse((out[0] & out[1]).any(), "les calques se chevauchent")
        union = out[0] | out[1]
        self.assertTrue(union.all(), "des pixels ne sont dans aucun calque")
        # L'objet de devant garde sa forme entière ; c'est le fond qui est creusé.
        self.assertEqual(int(out[1].sum()), 20 * 20)

    def test_fully_hidden_layer_is_removed(self):
        H = W = 40
        hidden = np.zeros((H, W), bool)
        hidden[10:20, 10:20] = True
        cover = np.ones((H, W), bool)
        self.assertEqual(len(self._partition([hidden, cover])), 1)


if __name__ == "__main__":
    unittest.main()
