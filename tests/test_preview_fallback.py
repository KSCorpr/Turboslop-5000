"""Aperçu de secours : il doit tenir debout tout seul.

Sa seule raison d'être est de marcher quand le chemin normal échoue. Un
aperçu de secours qui redemanderait le fichier au serveur ne servirait à
rien — c'est exactement ce qui casse. D'où le test central : la vignette est
un `data:` URI, donc les pixels sont DANS la page.
"""
import base64
import re
import unittest
from pathlib import Path

from atelier.ui import preview

_HEAD = re.compile(r"^data:image/png;base64,")


def _img(size=(1600, 1200), mode="RGB"):
    from PIL import Image
    return Image.new(mode, size, (200, 90, 40))


class DataUriTests(unittest.TestCase):
    def test_the_pixels_travel_in_the_page(self):
        uri = preview.data_uri(_img())
        self.assertRegex(uri, _HEAD)
        raw = base64.b64decode(uri.split(",", 1)[1])
        self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_it_never_points_at_the_server(self):
        """La seule ligne qui compte : aucune adresse, donc aucune requête."""
        out = preview.html(_img())
        self.assertIn("data:image/png;base64,", out)
        for forbidden in ("/gradio_api/", "/file=", "http://", "https://"):
            self.assertNotIn(forbidden, out)

    def test_large_images_are_scaled_down(self):
        """Un data: URI voyage dans le HTML et n'est pas mis en cache : une
        photo de 24 Mpx en base64 rendrait la page inutilisable."""
        from PIL import Image
        import io
        uri = preview.data_uri(_img((4000, 3000)))
        im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(max(im.size), preview.MAX_SIDE)
        self.assertEqual(im.size, (preview.MAX_SIDE, preview.MAX_SIDE * 3 // 4))
        self.assertLess(len(uri), 400_000, "vignette trop lourde pour la page")

    def test_small_images_are_not_blown_up(self):
        from PIL import Image
        import io
        uri = preview.data_uri(_img((120, 90)))
        im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(im.size, (120, 90))

    def test_transparency_survives(self):
        from PIL import Image
        import io
        uri = preview.data_uri(_img((300, 200), mode="RGBA"))
        im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(im.mode, "RGBA")

    def test_it_accepts_a_path_as_well_as_an_image(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            _img((200, 150)).save(p)
            self.assertRegex(preview.data_uri(str(p)), _HEAD)


class DegradationTests(unittest.TestCase):
    """Un aperçu de secours qui lève une exception ferait tomber l'outil qu'il
    est censé dépanner."""

    def test_no_image_gives_a_readable_message(self):
        self.assertEqual(preview.data_uri(None), "")
        self.assertIn("No image loaded", preview.html(None))

    def test_a_corrupt_file_does_not_raise(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "coupé.png"
            p.write_bytes(b"\x89PNG et rien de plus")   # en-tete puis rien
            self.assertEqual(preview.data_uri(str(p)), "")
            self.assertIn("No image loaded", preview.html(str(p)))

    def test_a_missing_file_does_not_raise(self):
        self.assertEqual(preview.data_uri("/nulle/part/x.png"), "")


class WiringTests(unittest.TestCase):
    def test_both_click_based_tools_offer_it(self):
        """SAM et Calques se pilotent au CLIC sur l'image : ne pas la voir y
        est bloquant, pas seulement gênant."""
        src = (Path(__file__).resolve().parent.parent / "atelier" / "ui"
               / "toolkit_tab.py").read_text(encoding="utf-8")
        self.assertIn("s_image.change(preview.html", src)
        self.assertIn("lay_image.change(preview.html", src)
        self.assertEqual(src.count("_FALLBACK_TITLE"), 3)  # 1 def + 2 usages


if __name__ == "__main__":
    unittest.main()
