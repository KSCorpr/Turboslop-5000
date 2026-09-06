"""Écriture PSD : le format doit être RELU correctement, pas seulement écrit.

Ces tests s'appuient sur `psd-tools` quand il est disponible — c'est un lecteur
indépendant, donc la seule façon honnête de vérifier qu'on produit un fichier
valide et pas seulement un fichier qui nous plaît. Sans lui, on se rabat sur les
propriétés structurelles qu'on peut vérifier seul.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from atelier.engine import psd


def _unpackbits(b: bytes) -> bytes:
    """Décodeur PackBits de référence (spécification Adobe/TIFF)."""
    out = bytearray()
    i = 0
    while i < len(b):
        n = b[i]
        i += 1
        if n < 128:
            out += b[i:i + n + 1]
            i += n + 1
        elif n > 128:
            out += bytes([b[i]]) * (257 - n)
            i += 1
    return bytes(out)


class PackBitsTests(unittest.TestCase):
    def test_round_trip_on_random_and_repetitive_data(self):
        import random
        random.seed(20260811)
        for trial in range(200):
            n = random.randint(1, 400)
            if trial % 3 == 0:
                # Cas réel d'une découpe : grandes plages identiques.
                data = bytes(random.choice([0, 0, 0, 255]) for _ in range(n))
            else:
                data = bytes(random.randrange(256) for _ in range(n))
            self.assertEqual(_unpackbits(psd._packbits(data)), data,
                             f"tirage {trial}")

    def test_repetitive_data_actually_shrinks(self):
        # Si la compression n'apportait rien, autant écrire du brut.
        data = b"\0" * 5000
        self.assertLess(len(psd._packbits(data)), len(data) // 10)


class CropTests(unittest.TestCase):
    def test_layer_is_cropped_to_its_content(self):
        rgba = np.zeros((100, 200, 4), "uint8")
        rgba[30:50, 60:90] = (255, 0, 0, 255)
        crop, top, left = psd._crop_to_content(rgba)
        self.assertEqual((top, left), (30, 60))
        self.assertEqual(crop.shape[:2], (20, 30))

    def test_empty_layer_does_not_crash(self):
        crop, top, left = psd._crop_to_content(np.zeros((50, 50, 4), "uint8"))
        self.assertEqual((top, left), (0, 0))
        self.assertEqual(crop.shape[:2], (1, 1))


def _scene(w=160, h=120):
    rgb = np.zeros((h, w, 3), "uint8")
    rgb[:, :] = (30, 40, 60)
    y, x = np.ogrid[:h, :w]
    disc = ((x - 60) ** 2 + (y - 60) ** 2) <= 30 ** 2
    rgb[disc] = (220, 90, 80)
    layer = np.zeros((h, w, 4), "uint8")
    layer[disc] = (220, 90, 80, 255)
    bg = np.dstack([rgb, np.full((h, w), 255, "uint8")])
    return rgb, bg, layer


class WritePsdTests(unittest.TestCase):
    def test_rejects_bad_input(self):
        rgb, bg, _ = _scene()
        with TemporaryDirectory() as d:
            p = Path(d) / "x.psd"
            with self.assertRaises(psd.PsdError):
                psd.write_psd(p, rgb, [])                       # aucun calque
            with self.assertRaises(psd.PsdError):
                psd.write_psd(p, rgb[:, :, 0], [("a", bg)])     # pas du RGB
            with self.assertRaises(psd.PsdError):
                psd.write_psd(p, rgb, [("a", rgb)])             # calque sans alpha

    def test_header_is_a_valid_psd(self):
        import struct
        rgb, bg, layer = _scene()
        with TemporaryDirectory() as d:
            p = psd.write_psd(Path(d) / "x.psd", rgb,
                              [("Fond", bg), ("Objet", layer)])
            raw = p.read_bytes()
        self.assertEqual(raw[:4], b"8BPS")
        self.assertEqual(struct.unpack(">H", raw[4:6])[0], 1)   # version 1
        # 4 (signature) + 2 (version) + 6 (réservé) = 12 octets d'en-tête avant
        # le bloc « canaux / hauteur / largeur / profondeur / mode ».
        chans, height, width, depth, mode = struct.unpack(">HIIHH", raw[12:26])
        self.assertEqual((chans, depth, mode), (3, 8, 3))       # RGB 8 bits
        self.assertEqual((width, height), (160, 120))

    def test_readable_by_psd_tools(self):
        try:
            from psd_tools import PSDImage
        except ImportError:
            self.skipTest("psd-tools absent")
        rgb, bg, layer = _scene()
        with TemporaryDirectory() as d:
            p = psd.write_psd(Path(d) / "x.psd", rgb,
                              [("Fond", bg), ("Objet ★", layer)])
            doc = PSDImage.open(p)
            names = [lay.name for lay in doc]
            self.assertEqual(names, ["Fond", "Objet ★"])   # Unicode préservé
            self.assertEqual((doc.width, doc.height), (160, 120))
            # Le calque doit être recadré, pas stocké en pleine toile.
            obj = list(doc)[1]
            self.assertLess(obj.width, doc.width)
            # Et le composite doit ressembler à ce qu'on a fourni.
            got = np.asarray(doc.composite().convert("RGB"))
            self.assertLess(float(np.abs(got.astype("int16")
                                         - rgb.astype("int16")).mean()), 2.0)


if __name__ == "__main__":
    unittest.main()
