"""Diagnostic « image cassée » : il doit dire vrai, surtout quand ça va mal.

Un diagnostic qui se trompe est pire que pas de diagnostic : il envoie
chercher au mauvais endroit. Ces tests vérifient donc surtout les VERDICTS —
qu'un cache hors du projet soit signalé, qu'un dossier non inscriptible ne
passe pas pour sain, et qu'aucun contrôle ne reste sans explication.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atelier import imgcheck, settings


class TempDirTests(unittest.TestCase):
    def test_the_environment_variable_wins(self):
        """C'est ce que Gradio lit réellement : le diagnostic doit lire la
        même chose, pas ce que l'application avait l'intention de poser."""
        with mock.patch.dict(os.environ, {"GRADIO_TEMP_DIR": "/quelque/part"}):
            self.assertEqual(imgcheck.temp_dir(), Path("/quelque/part"))

    def test_falls_back_to_the_system_temp(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(imgcheck.temp_dir().name, "gradio")


class WithinTests(unittest.TestCase):
    def test_a_child_is_within(self):
        self.assertTrue(imgcheck._is_within(settings.ROOT / "tmp" / "gradio",
                                            settings.ROOT))

    def test_a_sibling_is_not(self):
        self.assertFalse(imgcheck._is_within(Path("/tmp/gradio"),
                                             settings.ROOT))

    def test_a_missing_path_answers_instead_of_raising(self):
        # Le diagnostic tourne sur des machines cassées : un chemin qui
        # n'existe pas doit donner une réponse, pas une exception.
        self.assertFalse(imgcheck._is_within(Path("/nulle/part/ailleurs"),
                                             settings.ROOT))
        self.assertFalse(imgcheck._is_within(settings.ROOT,
                                             Path("/nulle/part/ailleurs")))


class TestImageTests(unittest.TestCase):
    def test_it_writes_a_real_png_and_reads_it_back(self):
        with tempfile.TemporaryDirectory() as d:
            dest, elapsed, err = imgcheck.write_test_image(Path(d) / "sub")
            self.assertEqual(err, "")
            self.assertIsNotNone(dest)
            self.assertTrue(dest.is_file())
            self.assertTrue(dest.read_bytes().startswith(b"\x89PNG"))
            self.assertGreaterEqual(elapsed, 0.0)

    def test_an_unwritable_folder_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            # chmod ne protège pas un dossier de la même manière sous Windows,
            # et root ignore les bits POSIX. Simuler l'erreur au point d'écriture
            # teste exactement le contrat sans dépendre de l'OS du runner.
            with mock.patch("PIL.Image.Image.save",
                            side_effect=PermissionError("accès refusé")):
                dest, _elapsed, err = imgcheck.write_test_image(Path(d) / "x")
            self.assertIsNone(dest)
            self.assertIn("accès refusé", err)


class ReportTests(unittest.TestCase):
    def test_a_healthy_setup_reports_nothing_wrong(self):
        with mock.patch.dict(
                os.environ,
                {"GRADIO_TEMP_DIR": str(settings.ROOT / "tmp" / "gradio")}):
            r = imgcheck.report()
        self.assertIn("Nothing abnormal", r.markdown)
        self.assertIsNotNone(r.test_image)
        self.assertTrue(Path(r.test_image).is_file())

    def test_the_report_carries_one_tile_per_format(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"GRADIO_TEMP_DIR": d}):
                r = imgcheck.report()
            # DANS le with : les tuiles vivent dans ce dossier temporaire.
            self.assertEqual(len(r.tiles), len(imgcheck.TEST_FORMATS))
            for path, label in r.tiles:
                self.assertTrue(Path(path).is_file(), label)

    def test_a_cache_outside_the_project_is_flagged(self):
        """Le cas qui explique les images cassées PAR INTERMITTENCE : dans le
        dossier temporaire du système, le ménage passe quand il veut."""
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"GRADIO_TEMP_DIR": d}):
                items, _dest = imgcheck.checks()
        bad = [c for c in items if c.ok is False]
        self.assertTrue(bad, "un cache hors du projet doit être signalé")
        self.assertTrue(any("project" in c.label for c in bad))

    def test_every_failed_check_explains_what_to_do(self):
        """Un ❌ sans explication laisse l'utilisateur exactement où il était."""
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"GRADIO_TEMP_DIR": d}):
                items, _dest = imgcheck.checks()
        for c in items:
            if c.ok is False:
                self.assertGreater(len(c.detail), 30, c.label)

    def test_marks_are_readable(self):
        self.assertEqual(imgcheck.Check(True, "x").mark, "✅")
        self.assertEqual(imgcheck.Check(False, "x").mark, "❌")
        self.assertEqual(imgcheck.Check(None, "x").mark, "•")


class FormatProbeTests(unittest.TestCase):
    """La tuile par format est le contrôle qui manquait : le diagnostic
    général disait « tout va bien » pendant que les imports cassaient."""

    def test_it_writes_one_readable_tile_per_format(self):
        with tempfile.TemporaryDirectory() as d:
            tiles, lines = imgcheck.format_probe(Path(d))
            self.assertEqual(len(tiles), len(imgcheck.TEST_FORMATS))
            self.assertEqual(len(lines), len(imgcheck.TEST_FORMATS))
            for path, _label in tiles:
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_a_missing_mime_is_flagged(self):
        """Si même après nos associations explicites Python ne rend aucun MIME,
        le rapport doit le signaler au lieu de prétendre que tout va bien."""
        with mock.patch("mimetypes.guess_type", return_value=(None, None)):
            with tempfile.TemporaryDirectory() as d:
                _tiles, lines = imgcheck.format_probe(Path(d))
        self.assertTrue(all(l.startswith("❌") for l in lines), lines)
        self.assertTrue(any("registry" in l for l in lines))

    def test_known_extensions_resolve_to_an_image_type(self):
        for _name, suffix in imgcheck.TEST_FORMATS:
            self.assertTrue(imgcheck.mime_of(suffix).startswith("image/"),
                            f"{suffix} -> {imgcheck.mime_of(suffix)!r}")


class RecentUploadsTests(unittest.TestCase):
    def test_our_own_test_tiles_are_not_counted_as_imports(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            (cache / "diagnostic").mkdir()
            (cache / "diagnostic" / "test_png.png").write_bytes(b"x")
            (cache / "abc123").mkdir()
            (cache / "abc123" / "photo.jpg").write_bytes(b"y")
            got = imgcheck.recent_uploads(cache)
        self.assertEqual([p.name for p in got], ["photo.jpg"])

    def test_the_most_recent_comes_first(self):
        import time as _time
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            for i in range(3):
                sub = cache / f"h{i}"
                sub.mkdir()
                (sub / f"f{i}.png").write_bytes(b"z")
                _time.sleep(0.01)
            got = imgcheck.recent_uploads(cache)
        self.assertEqual([p.name for p in got], ["f2.png", "f1.png", "f0.png"])

    def test_a_missing_cache_answers_empty(self):
        self.assertEqual(imgcheck.recent_uploads(Path("/nulle/part")), [])


class DescribeFileTests(unittest.TestCase):
    """Trois fichiers parfaitement servis par le serveur et pourtant cassés
    dans le navigateur : tronqué, mal étiqueté, gigantesque. Le rapport doit
    savoir les distinguer d'un fichier sain."""

    def _png(self, path, size=(100, 80), fmt=None):
        from PIL import Image
        Image.new("RGB", size).save(path, format=fmt)

    def test_a_healthy_file_is_described_not_accused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ok.png"
            self._png(p)
            got = imgcheck.describe_file(p)
        self.assertIn("100×80", got)
        self.assertIn("PNG", got)
        self.assertNotIn("❌", got)

    def test_a_truncated_file_is_called_out(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d) / "ok.png", Path(d) / "coupé.png"
            self._png(good)
            bad.write_bytes(good.read_bytes()[:40])
            got = imgcheck.describe_file(bad)
        self.assertIn("❌", got)
        self.assertIn("truncated", got)

    def test_a_mislabelled_extension_is_called_out(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "faux.png"
            self._png(p, fmt="JPEG")
            got = imgcheck.describe_file(p)
        self.assertIn("⚠️", got)
        self.assertIn("JPEG", got)

    def test_jpg_and_jpeg_are_not_false_positives(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("photo.jpg", "photo.jpeg"):
                p = Path(d) / name
                self._png(p, fmt="JPEG")
                self.assertNotIn("ne correspond pas",
                                 imgcheck.describe_file(p), name)

    def test_a_huge_image_is_flagged_without_being_decoded(self):
        # Pas de vraie image de 200 Mpx en test : on interroge le seuil.
        self.assertGreater(imgcheck.HUGE_PIXELS, 50_000_000)
        self.assertGreater(imgcheck.HUGE_BYTES, 10 * 1024 * 1024)

    def test_a_missing_file_answers_instead_of_raising(self):
        self.assertIn("❌", imgcheck.describe_file(Path("/nulle/part/x.png")))


class SameVolumeTests(unittest.TestCase):
    """La cause avérée de l'icône cassée à l'import : dépôt et cache sur deux
    volumes -> copie en tâche de fond -> réponse tronquée."""

    def test_a_folder_shares_its_own_volume(self):
        self.assertTrue(imgcheck._same_volume(settings.ROOT, settings.ROOT))

    def test_a_missing_path_answers_false(self):
        self.assertFalse(imgcheck._same_volume(Path("/nulle/part"),
                                               settings.ROOT))

    def test_the_check_appears_once_the_upload_folder_exists(self):
        (settings.ROOT / "tmp" / "upload").mkdir(parents=True, exist_ok=True)
        with mock.patch.dict(
                os.environ,
                {"GRADIO_TEMP_DIR": str(settings.ROOT / "tmp" / "gradio")}):
            items, _dest = imgcheck.checks()
        labels = [c.label for c in items]
        self.assertIn("Upload and cache on the same drive", labels)


class DriveKindTests(unittest.TestCase):
    def test_it_stays_silent_outside_windows(self):
        # Aucun type de lecteur à annoncer ailleurs : mieux vaut se taire que
        # d'inventer une ligne qui ne veut rien dire.
        with mock.patch("atelier.imgcheck.sys.platform", "linux"):
            self.assertEqual(imgcheck.drive_kind(Path("/tmp")), "")


if __name__ == "__main__":
    unittest.main()
