"""Mise à jour de l'application : ce qu'elle écrit, et surtout ce qu'elle ne
touche pas.

Un updater qui se trompe ne coûte pas un bug : il coûte des modèles de plusieurs
gigaoctets, des images générées, des réglages. Tout ce qui suit vérifie donc
d'abord les refus — dossiers de données, archive douteuse, chemin qui remonte —
avant de vérifier qu'il fait bien son travail.
"""
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "update_app.py"
_SPEC = importlib.util.spec_from_file_location("update_app_test", _PATH)
U = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(U)


def archive(files: dict) -> bytes:
    """Une archive GitHub : tout est sous un dossier racine unique."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(f"Turboslop-5000-main/{name}", content)
    return buf.getvalue()


SANE = {"app.py": "print('v2')\n",
        "atelier/__init__.py": "",
        "atelier/ui/tab.py": "X = 2\n"}


class ArchiveTests(unittest.TestCase):
    def test_a_normal_archive_is_read_without_its_root_folder(self):
        files = U._archive_files(archive(SANE))
        self.assertEqual(set(files), set(SANE))

    def test_an_html_page_is_refused(self):
        # Un proxy d'entreprise répond une page HTML : l'écrire par-dessus
        # l'installation serait bien pire que de ne rien faire.
        with self.assertRaisesRegex(RuntimeError, "ZIP"):
            U._archive_files(b"<html>Access denied</html>")

    def test_an_archive_without_the_application_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "app.py absent"):
            U._archive_files(archive({"LISEZMOI.txt": "rien"}))

    def test_a_path_that_climbs_out_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "dangereux"):
            U._archive_files(archive({**SANE, "../evade.py": "bad"}))

    def test_user_data_inside_the_archive_is_ignored(self):
        files = U._archive_files(archive({**SANE, "models/gros.gguf": "X" * 10,
                                          "userdata/prefs.json": "{}",
                                          ".github/workflows/ci.yml": "on: push"}))
        self.assertEqual(set(files), set(SANE))


class PlanTests(unittest.TestCase):
    def _root(self, tmp, local: dict):
        root = Path(tmp)
        for rel, content in local.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            # The updater compares bytes. Match the archive's LF bytes on
            # Windows too; text mode would turn them into CRLF and manufacture
            # a difference in the fixture marked as identical.
            p.write_bytes(content.encode("utf-8"))
        return root

    def test_only_real_differences_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, {"app.py": "print('v2')\n",   # identique
                                    "atelier/ui/tab.py": "X = 1\n"})  # différent
            with patch.object(U, "ROOT", root):
                added, updated, removed = U._plan(
                    {k: v.encode() for k, v in SANE.items()}, {})
        self.assertEqual(added, ["atelier/__init__.py"])
        self.assertEqual(updated, ["atelier/ui/tab.py"])
        self.assertEqual(removed, [])

    def test_only_files_we_installed_ourselves_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, {"atelier/vieux.py": "reste d'avant\n",
                                    "atelier/a_moi.py": "écrit par l'utilisateur\n"})
            manifest = {"files": ["atelier/vieux.py"]}
            with patch.object(U, "ROOT", root):
                _, _, removed = U._plan({k: v.encode() for k, v in SANE.items()},
                                        manifest)
        # Le fichier connu du manifeste part ; celui qu'on n'a jamais posé reste.
        self.assertEqual(removed, ["atelier/vieux.py"])


class UpdateTests(unittest.TestCase):
    """Le cycle complet, sur un dossier jetable et sans réseau."""

    def _run(self, tmp, blob, local=None, manifest=None, **kwargs):
        root = Path(tmp)
        for rel, content in (local or {}).items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        man = root / "userdata" / "app-update.json"
        if manifest is not None:
            man.parent.mkdir(parents=True, exist_ok=True)
            man.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(U, "ROOT", root), \
             patch.object(U, "MANIFEST", man), \
             patch.object(U, "BACKUP_DIR", root / ".update-backup"), \
             patch.object(U, "_fetch", return_value=blob), \
             patch.object(U, "_latest_commit", return_value={}):
            code = U.update(**kwargs)
        return code, root

    def test_it_writes_the_new_code_and_records_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, root = self._run(tmp, archive(SANE),
                                   local={"app.py": "print('v1')\n"})
            self.assertEqual(code, 0)
            self.assertEqual((root / "app.py").read_text(), "print('v2')\n")
            self.assertEqual((root / "atelier" / "ui" / "tab.py").read_text(),
                             "X = 2\n")
            manifest = json.loads((root / "userdata" / "app-update.json")
                                  .read_text(encoding="utf-8"))
            self.assertIn("app.py", manifest["files"])

    def test_check_only_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, root = self._run(tmp, archive(SANE),
                                   local={"app.py": "print('v1')\n"},
                                   check_only=True)
            self.assertEqual(code, 0)
            self.assertEqual((root / "app.py").read_text(), "print('v1')\n")
            self.assertFalse((root / "atelier").exists())

    def test_user_data_survives_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {"models/krea2.gguf": "des gigaoctets",
                    "outputs/image.png": "ma création",
                    "loras/style.safetensors": "à moi",
                    "userdata/prefs.json": '{"lang": "fr"}'}
            code, root = self._run(tmp, archive(SANE),
                                   local={"app.py": "print('v1')\n", **data})
            self.assertEqual(code, 0)
            for rel, content in data.items():
                self.assertEqual((root / rel).read_text(), content, rel)

    def test_code_that_does_not_compile_is_rolled_back(self):
        broken = {**SANE, "atelier/ui/tab.py": "def cassé(:\n"}
        with tempfile.TemporaryDirectory() as tmp:
            code, root = self._run(tmp, archive(broken),
                                   local={"app.py": "print('v1')\n",
                                          "atelier/ui/tab.py": "X = 1\n"})
            self.assertEqual(code, 1)
            # Tout est revenu : le fichier modifié comme le fichier ajouté.
            self.assertEqual((root / "app.py").read_text(), "print('v1')\n")
            self.assertEqual((root / "atelier" / "ui" / "tab.py").read_text(),
                             "X = 1\n")
            self.assertFalse((root / "atelier" / "__init__.py").exists())

    def test_a_second_run_finds_nothing_to_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, root = self._run(tmp, archive(SANE),
                                   local={"app.py": "print('v1')\n"})
            self.assertEqual(code, 0)
            man = json.loads((root / "userdata" / "app-update.json").read_text())
            with patch.object(U, "ROOT", root), \
                 patch.object(U, "MANIFEST", root / "userdata" / "app-update.json"), \
                 patch.object(U, "BACKUP_DIR", root / ".update-backup"), \
                 patch.object(U, "_fetch", return_value=archive(SANE)), \
                 patch.object(U, "_latest_commit", return_value={}):
                added, updated, removed = U._plan(
                    {k: v.encode() for k, v in SANE.items()}, man)
        self.assertEqual((added, updated, removed), ([], [], []))

    def test_rollback_restores_the_previous_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, root = self._run(tmp, archive(SANE),
                                   local={"app.py": "print('v1')\n"})
            self.assertEqual(code, 0)
            with patch.object(U, "ROOT", root), \
                 patch.object(U, "BACKUP_DIR", root / ".update-backup"):
                self.assertEqual(U._rollback(), 0)
            self.assertEqual((root / "app.py").read_text(), "print('v1')\n")
            self.assertFalse((root / "atelier" / "ui" / "tab.py").exists())


class MissingFilesTests(unittest.TestCase):
    def test_it_names_the_file_that_disappeared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            man = root / "userdata" / "app-update.json"
            man.parent.mkdir(parents=True)
            man.write_text(json.dumps(
                {"files": ["app.py", "atelier/engine/sdserver.py"]}),
                encoding="utf-8")
            (root / "app.py").write_text("ok", encoding="utf-8")
            with patch.object(U, "ROOT", root), patch.object(U, "MANIFEST", man):
                self.assertEqual(U.missing_files(),
                                 ["atelier/engine/sdserver.py"])


if __name__ == "__main__":
    unittest.main()
