"""Restauration de visages : la commande envoyée au sous-process.

Le runner lui-même ne peut pas tourner ici (il lui faut torch, facexlib et
577 Mo de poids). Ce qu'on garde sous contrôle, c'est tout ce qui se décide
AVANT le sous-process : ce qui compte comme « installé », les bornes du
réglage de fidélité, et le fait que l'option « visage principal » ne parte que
si elle a été demandée — trois choses qui, si elles se trompent, produisent
soit un plantage au milieu du traitement, soit un résultat silencieusement
différent de ce qui a été coché.
"""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier.engine import tools

ROOT = Path(__file__).resolve().parent.parent


class FaceInstallTests(unittest.TestCase):
    def test_needs_the_shared_parts_and_at_least_one_restorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            with patch.object(tools, "FACE_MODEL_DIR", model_dir):
                # Détecteur et segmentation seuls : rien ne peut tourner.
                for name in tools.FACE_SHARED_FILES:
                    self.assertFalse(tools.face_is_installed())
                    (model_dir / name).write_bytes(b"x")
                self.assertFalse(tools.face_is_installed(),
                                 "installé sans aucun restaurateur")
                # Un seul restaurateur suffit : une installation interrompue
                # après le premier modèle reste utilisable.
                (model_dir / tools.FACE_DEFAULT).write_bytes(b"x")
                self.assertTrue(tools.face_is_installed())

    def test_a_restorer_without_the_shared_parts_is_not_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / tools.FACE_DEFAULT).write_bytes(b"x")
            with patch.object(tools, "FACE_MODEL_DIR", model_dir):
                self.assertFalse(tools.face_is_installed())

    def test_it_lists_only_the_models_actually_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "codeformer.pth").write_bytes(b"x")
            with patch.object(tools, "FACE_MODEL_DIR", model_dir):
                present = tools.face_models_installed()
        self.assertEqual([f for f, _, _ in present], ["codeformer.pth"])

    def test_every_model_declares_its_licence(self):
        # La licence décide de ce qu'on a le droit de faire du résultat : elle
        # est affichée AVEC le modèle, pas dans une note de bas de page.
        for filename, label, licence in tools.FACE_MODELS:
            self.assertTrue(label and licence, filename)
        by_file = {f: lic for f, _, lic in tools.FACE_MODELS}
        self.assertIn("NON COMMERCIAL", by_file["codeformer.pth"])
        self.assertIn("Apache-2.0", by_file["GFPGANv1.4.pth"])
        self.assertIn("Apache-2.0", by_file["RestoreFormer++.ckpt"])

    def test_the_default_is_free_of_commercial_restriction(self):
        licence = dict((f, lic) for f, _, lic in tools.FACE_MODELS)[
            tools.FACE_DEFAULT]
        self.assertNotIn("NON COMMERCIAL", licence)

    def test_refuses_to_run_when_not_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tools, "FACE_MODEL_DIR", Path(tmp)):
                with self.assertRaises(tools.ToolError):
                    tools.face_restore(Image.new("RGB", (8, 8)))


class FaceCommandTests(unittest.TestCase):
    def _run(self, **kwargs):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model_dir = base / "model"
            model_dir.mkdir()
            for name in tools.FACE_SHARED_FILES:
                (model_dir / name).write_bytes(b"x")
            for filename, _, _ in tools.FACE_MODELS:
                (model_dir / filename).write_bytes(b"x")

            def fake_run(cmd, log, err_msg, gpu_index=None, cwd=None, env=None):
                captured["cmd"] = cmd
                out_dir = Path(cmd[cmd.index("--output-dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), "navy").save(out_dir / "a_face.png")

            with (
                patch.object(tools, "FACE_MODEL_DIR", model_dir),
                patch.object(tools.settings, "TMP_DIR", base / "tmp"),
                patch.object(tools.settings, "OUTPUT_DIR", base / "out"),
                patch.object(tools, "_gen_gpu_index", return_value=0),
                patch.object(tools, "_run_tool", side_effect=fake_run),
            ):
                (base / "out").mkdir(parents=True, exist_ok=True)
                out = tools.face_restore(Image.new("RGB", (8, 8)), **kwargs)
        return captured["cmd"], out

    def test_fidelity_is_passed_and_clamped(self):
        for asked, sent in ((0.5, "0.50"), (0.0, "0.00"), (1.0, "1.00"),
                            (-3.0, "0.00"), (42.0, "1.00")):
            cmd, _ = self._run(fidelity=asked)
            self.assertEqual(cmd[cmd.index("--fidelity") + 1], sent)

    def test_center_flag_only_when_asked(self):
        cmd, _ = self._run(only_center=True)
        self.assertIn("--only-center", cmd)
        cmd, _ = self._run(only_center=False)
        self.assertNotIn("--only-center", cmd)

    def test_the_chosen_model_is_the_one_sent(self):
        for filename, _, _ in tools.FACE_MODELS:
            cmd, _ = self._run(model=filename)
            self.assertEqual(cmd[cmd.index("--weights") + 1], filename)

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(tools.ToolError):
            self._run(model="../../etc/passwd.pth")

    def test_result_lands_in_outputs(self):
        _, out = self._run()
        self.assertTrue(out.name.startswith("face-"))
        self.assertEqual(out.suffix, ".png")


class FaceRunnerTests(unittest.TestCase):
    """Le runner ne peut pas s'exécuter ici, mais il doit rester valide."""

    def test_runner_parses_and_declares_its_options(self):
        source = (ROOT / "scripts" / "tools" / "run_face.py").read_text(
            encoding="utf-8")
        ast.parse(source)   # syntaxe
        for option in ("--model-dir", "--input", "--output-dir", "--fidelity",
                       "--only-center"):
            self.assertIn(option, source)

    def test_the_runner_knows_the_three_architectures(self):
        source = (ROOT / "scripts" / "tools" / "run_face.py").read_text(
            encoding="utf-8")
        for arch in ("CodeFormer", "GFPGAN", "RestoreFormer"):
            self.assertIn(arch, source)

    def test_only_codeformer_gets_the_fidelity_weight(self):
        # Passer « weight= » à GFPGAN lèverait un TypeError en plein
        # traitement ; spandrel connaît la convention de chaque architecture.
        source = (ROOT / "scripts" / "tools" / "run_face.py").read_text(
            encoding="utf-8")
        head = source[:source.index("weight=fidelity")]
        self.assertIn('arch == "CodeFormer"', head)

    def test_runner_does_not_depend_on_basicsr(self):
        # basicsr importe torchvision.transforms.functional_tensor, supprimé
        # depuis torchvision 0.17 : s'il revenait, l'outil ne s'installerait
        # plus du tout sur notre socle.
        source = (ROOT / "scripts" / "tools" / "run_face.py").read_text(
            encoding="utf-8")
        self.assertNotIn("import basicsr", source)
        self.assertNotIn("from basicsr", source)


class AddonRegistriesTests(unittest.TestCase):
    """Un add-on installé doit être connu des deux registres.

    Il y en a deux, et ils servaient à des choses opposées : l'inventaire dit
    ce que l'utilisateur peut voir et libérer, la maintenance dit ce qui est
    LÉGITIME dans tools_repo/. Les deux étaient recopiés à la main et avaient
    pris du retard — `describe` (7,5 Go) n'apparaissait dans aucun des deux,
    donc `maintenance --purge` le proposait à la suppression comme un dossier
    inconnu. Ce test empêche le prochain add-on de repartir dans le même trou.
    """

    def _addon_dirs(self):
        return {name: value for name, value in vars(tools).items()
                if name.endswith("_DIR") and isinstance(value, Path)
                and value != tools.TOOLS_DIR
                and tools.TOOLS_DIR in value.parents}

    def test_maintenance_knows_every_addon_folder(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "maintenance_under_test", ROOT / "scripts" / "maintenance.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        known = module._known_addon_dirs()
        for name, path in self._addon_dirs().items():
            folder = path.relative_to(tools.TOOLS_DIR).parts[0]
            self.assertIn(folder, known, f"{name} inconnu de la maintenance")

    def test_inventory_lists_every_addon_folder(self):
        from atelier import inventory
        listed = {p.resolve() for item in inventory.items() for p in item.paths}
        for name, path in self._addon_dirs().items():
            covered = path.resolve() in listed or any(
                parent in listed for parent in path.resolve().parents)
            self.assertTrue(covered, f"{name} absent de l'inventaire")


if __name__ == "__main__":
    unittest.main()
