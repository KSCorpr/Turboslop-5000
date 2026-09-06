"""Le fichier déposé doit atterrir sur le même disque que le cache.

L'histoire, parce qu'elle explique le test : pour protéger le cache d'images
du ménage de Windows, on l'a déplacé de %TEMP% vers `tmp/gradio`, dans le
projet. Sur une machine dont le projet ne vit pas sur `C:`, départ et arrivée
se sont retrouvés sur DEUX DISQUES — et Gradio 5.50 réagit très mal à ça :

    try:
        os.rename(temp_file.file.name, dest)   # échoue entre deux volumes
    except OSError:
        files_to_copy.append(...)              # copie EN TÂCHE DE FOND
    output_files.append(dest)                  # ... mais on répond maintenant

La réponse part avec le chemin final avant que la copie ait commencé. Le
navigateur demande l'image aussitôt, `FileResponse` annonce la taille du
fichier PARTIEL en `Content-Length`, puis lit davantage à mesure que la copie
avance — h11 coupe la réponse (« Too much data for declared Content-Length »)
et le navigateur affiche une icône cassée, alors que le fichier finit bien par
être complet et correctement utilisé par l'outil.

Mesuré : sur une image de 11 Mo, le dépôt renvoie un chemin vers un fichier de
0 Ko, et la requête suivante annonce 65 536 octets pour en livrer 65 536 sur
11 234 505. Une fois le fichier temporaire posé sur le même volume, la même
mesure donne 11 234 505 sur 11 234 505.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Sous-process : importer app.py pose des variables d'environnement et
# rapièce Gradio pour tout l'interpréteur. C'est justement ce qu'on teste, mais
# ce n'est pas quelque chose à faire subir aux autres tests.
_PROBE = r"""
import os, sys
sys.path.insert(0, %(root)r)
import app                      # pose GRADIO_TEMP_DIR puis applique les patchs
import gradio.route_utils as ru
from pathlib import Path

cache = Path(os.environ["GRADIO_TEMP_DIR"])
cache.mkdir(parents=True, exist_ok=True)
f = ru.NamedTemporaryFile(delete=False)
tmp = Path(f.name)
f.close()
try:
    # Le seul critère qui compte : os.rename doit réussir.
    dest = cache / "test_rename_probe.bin"
    os.rename(tmp, dest)
    renamed = dest.is_file()
    dest.unlink()
except OSError as exc:
    renamed = False
    print("RENAME_ERROR", exc)
print("TMP", tmp)
print("CACHE", cache)
print("SAME_DEV", tmp.parent.stat().st_dev == cache.stat().st_dev)
print("SAME_DRIVE", os.path.splitdrive(tmp)[0] == os.path.splitdrive(cache)[0])
print("RENAMED", renamed)
print("UNDER_PROJECT", str(tmp).startswith(%(root)r))
""" % {"root": str(ROOT)}


class UploadLandsNextToTheCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise AssertionError(proc.stderr[-2000:])
        cls.out = dict(
            line.split(" ", 1) for line in proc.stdout.strip().splitlines()
            if " " in line)

    def test_rename_succeeds(self):
        """Si `os.rename` réussit, il n'y a pas de copie différée, donc pas de
        course, donc pas de réponse tronquée. Tout le correctif est là."""
        self.assertEqual(self.out.get("RENAMED"), "True",
                         self.out.get("RENAME_ERROR", ""))

    def test_the_upload_temp_file_is_on_the_same_volume(self):
        self.assertEqual(self.out.get("SAME_DEV"), "True", self.out)
        self.assertEqual(self.out.get("SAME_DRIVE"), "True", self.out)

    def test_it_lives_under_the_project(self):
        """Donc visible et nettoyable depuis « Gestion & nettoyage », comme le
        reste de `tmp/` — un dossier de travail invisible finit par grossir
        sans que personne ne sache d'où il vient."""
        self.assertEqual(self.out.get("UNDER_PROJECT"), "True", self.out)


class PatchIsWiredTests(unittest.TestCase):
    """Le correctif est un rapiéçage d'une fonction d'une bibliothèque : il
    peut cesser de mordre sans rien casser de visible."""

    def test_app_applies_it_at_startup(self):
        import ast
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("_upload_on_the_same_volume", called)

    def test_gradio_still_creates_its_upload_file_the_expected_way(self):
        """Si l'amont change d'appel, le rapiéçage devient un décor. Autant
        que le test le dise, plutôt que l'utilisateur."""
        import gradio.route_utils as ru
        src = Path(ru.__file__).read_text(encoding="utf-8")
        self.assertIn("NamedTemporaryFile(delete=False)", src,
                      "Gradio a changé la création du fichier de dépôt : "
                      "vérifier _upload_on_the_same_volume() dans app.py")

    def test_gradio_still_falls_back_to_a_background_copy(self):
        """Le comportement amont qui rend le correctif nécessaire.

        Cherché dans les DEUX modules : Gradio 6 a déplacé ce code de
        `routes.py` vers `route_utils.py` sans rien changer au comportement.
        Vérifier le fichier plutôt que le comportement faisait échouer le test
        sur un simple déménagement — un faux blocage à la mise à jour.
        """
        import gradio.route_utils as ru
        import gradio.routes as gr_routes
        src = "\n".join(Path(m.__file__).read_text(encoding="utf-8")
                        for m in (gr_routes, ru))
        self.assertIn("os.rename(temp_file.file.name, dest)", src)
        self.assertIn("files_to_copy.append", src)
        self.assertIn("move_uploaded_files_to_cache", src)


if __name__ == "__main__":
    unittest.main()
