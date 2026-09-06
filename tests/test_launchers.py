"""Les lanceurs `.bat` / `.sh` — le seul point d'entrée pour qui n'a pas Python.

Ce fichier existe parce que j'ai livré une sonde utilisable uniquement par
`python scripts/...`, sur un projet dont l'utilisateur n'a pas Python installé :
il y a un interpréteur PORTABLE dans `python\\`, et tout passe par un `.bat`.
Un script sans lanceur est un script inexistant.

Deux défauts se voient au double-clic et nulle part ailleurs : un `.bat` qui
appelle un fichier qui n'existe plus, et un `.bat` qui invoque `python` tout
court — ce qui marche sur la machine du développeur et échoue chez tout le
monde.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# L'exigence n'est pas une orthographe précise mais un FAIT : le lanceur doit
# viser un python.exe DU PROJET. run.bat l'écrit « %~dp0python\python.exe »,
# install.bat « %PYDIR%\python.exe » — il installe cet interpréteur, il ne peut
# pas encore s'y référer autrement. Les deux sont corrects.
_PORTABLE = re.compile(r'set\s+"PY=[^"]*python\.exe"', re.I)
# Script Python appelé par un lanceur : « scripts\x.py », « app.py »…
_CALLS = re.compile(r"([\w./\\-]+\.py)")


def _launchers(suffix: str) -> list[Path]:
    return sorted(p for p in ROOT.glob(f"*{suffix}"))


class LaunchersExistTests(unittest.TestCase):
    def test_there_are_launchers_to_check(self):
        self.assertTrue(_launchers(".bat"))
        self.assertTrue(_launchers(".sh"))

    def test_every_launcher_calls_a_script_that_exists(self):
        missing = []
        for path in _launchers(".bat") + _launchers(".sh"):
            for line in path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                # Une URL ou un fichier de %TEMP% n'est pas un script du dépôt :
                # install.bat télécharge get-pip.py, ce n'est pas un oubli.
                if "http" in line.lower() or "%TEMP%" in line.upper():
                    continue
                for call in set(_CALLS.findall(line)):
                    target = ROOT / call.replace("\\", "/")
                    if not target.is_file():
                        missing.append(f"{path.name} -> {call}")
        self.assertEqual(missing, [], "\n".join(missing))


class PortablePythonTests(unittest.TestCase):
    """Un `.bat` qui écrit juste `python` marche chez le développeur et
    échoue chez l'utilisateur, qui n'a que l'interpréteur du dossier."""

    def test_every_bat_prefers_the_bundled_interpreter(self):
        wrong = []
        for path in _launchers(".bat"):
            src = path.read_text(encoding="utf-8", errors="replace")
            if ".py" not in src:
                continue            # lanceur qui n'exécute pas de Python
            if not _PORTABLE.search(src):
                wrong.append(path.name)
        self.assertEqual(wrong, [], f"sans Python portable : {wrong}")

    def test_every_bat_runs_from_its_own_folder(self):
        """Sans `cd /d %~dp0`, un double-clic depuis un raccourci démarre
        ailleurs et ne trouve plus rien."""
        wrong = [p.name for p in _launchers(".bat")
                 if "cd /d" not in p.read_text(encoding="utf-8",
                                               errors="replace").lower()]
        self.assertEqual(wrong, [], f"sans cd : {wrong}")


class EveryEntryPointIsReachableTests(unittest.TestCase):
    """Un script destiné à l'utilisateur DOIT avoir son lanceur."""

    # Scripts internes, appelés par le code et jamais à la main.
    INTERNAL = {"get_sdcpp.py", "get_trellis.py", "setup_tools.py",
                "setup_seedvr2.py",   # lancé par le bouton « Installer »
                "_torch_setup.py", "convert_gguf.py"}

    def test_user_facing_scripts_have_a_launcher(self):
        launched = set()
        for path in _launchers(".bat") + _launchers(".sh"):
            launched |= {Path(c).name
                         for c in _CALLS.findall(
                             path.read_text(encoding="utf-8",
                                            errors="replace"))}
        orphans = [p.name for p in sorted((ROOT / "scripts").glob("*.py"))
                   if p.name not in launched and p.name not in self.INTERNAL]
        self.assertEqual(orphans, [],
                         f"scripts sans lanceur : {orphans} — ajoutez un .bat "
                         "ou classez-les dans INTERNAL")


if __name__ == "__main__":
    unittest.main()
