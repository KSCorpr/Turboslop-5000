"""Les sous-process savent-ils importer le code de l'application ?

Ce test existe à cause d'un plantage précis : `run_layers.py` importait
`atelier.engine.masks` alors qu'un sous-process lancé par
`python scripts/tools/run_layers.py` ne reçoit dans `sys.path` que le dossier
du SCRIPT, jamais la racine du dépôt. Résultat, `ModuleNotFoundError` — et pas
tout de suite : après la segmentation SAM, donc après plusieurs minutes
d'attente, sur une seule ligne de traceback.

Le test est DYNAMIQUE (on charge réellement chaque runner dans un sous-process
propre, depuis un répertoire courant neutre) parce qu'une vérification
statique du texte passerait à côté du seul cas qui compte : celui où le chemin
existe mais désigne le mauvais dossier.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "scripts" / "tools"

# Charge le runner par son CHEMIN (donc en exécutant son en-tête, sys.path
# compris), puis tente les imports qu'il fera plus tard.
_PROBE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("runner_under_test", {path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
{statement}
print("OK")
"""


def _runners_importing_atelier() -> list[tuple[Path, list[str]]]:
    """(runner, instructions d'import `atelier.*` qu'il contient).

    On rejoue l'instruction TELLE QU'ÉCRITE plutôt que de reconstruire un nom
    de module : `from atelier.engine.masks import touches` importe une
    FONCTION, pas un sous-module, et la transformer en `import
    atelier.engine.masks.touches` produit une erreur qui n'existe pas.
    """
    import re
    out = []
    for path in sorted(TOOLS.glob("run_*.py")):
        src = path.read_text(encoding="utf-8")
        stmts = set()
        for m in re.finditer(r"from (atelier[\w.]*) import ([\w, ]+)", src):
            names = ", ".join(n.strip() for n in m.group(2).split(","))
            stmts.add(f"from {m.group(1)} import {names}")
        for m in re.finditer(r"^\s*import (atelier[\w.]*)", src, re.M):
            stmts.add(f"import {m.group(1)}")
        if stmts:
            out.append((path, sorted(stmts)))
    return out


class RunnerBootstrapTests(unittest.TestCase):
    def test_at_least_one_runner_is_covered(self):
        """Si plus aucun runner n'importe `atelier`, ce test ne teste rien —
        autant le savoir plutôt que de le voir passer en silence."""
        self.assertTrue(_runners_importing_atelier())

    def test_every_runner_can_import_what_it_uses(self):
        failures = []
        with tempfile.TemporaryDirectory() as neutral:
            for path, stmts in _runners_importing_atelier():
                for statement in stmts:
                    code = _PROBE.format(path=str(path), statement=statement)
                    # cwd neutre + PYTHONPATH vide : exactement les conditions
                    # d'un sous-process lancé par l'application.
                    proc = subprocess.run(
                        [sys.executable, "-c", code], cwd=neutral,
                        capture_output=True, text=True,
                        env={"PATH": "/usr/bin:/bin", "SYSTEMROOT": ""})
                    if proc.returncode != 0:
                        failures.append(
                            f"{path.name} : « {statement} » échoue —\n"
                            f"{proc.stderr.strip().splitlines()[-1]}")
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
