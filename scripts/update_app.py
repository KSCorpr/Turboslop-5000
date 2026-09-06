#!/usr/bin/env python3
"""Met à jour l'APPLICATION (le code) depuis GitHub, sans rien retélécharger
à la main et sans jamais toucher à vos données.

Jusqu'ici, mettre à jour voulait dire : retélécharger l'archive complète du
dépôt, la dézipper par-dessus le dossier, et espérer. Deux défauts, tous deux
rencontrés en vrai :
  • une extraction par-dessus n'EFFACE jamais ce qui a été retiré en amont, et
    laisse traîner du code mort qui finit par s'exécuter ;
  • une extraction partielle (application ouverte, antivirus, copie coupée)
    laisse un dossier où un fichier manque, et l'application ne démarre plus
    sans qu'on sache lequel.

Ce script fait les deux correctement :
  • il télécharge l'archive du dépôt, la vérifie AVANT de toucher à quoi que
    ce soit, et n'écrit que les fichiers réellement différents ;
  • il tient un MANIFESTE de ce qu'il a posé, donc il sait supprimer ce qui a
    disparu du projet — et lui seul ;
  • il sauvegarde tout fichier qu'il remplace, et `--rollback` revient en
    arrière ;
  • si le code posé ne compile pas, il revient en arrière tout seul.

Ce qu'il ne touche JAMAIS : models/, loras/, outputs/, userdata/, tools_repo/,
bin/, python/, tmp/ — vos modèles, vos images, vos réglages, vos moteurs.

    update.bat              met à jour
    update.bat --check      dit seulement ce qui changerait
    update.bat --rollback   annule la dernière mise à jour
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
REPO = "KSCorpr/Turboslop-5000"
BRANCH = "main"
ARCHIVE = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"
COMMITS = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"

MANIFEST = ROOT / "userdata" / "app-update.json"
BACKUP_DIR = ROOT / ".update-backup"

# Ce qui n'appartient PAS au code : jamais écrit, jamais supprimé, jamais lu.
# Un dossier de cette liste est intouchable même si l'archive en contient un du
# même nom (elle ne devrait pas — c'est une ceinture en plus de la bretelle).
PROTECTED_TOP = {
    "models", "loras", "outputs", "userdata", "tools_repo", "bin", "python",
    "tmp", ".git", ".update-backup", "venv", ".venv",
}
# Fichiers d'un dépôt qui n'ont rien à faire dans une installation.
SKIP_NAMES = {".gitignore", ".gitattributes"}
SKIP_PREFIX = (".github/",)

OK, WARN, ERR, INFO = "  [OK] ", "  [!] ", "  [X] ", "  [i] "
_UA = {"User-Agent": "turbo-slop-updater"}


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_protected(rel: str) -> bool:
    top = rel.split("/", 1)[0]
    return (top in PROTECTED_TOP or rel in SKIP_NAMES
            or rel.startswith(SKIP_PREFIX))


# --------------------------------------------------------------------------- #
#  Téléchargement
# --------------------------------------------------------------------------- #
def _fetch(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _latest_commit() -> dict:
    """Dernier commit de la branche : sha, date, titre. {} si indisponible.

    Purement informatif — la mise à jour ne dépend PAS de l'API GitHub, qui
    limite les requêtes anonymes. C'est le contenu de l'archive qui fait foi.
    """
    try:
        data = json.loads(_fetch(COMMITS, timeout=30).decode("utf-8"))
        return {
            "sha": data.get("sha", "")[:12],
            "date": (data.get("commit", {}).get("author", {}) or {}).get("date", ""),
            "title": (data.get("commit", {}).get("message", "")
                      or "").splitlines()[0][:100],
        }
    except Exception:  # noqa: BLE001
        return {}


def _archive_files(blob: bytes) -> dict[str, bytes]:
    """Contenu de l'archive, chemins relatifs à la racine du projet.

    Refuse une archive qui ne ressemble pas au projet : un proxy d'entreprise
    qui répond une page HTML produit un « zip » parfaitement lisible et
    parfaitement faux, et l'écrire par-dessus l'installation serait pire que
    ne rien faire.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        raise RuntimeError(
            "le téléchargement n'est pas une archive ZIP (proxy ou "
            "connexion filtrée ?)")
    out: dict[str, bytes] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        rel = parts[1]
        # Zip-slip : un chemin qui remonte n'a rien à faire ici.
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise RuntimeError(f"chemin dangereux dans l'archive : {rel}")
        if _is_protected(rel):
            continue
        out[rel] = archive.read(info)
    if "app.py" not in out or not any(f.startswith("atelier/") for f in out):
        raise RuntimeError(
            "l'archive ne contient pas l'application (app.py absent) — "
            "téléchargement incomplet ou redirigé")
    return out


# --------------------------------------------------------------------------- #
#  Comparaison
# --------------------------------------------------------------------------- #
def _plan(files: dict[str, bytes], manifest: dict) -> tuple[list, list, list]:
    """(à ajouter, à mettre à jour, à supprimer)."""
    added, updated = [], []
    for rel, data in sorted(files.items()):
        local = ROOT / rel
        if not local.is_file():
            added.append(rel)
        else:
            try:
                if local.read_bytes() != data:
                    updated.append(rel)
            except OSError:
                updated.append(rel)
    # On ne supprime QUE ce qu'on a soi-même posé lors d'une mise à jour
    # précédente. Un fichier qu'on n'a jamais écrit n'est pas à nous : il peut
    # venir de l'utilisateur, et le supprimer serait une perte de données.
    known = set(manifest.get("files") or [])
    removed = sorted(rel for rel in known
                     if rel not in files and (ROOT / rel).is_file()
                     and not _is_protected(rel))
    return added, updated, removed


# --------------------------------------------------------------------------- #
#  Écriture
# --------------------------------------------------------------------------- #
def _backup(rel: str, stamp_dir: Path) -> None:
    src = ROOT / rel
    if not src.is_file():
        return
    dest = stamp_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _write(rel: str, data: bytes) -> None:
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Écriture par fichier temporaire puis remplacement : une coupure au
    # mauvais moment laisse l'ancien fichier intact, jamais un fichier à
    # moitié écrit. C'est précisément le défaut qu'on cherche à éliminer.
    tmp = dest.with_name(dest.name + ".new")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def _purge_pycache() -> int:
    """Les .pyc d'un module supprimé restent importables : on les enlève."""
    n = 0
    for cache in ROOT.rglob("__pycache__"):
        if _is_protected(str(cache.relative_to(ROOT)).replace("\\", "/")):
            continue
        shutil.rmtree(cache, ignore_errors=True)
        n += 1
    return n


def _compiles() -> str:
    """"" si tout le code Python compile, sinon le motif de l'échec.

    `quiet=1` et pas 2 : on veut le message d'erreur, c'est lui qu'on affiche.
    Et si la vérification échoue sans rien écrire, on renvoie quand même une
    phrase — une chaîne vide voudrait dire « tout va bien », ce qui est
    exactement l'inverse.
    """
    import compileall
    import contextlib
    import py_compile
    buf = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        for folder in ("atelier", "scripts"):
            if (ROOT / folder).is_dir():
                ok = compileall.compile_dir(
                    str(ROOT / folder), quiet=1, force=True) and ok
        app = ROOT / "app.py"
        if app.is_file():
            try:
                py_compile.compile(str(app), doraise=True)
            except py_compile.PyCompileError as exc:
                ok = False
                print(exc)
    if ok:
        return ""
    detail = [x for x in buf.getvalue().strip().splitlines() if x.strip()]
    return detail[-1] if detail else "erreur de syntaxe dans le code téléchargé"


def _restore(stamp_dir: Path, written: list[str], removed: list[str]) -> None:
    """Remet l'installation dans l'état d'avant la mise à jour."""
    for rel in written:
        saved = stamp_dir / rel
        if saved.is_file():
            _write(rel, saved.read_bytes())
        else:
            # Fichier AJOUTÉ par la mise à jour : il n'existait pas avant.
            (ROOT / rel).unlink(missing_ok=True)
    for rel in removed:
        saved = stamp_dir / rel
        if saved.is_file():
            _write(rel, saved.read_bytes())


# --------------------------------------------------------------------------- #
#  Manifeste
# --------------------------------------------------------------------------- #
def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(files: dict[str, bytes], commit: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "schema": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "files": sorted(files),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def missing_files() -> list[str]:
    """Fichiers que la dernière mise à jour a posés et qui ont disparu.

    C'est le diagnostic qui manquait quand un module s'était volatilisé :
    l'application ne démarrait plus et rien ne disait quel fichier chercher.
    """
    manifest = _load_manifest()
    return [rel for rel in (manifest.get("files") or [])
            if not (ROOT / rel).exists()]


# --------------------------------------------------------------------------- #
#  Commandes
# --------------------------------------------------------------------------- #
def _rollback() -> int:
    if not BACKUP_DIR.is_dir():
        _say(ERR + "aucune sauvegarde : rien à annuler.")
        return 1
    saves = sorted(p for p in BACKUP_DIR.iterdir() if p.is_dir())
    if not saves:
        _say(ERR + "aucune sauvegarde : rien à annuler.")
        return 1
    last = saves[-1]
    state = {}
    try:
        state = json.loads((last / "_update.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    _say(f"Annulation de la mise à jour du {last.name}…")
    _restore(last, state.get("written") or [], state.get("removed") or [])
    _purge_pycache()
    shutil.rmtree(last, ignore_errors=True)
    _say(OK + "installation revenue à son état précédent.")
    return 0


def update(check_only: bool = False) -> int:
    _say("=" * 60)
    _say("  Mise à jour de Turboslop 5000")
    _say("=" * 60)

    commit = _latest_commit()
    if commit:
        _say(f"{INFO}dernier commit : {commit['sha']} — {commit['title']}")

    _say("Téléchargement du code depuis GitHub…")
    try:
        blob = _fetch(ARCHIVE)
        files = _archive_files(blob)
    except Exception as exc:  # noqa: BLE001
        _say(ERR + f"téléchargement impossible : {exc}")
        _say("    Sur un réseau d'entreprise, définissez HTTPS_PROXY avant de")
        _say("    lancer update.bat, ou récupérez l'archive à la main :")
        _say(f"    https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip")
        return 1
    _say(OK + f"archive lue : {len(files)} fichiers, "
         f"empreinte {_sha(blob)[:12]}.")

    manifest = _load_manifest()
    added, updated, removed = _plan(files, manifest)
    absent = missing_files()

    if not (added or updated or removed):
        _say(OK + "déjà à jour — aucun fichier ne change.")
        _save_manifest(files, commit or manifest.get("commit") or {})
        return 0

    _say("")
    for label, group in (("ajouté", added), ("mis à jour", updated),
                         ("supprimé", removed)):
        if group:
            _say(f"  {len(group)} fichier(s) {label} :")
            for rel in group[:12]:
                _say(f"    - {rel}")
            if len(group) > 12:
                _say(f"    … et {len(group) - 12} autre(s)")
    if absent:
        _say(f"{WARN}{len(absent)} fichier(s) de la version installée avaient "
             "disparu — ils sont remis.")

    if check_only:
        _say("")
        _say(INFO + "mode --check : rien n'a été écrit.")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    stamp_dir = BACKUP_DIR / stamp
    stamp_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    _say("")
    _say("Écriture…")
    try:
        for rel in added + updated:
            _backup(rel, stamp_dir)
            _write(rel, files[rel])
            written.append(rel)
        for rel in removed:
            _backup(rel, stamp_dir)
            (ROOT / rel).unlink(missing_ok=True)
    except OSError as exc:
        _say(ERR + f"écriture impossible ({exc}).")
        _say("    L'application est-elle encore ouverte ? Fermez-la et "
             "relancez update.bat.")
        _restore(stamp_dir, written, [])
        shutil.rmtree(stamp_dir, ignore_errors=True)
        return 1

    (stamp_dir / "_update.json").write_text(json.dumps(
        {"written": written, "removed": removed}, indent=2), encoding="utf-8")

    caches = _purge_pycache()
    if caches:
        _say(OK + f"{caches} dossier(s) __pycache__ purgé(s).")

    problem = _compiles()
    if problem:
        _say(ERR + f"le code mis à jour ne compile pas : {problem}")
        _say("    Retour à la version précédente…")
        _restore(stamp_dir, written, removed)
        _purge_pycache()
        shutil.rmtree(stamp_dir, ignore_errors=True)
        return 1
    _say(OK + "tout le code compile.")

    _save_manifest(files, commit or {})
    # Une seule sauvegarde conservée : celle qui précède la mise à jour.
    for old in sorted(p for p in BACKUP_DIR.iterdir() if p.is_dir()):
        if old != stamp_dir:
            shutil.rmtree(old, ignore_errors=True)

    _say("")
    _say(OK + f"mise à jour terminée ({len(written)} fichier(s) écrits).")
    _say(INFO + "annulable avec : update.bat --rollback")
    _say(INFO + "relancez run.bat.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="montrer ce qui changerait, sans rien écrire")
    ap.add_argument("--rollback", action="store_true",
                    help="annuler la dernière mise à jour")
    args = ap.parse_args()
    if args.rollback:
        return _rollback()
    return update(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
