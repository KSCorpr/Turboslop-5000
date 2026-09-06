"""Emplacement des modèles : validation d'un dossier externe et déplacement.

Permet de sortir les modèles du dossier du projet (ex. vers un NVMe rapide,
ou un disque plus grand) sans rien casser. Le chemin est mémorisé dans les
préférences (`models_dir`) et pris en compte au **redémarrage** — plusieurs
modules capturent le dossier à l'import.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from . import settings


def current() -> Path:
    return settings.MODELS_DIR


def is_default() -> bool:
    return settings.MODELS_DIR.resolve() == settings.DEFAULT_MODELS_DIR.resolve()


def configured() -> str:
    """Chemin enregistré dans les préférences ("" = défaut du projet)."""
    return (settings.load_prefs().get("models_dir") or "").strip()


def dir_size(p: Path) -> int:
    try:
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        pass
    return 0


def free_space(p: Path) -> int:
    """Espace libre sur le volume du chemin (remonte au premier parent existant)."""
    probe = p
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def validate(dest_raw: str) -> tuple[Path | None, str]:
    """Valide un dossier cible. Retourne (chemin, message d'erreur éventuel)."""
    raw = (dest_raw or "").strip().strip('"')
    if not raw:
        return None, "Give a path (or use “Back to the project folder”)."
    dest = Path(raw).expanduser()
    if not dest.is_absolute():
        return None, ("The path must be **absolute** (e.g. `D:\\AI\\models` "
                      "or `/mnt/nvme/models`).")
    src = current().resolve()
    try:
        d = dest.resolve()
    except OSError as exc:
        return None, f"Chemin invalide : {exc}"
    if d == src:
        return None, "That is already the current models folder."
    if src in d.parents:
        return None, ("The destination is **inside** the current models "
                      "folder — pick a folder outside it.")
    if d in src.parents:
        return None, ("The destination **contains** the current models folder "
                      "— pick another one.")
    # Le parent doit exister (on ne crée qu'un niveau).
    if not dest.exists() and not dest.parent.exists():
        return None, f"The parent folder does not exist: `{dest.parent}`"
    if dest.exists() and not dest.is_dir():
        return None, "The destination exists and is not a folder."
    return dest, ""


def save(dest: Path | None) -> str:
    """Enregistre l'emplacement (None = revenir au dossier du projet)."""
    prefs = settings.load_prefs()
    prefs["models_dir"] = None if dest is None else str(dest)
    settings.save_prefs(prefs)
    where = "the project folder (`models/`)" if dest is None else f"`{dest}`"
    return (f"✅ Location saved: {where}.\n\n"
            "**Restart the application** (`run.bat` / `run.sh`) to "
            "apply it.")


def move(dest: Path, log=None) -> Iterator[str]:
    """Déplace le contenu des modèles vers `dest`, en streamant la progression.

    Déplacement entrée par entrée : sur le même disque c'est instantané ;
    d'un disque à l'autre, c'est une copie + suppression (long).
    """
    src = current()
    def _emit(m: str) -> str:
        if log:
            log(m)
        return m

    if not src.is_dir() or not any(src.iterdir()):
        yield _emit("No model to move (the source folder is empty).")
        return

    total = dir_size(src)
    dest.mkdir(parents=True, exist_ok=True)
    avail = free_space(dest)
    yield _emit(f"Source: {src}")
    yield _emit(f"Destination: {dest}")
    yield _emit(f"To move: {_human(total)} · free at the destination: "
                f"{_human(avail)}")
    # Marge de 2 % : la copie inter-disques a besoin de la place complète.
    if avail and avail < total * 1.02:
        yield _emit("❌ Not enough room at the destination — aborting.")
        return

    entries = sorted(src.iterdir())
    done = 0
    errors = 0
    for i, entry in enumerate(entries, 1):
        target = dest / entry.name
        yield _emit(f"[{i}/{len(entries)}] {entry.name} "
                    f"({_human(dir_size(entry) if entry.is_dir() else entry.stat().st_size)})…")
        try:
            if target.exists():
                yield _emit(f"    already present at the destination — skipped.")
                continue
            shutil.move(str(entry), str(target))
            done += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            yield _emit(f"    ⚠️ failed: {exc}")

    yield _emit(f"\n{done} item(s) moved"
                + (f", {errors} failure(s)" if errors else "") + ".")
    if errors:
        yield _emit("⚠️ Some items could not be moved: the location was NOT "
                    "changed. Close whatever might be using them, then try "
                    "again.")
        return
    yield _emit(save(dest))


# --------------------------------------------------------------------------
#  Déplacement SÉLECTIF (par élément) avec lien de retour.
#  On déplace un dossier ailleurs, puis on crée à son ancien emplacement une
#  JONCTION (Windows) ou un lien symbolique (Linux/Mac). Tout le code de l'app
#  continue d'utiliser le chemin d'origine : aucun réglage, aucun redémarrage.
# --------------------------------------------------------------------------
def is_link(p: Path) -> bool:
    """Vrai si le chemin est un lien/jonction (et non un vrai dossier)."""
    try:
        if p.is_symlink():
            return True
        if os.name == "nt" and p.exists():
            # Jonction NTFS : attribut « point d'analyse » (reparse point).
            return bool(p.lstat().st_file_attributes & 0x400)  # REPARSE_POINT
    except (OSError, AttributeError):
        pass
    return False


def link_target(p: Path) -> Path | None:
    try:
        return Path(os.path.realpath(p)) if is_link(p) else None
    except OSError:
        return None


def _make_link(link: Path, target: Path) -> None:
    """Crée un lien `link` -> `target` (jonction sous Windows, symlink sinon)."""
    if os.name == "nt":
        # mklink /J : jonction de dossier — ne demande PAS de privilèges
        # administrateur (contrairement à /D pour les liens symboliques).
        res = subprocess.run(["cmd", "/c", "mklink", "/J", str(link),
                              str(target)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise OSError((res.stderr or res.stdout or "mklink failed").strip())
    else:
        link.symlink_to(target, target_is_directory=True)


def _unlink(p: Path) -> None:
    """Supprime un lien/jonction sans toucher à sa cible."""
    if os.name == "nt" and not p.is_symlink():
        p.rmdir()          # une jonction se retire comme un dossier vide
    else:
        p.unlink()


def relocate(paths: list[Path], dest_root: Path, log=None) -> Iterator[str]:
    """Déplace des dossiers vers `dest_root` et laisse un lien à leur place."""
    def _emit(m: str) -> str:
        if log:
            log(m)
        return m

    dest_root.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if p.exists() and not is_link(p)]
    if not todo:
        yield _emit("Nothing to move (missing, or already moved through a link).")
        return

    total = sum(dir_size(p) for p in todo)
    avail = free_space(dest_root)
    yield _emit(f"To move: {_human(total)} · free at the destination: "
                f"{_human(avail)}")
    if avail and avail < total * 1.02:
        yield _emit("❌ Not enough room at the destination — aborting.")
        return

    for i, src in enumerate(todo, 1):
        target = dest_root / src.name
        yield _emit(f"[{i}/{len(todo)}] {src.name} ({_human(dir_size(src))})…")
        if target.exists():
            yield _emit("    ⚠️ already present at the destination — skipped.")
            continue
        try:
            shutil.move(str(src), str(target))
        except Exception as exc:  # noqa: BLE001
            yield _emit(f"    ❌ cannot move: {exc}")
            continue
        try:
            _make_link(src, target)
            yield _emit(f"    ✓ moved, link created: {src.name} → {target}")
        except Exception as exc:  # noqa: BLE001
            # Le lien a échoué : on remet en place pour ne rien casser.
            try:
                shutil.move(str(target), str(src))
                yield _emit(f"    ❌ cannot create the link ({exc}) — put back.")
            except Exception as exc2:  # noqa: BLE001
                yield _emit(f"    ‼️ cannot create the link ({exc}) AND could not "
                            f"put it back ({exc2}). Files are here: "
                            f"{target}")


def restore(paths: list[Path], log=None) -> Iterator[str]:
    """Ramène dans le projet des éléments déplacés (supprime le lien)."""
    def _emit(m: str) -> str:
        if log:
            log(m)
        return m

    todo = [p for p in paths if is_link(p)]
    if not todo:
        yield _emit("No moved item to bring back.")
        return
    for i, link in enumerate(todo, 1):
        target = link_target(link)
        yield _emit(f"[{i}/{len(todo)}] {link.name} ← {target}")
        if target is None or not target.exists():
            yield _emit("    ❌ target not found (drive unplugged?) — skipped.")
            continue
        try:
            _unlink(link)
            shutil.move(str(target), str(link))
            yield _emit("    ✓ brought back into the project.")
        except Exception as exc:  # noqa: BLE001
            yield _emit(f"    ❌ failed: {exc}")


def _human(n: int) -> str:
    if n <= 0:
        return "0 o"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "To":
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} To"
