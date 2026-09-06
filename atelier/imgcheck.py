"""Diagnostic de l'affichage des images (icône d'image cassée).

Pourquoi ce module existe : le symptôme « j'importe une image, elle n'apparaît
pas » est un des rares qu'on ne peut PAS reproduire depuis le code. Six noms de
fichiers hostiles (accents, `#`, `%`, cyrillique, 120 caractères) passent tous
en HTTP 200 sur la machine de développement. Le problème est donc dans
l'environnement — et demander à quelqu'un d'ouvrir la console du navigateur
pour lire un code HTTP n'est pas une réponse acceptable dans une application
qui se veut utilisable sans être développeur.

D'où ce diagnostic : il refait le trajet exact d'une image importée (écriture
dans le cache de Gradio, puis affichage par le serveur web) et affiche le
résultat. Si l'image de test s'affiche, la chaîne de service fonctionne et le
problème est en amont ; si elle est cassée, le rapport dit lequel des maillons
a lâché, avec le chemin exact.

Les causes qu'il sait nommer sont celles qui frappent réellement sous Windows :
un cache posé sur un lecteur RÉSEAU ou un lecteur `subst` (les deux se
comportent mal avec les chemins résolus), un disque plein, un dossier non
inscriptible, un antivirus qui verrouille le fichier le temps de l'analyser
(visible dans le délai de relecture), et un chemin trop long pour l'API Win32.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import settings

# Longueur maximale d'un chemin dans l'API Win32 « ANSI » historique. Au-delà,
# des programmes échouent à ouvrir le fichier alors qu'il existe.
WIN_MAX_PATH = 260

# Au-delà, la relecture d'un fichier qu'on vient d'écrire n'est plus normale :
# quelque chose s'interpose (analyse antivirus, lecteur réseau lent).
SLOW_READBACK_S = 1.0


@dataclass
class Check:
    """Un point de contrôle. `ok` à None = information, pas un verdict."""
    ok: bool | None
    label: str
    detail: str = ""

    @property
    def mark(self) -> str:
        return {True: "✅", False: "❌", None: "•"}[self.ok]


def temp_dir() -> Path:
    """Le cache que Gradio utilise VRAIMENT (la variable, pas notre intention).

    `app.py` la pose avec `setdefault` : un environnement qui la définissait
    déjà garde la main, et le cache peut alors être ailleurs que sous le
    projet — donc hors des dossiers que le serveur a le droit de servir.
    """
    raw = os.environ.get("GRADIO_TEMP_DIR") or ""
    if raw:
        return Path(raw)
    import tempfile
    return Path(tempfile.gettempdir()) / "gradio"


def drive_kind(path: Path) -> str:
    """Type de lecteur Windows : « fixe », « réseau », « amovible »…

    Un cache sur un lecteur réseau ou sur un lecteur `subst` est la cause la
    plus courante d'images cassées de façon INTERMITTENTE : le chemin résolu
    n'est plus celui qu'on croit, et la latence fait échouer des lectures qui
    passeraient en local.
    """
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        drive = os.path.splitdrive(str(path.absolute()))[0]
        if not drive:
            return ""
        kinds = {0: "inconnu", 1: "inexistant", 2: "amovible", 3: "fixe",
                 4: "network", 5: "lecteur optique", 6: "RAM disk"}
        code = ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\")
        return kinds.get(int(code), "inconnu")
    except Exception:  # noqa: BLE001
        return ""


def _free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1.0


def cache_size(path: Path) -> tuple[int, int]:
    """(nombre de fichiers, octets) du cache — sans suivre les liens."""
    n = size = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                n += 1
                size += p.stat().st_size
    except OSError:
        pass
    return n, size


def write_test_image(dest_dir: Path) -> tuple[Path | None, float, str]:
    """Écrit un PNG puis le relit. Retourne (chemin, secondes, erreur).

    La relecture n'est pas une précaution de style : c'est elle qui révèle un
    antivirus qui verrouille le fichier le temps de l'analyser, ou un lecteur
    réseau qui n'a pas encore vu l'écriture.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - Pillow est une dépendance
        return None, 0.0, f"Pillow indisponible : {exc}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"diagnostic_{int(time.time())}.png"
        img = Image.new("RGB", (320, 200), (24, 132, 168))
        d = ImageDraw.Draw(img)
        d.rectangle((8, 8, 311, 191), outline=(255, 255, 255), width=3)
        d.text((24, 92), "TEST", fill=(255, 255, 255))
        start = time.time()
        img.save(dest, format="PNG")
        with open(dest, "rb") as fh:          # relecture réelle
            data = fh.read()
        elapsed = time.time() - start
        if not data.startswith(b"\x89PNG"):
            return None, elapsed, "the file read back is not a valid PNG"
        return dest, elapsed, ""
    except OSError as exc:
        return None, 0.0, str(exc)


# Formats qu'un composant image accepte couramment. La tuile de test est
# écrite dans CHACUN : si le PNG s'affiche et pas le JPEG, le problème n'est
# pas la chaîne de service mais le type de fichier — une piste qu'aucun
# contrôle général ne peut donner.
TEST_FORMATS = [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp"),
                ("GIF", ".gif"), ("BMP", ".bmp")]

_IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def mime_of(suffix: str) -> str:
    """Type MIME que Python associe à une extension.

    Sous Windows, `mimetypes` consulte aussi la BASE DE REGISTRE. Le runner
    Windows de GitHub, comme certaines installations utilisateur, n'y déclare
    pas `.webp`. On enregistre donc explicitement les formats que l'application
    sait produire avant d'interroger Python : une association système absente
    ou détournée ne doit pas casser l'affichage dans notre serveur local.
    """
    import mimetypes
    canonical = _IMAGE_MIMES.get(suffix.lower())
    if canonical:
        # Les deux tables sont utilisées selon la valeur de `strict` du code
        # appelant ; les renseigner toutes les deux rend le résultat stable.
        mimetypes.add_type(canonical, suffix.lower(), strict=True)
        mimetypes.add_type(canonical, suffix.lower(), strict=False)
    return mimetypes.guess_type(f"x{suffix}")[0] or ""


def format_probe(dest_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """(libellés+chemins des tuiles écrites, lignes de rapport MIME)."""
    tiles: list[tuple[str, str]] = []
    lines: list[str] = []
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return tiles, ["Pillow indisponible."]
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, suffix in TEST_FORMATS:
        mime = mime_of(suffix)
        ok_mime = mime.startswith("image/")
        path = dest_dir / f"test_{name.lower()}{suffix}"
        try:
            img = Image.new("RGB", (160, 100), (24, 132, 168))
            img.save(path, format=name)
            tiles.append((str(path), f"{name} — {mime or 'MIME inconnu'}"))
            wrote = True
        except (OSError, KeyError, ValueError) as exc:
            wrote = False
            lines.append(f"❌ **{name}** — could not be written: {exc}")
        if wrote:
            mark = "✅" if ok_mime else "❌"
            detail = (mime if ok_mime else
                      f"`{mime or 'none'}` — Windows does not recognize "
                      f"`{suffix}` as an image (registry database). Gradio "
                      "then serves it as a download and the browser "
                      "displays nothing.")
            lines.append(f"{mark} **{name}** ({suffix}) — {detail}")
    return tiles, lines


def recent_uploads(cache: Path, limit: int = 5) -> list[Path]:
    """Les derniers fichiers réellement déposés par le navigateur.

    C'est le contrôle qui tranche : si l'image que vous venez d'importer est
    là, le dépôt a fonctionné et seul l'affichage est en cause ; si elle n'y
    est pas, c'est l'envoi qui échoue, et regarder du côté du serveur d'images
    ne mènera nulle part.
    """
    out: list[Path] = []
    try:
        for p in cache.rglob("*"):
            # Nos propres tuiles de test ne comptent pas comme des imports.
            if p.is_file() and "diagnostic" not in p.parts:
                out.append(p)
    except OSError:
        return []
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out[:limit]


# Au-delà, un navigateur peut renoncer à décoder l'image et n'afficher qu'une
# icône cassée, alors que le fichier est parfaitement valide. Repère indicatif :
# les limites réelles dépendent du navigateur et de la mémoire disponible.
HUGE_PIXELS = 80_000_000
HUGE_BYTES = 50 * 1024 * 1024


def describe_file(path: Path) -> str:
    """Ce que le fichier EST vraiment : format, taille, dimensions.

    L'extension ne prouve rien — un `.png` qui est en fait un JPEG, un fichier
    tronqué par un transfert, une image de 200 mégapixels : trois cas où le
    navigateur renonce et n'affiche qu'une icône cassée, sans que rien côté
    serveur n'ait échoué.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"❌ illisible : {exc}"
    parts = [f"{size / 1024:.0f} KB"]
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            fmt = im.format or "?"
        parts.append(f"{w}×{h}, {fmt}")
        if fmt and f".{fmt.lower()}" != path.suffix.lower() and not (
                fmt == "JPEG" and path.suffix.lower() in (".jpg", ".jpeg")):
            parts.append(f"⚠️ the extension `{path.suffix}` does not match "
                         f"the content ({fmt})")
        if w * h > HUGE_PIXELS:
            parts.append(f"⚠️ {w * h / 1e6:.0f} megapixels — some browsers "
                         "give up decoding past that")
        if size > HUGE_BYTES:
            parts.append(f"⚠️ {size / 1e6:.0f} MB — very heavy to transfer "
                         "and then decode")
    except Exception as exc:  # noqa: BLE001 - Pillow lève un peu de tout
        parts.append(f"❌ unreadable even by the application: {exc}. The "
                     "file is probably corrupt or truncated")
    return " · ".join(parts)


def checks() -> tuple[list[Check], Path | None]:
    """Tous les points de contrôle + l'image de test à afficher."""
    out: list[Check] = []
    cache = temp_dir()

    try:
        import gradio
        out.append(Check(None, "Gradio version", gradio.__version__))
    except ImportError:  # pragma: no cover
        pass

    out.append(Check(None, "Image cache", f"`{cache}`"))

    # 1. Le cache est-il DANS le projet ?
    #
    # Précision qui a son importance : hors du projet, ce n'est PAS un refus
    # d'autorisation — Gradio sert toujours son propre dossier de dépôt, quel
    # qu'il soit. Le danger est ailleurs : dans %TEMP%, le système se croit
    # autorisé à faire le ménage (Storage Sense, nettoyage de disque,
    # antivirus, redémarrage). La copie disparaît sous les pieds du
    # navigateur, la requête renvoie 404 et l'image casse — par intermittence,
    # ce qui est exactement la signature du symptôme.
    inside = _is_within(cache, settings.ROOT)
    out.append(Check(
        inside, "Cache inside the project folder",
        "" if inside else
        "the cache is outside the project (probably in the system temp "
        "folder). Windows cleans that out whenever it feels like it, and "
        "images already on screen break all at once. A `GRADIO_TEMP_DIR` "
        "variable set in your environment overrides the application's own — "
        "remove it, then restart."))

    # 2. Le dépôt et le cache sont-ils sur le MÊME disque ?
    #
    # Cause avérée, mesurée : quand `os.rename` ne peut pas franchir les deux
    # volumes, Gradio recopie le fichier en tâche de fond MAIS répond tout de
    # suite avec le chemin final. Le navigateur demande alors une image encore
    # incomplète, la réponse est tronquée, et l'icône casse — pendant que
    # l'outil, lui, reçoit le fichier complet une fois la copie finie.
    upload_tmp = Path(settings.ROOT) / "tmp" / "upload"
    if upload_tmp.is_dir():
        same = _same_volume(upload_tmp, cache)
        out.append(Check(
            same, "Upload and cache on the same drive",
            "" if same else
            "uploaded files pass through a different volume from the cache: "
            "the thumbnail is requested before the copy has finished, and the "
            "response arrives truncated. Report it — this is a flaw in the "
            "application, not in your machine."))

    # 3. Type de lecteur (Windows). Réseau/amovible = cause connue.
    kind = drive_kind(cache)
    if kind:
        ok = kind == "fixe"
        out.append(Check(
            ok, f"Drive type: {kind}",
            "" if ok else
            "a cache on a network or removable drive gives intermittently "
            "broken images. Move the project to an internal drive, or point "
            "`GRADIO_TEMP_DIR` at a local folder."))

    # 3. Longueur du chemin (Windows).
    if sys.platform == "win32":
        # Le fichier final ajoute le dossier de hachage (64) et le nom.
        projected = len(str(cache)) + 64 + 40
        ok = projected < WIN_MAX_PATH
        out.append(Check(
            ok, f"Path length: ~{projected} characters",
            "" if ok else
            f"beyond {WIN_MAX_PATH} characters, Windows refuses to open "
            "files that do exist. Put the project closer to the drive root "
            "(e.g. `C:\\TurboSlop`)."))

    # 4. Espace libre.
    free = _free_gb(cache if cache.exists() else settings.ROOT)
    if free >= 0:
        ok = free > 1.0
        out.append(Check(ok, f"Free space: {free:.1f} GB",
                         "" if ok else "a full disk prevents writing the copy "
                                       "the browser is about to ask for."))

    # 5. L'aller-retour écriture/relecture.
    dest, elapsed, err = write_test_image(cache / "diagnostic")
    if err:
        out.append(Check(False, "Writing to the cache", err))
    else:
        slow = elapsed > SLOW_READBACK_S
        out.append(Check(
            not slow, f"Write then read back: {elapsed * 1000:.0f} ms",
            "" if not slow else
            "that is abnormally slow for 60 KB. An antivirus is probably "
            "scanning every file written: add the project folder to its "
            "exclusions."))

    # 6. Taille du cache — informatif, mais un cache énorme se nettoie.
    n, size = cache_size(cache)
    out.append(Check(None, "Cache contents",
                     f"{n} file(s), {size / 1e6:.0f} MB"))
    return out, dest


def _same_volume(a: Path, b: Path) -> bool:
    """Deux chemins sur le même volume ? (lettre de lecteur, puis `st_dev`)."""
    try:
        if os.path.splitdrive(a.absolute())[0].lower() != \
                os.path.splitdrive(b.absolute())[0].lower():
            return False
        return a.stat().st_dev == b.stat().st_dev
    except OSError:
        return False


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(path)).resolve().relative_to(
            Path(os.path.abspath(parent)).resolve())
        return True
    except (ValueError, OSError):
        return False


@dataclass
class Report:
    markdown: str
    test_image: str | None
    tiles: list[tuple[str, str]]     # (chemin, libellé) pour la galerie
    last_upload: str | None


def report() -> Report:
    """Tout ce que l'interface affiche : rapport, image de test, tuiles de
    format, et le dernier fichier réellement importé."""
    cache = temp_dir()
    items, dest = checks()
    bad = [c for c in items if c.ok is False]

    lines = []
    for c in items:
        detail = f" — {c.detail}" if c.detail else ""
        lines.append(f"{c.mark} **{c.label}**{detail}")
    if dest is None:
        lines.append("⚠️ The test image could not be written: that alone is "
                     "the explanation.")

    tiles, mime_lines = format_probe(cache / "diagnostic")
    bad_mime = [l for l in mime_lines if l.startswith("❌")]
    lines.append("\n**File types** — each tile below is written in a "
                 "different format. The ones that fail to display name the "
                 "culprit.")
    lines += mime_lines

    ups = recent_uploads(cache)
    lines.append("\n**Last files uploaded by the browser**")
    if ups:
        for p in ups:
            age = max(0, int(time.time() - p.stat().st_mtime))
            lines.append(f"• `{p.name}` — there are {age // 60} min {age % 60} s "
                         f"· {mime_of(p.suffix) or 'MIME inconnu'} · "
                         f"{describe_file(p)}")
        lines.append("The most recent one is shown at the bottom. **If it "
                     "displays here but not in the tool, the file is intact "
                     "and the problem lies elsewhere; if it is broken here "
                     "too, that file is the one the browser cannot read.**")
    else:
        lines.append("• *None.* Upload an image into a tool, then run this "
                     "diagnostic again: if nothing appears here, it is the "
                     "UPLOAD that fails, not the display.")

    if bad or bad_mime:
        head = (f"### ❌ {len(bad) + len(bad_mime)} problem(s) found\n"
                "The ❌ lines below say what to do.")
    else:
        head = ("### ✅ Nothing abnormal detected\nThe serving chain works. "
                "Look instead at the format tiles and the last uploaded file, "
                "below: that is where a problem specific to ONE file shows "
                "up.")
    return Report("\n\n".join([head] + lines),
                  str(dest) if dest else None,
                  tiles,
                  str(ups[0]) if ups else None)
