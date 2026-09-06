#!/usr/bin/env python3
"""Télécharge un binaire pré-compilé de stable-diffusion.cpp (sd-cli) dans ./bin.

Pour Windows + CUDA, récupère DEUX archives :
  - la build principale  : sd-master-*-bin-win-cuda12-x64.zip  (contient sd-cli)
  - le runtime CUDA      : cudart-sd-bin-win-cu12-x64.zip       (DLLs CUDA)
…et les décompresse côte à côte (indispensable sans CUDA Toolkit installé).

Usage :
    python scripts/get_sdcpp.py                 # auto (CUDA)
    python scripts/get_sdcpp.py --variant cpu   # build CPU/AVX2
    python scripts/get_sdcpp.py --list          # liste les archives dispo
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

try:
    import requests  # plus robuste qu'urllib sur les redirections GitHub (Windows)
except ImportError:
    requests = None

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"
PREVIOUS_DIR = ROOT / ".engine-previous"
ENGINE_MANIFEST = "engine-manifest.json"
# Les correctifs de découpe du graphe et de re-clamp du budget VRAM sont dans
# les builds officiels 823+. Refuser un build plus ancien évite une régression
# comportementale que la simple détection de flags ne peut pas voir.
MIN_OFFICIAL_BUILD = 823
RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=10"
_UA = {"User-Agent": "atelier"}


def _has_sd_cli(root: Path = BIN_DIR) -> bool:
    names = ("sd-cli.exe", "sd.exe") if platform.system() == "Windows" \
        else ("sd-cli", "sd")
    return any(any(root.rglob(n)) for n in names) if root.exists() else False


# DLL du runtime CUDA 12 nécessaires à la build CUDA de stable-diffusion.cpp.
_CUDA_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")


def _cuda_runtime_present(root: Path = BIN_DIR) -> bool:
    return all((root / d).is_file() for d in _CUDA_DLLS) or \
        bool(list(root.rglob("cudart64_12.dll")))


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def _copy_dlls_from(root: Path, target: Path = BIN_DIR) -> int:
    """Copie les DLL CUDA trouvées sous `root` vers le dossier cible."""
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    wanted = {d.lower() for d in _CUDA_DLLS}
    for dll in root.rglob("*.dll"):
        if dll.name.lower() in wanted and not (target / dll.name).exists():
            shutil.copy(dll, target / dll.name)
            print(f"     + {dll.name}", flush=True)
            n += 1
    return n


def _torch_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if not spec or not spec.submodule_search_locations:
        return None
    lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    return lib if lib.is_dir() else None


def _nvidia_pkg_dir() -> Path | None:
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    return None


def ensure_cuda_runtime(target: Path = BIN_DIR) -> bool:
    """Met les DLL du runtime CUDA dans bin/ via des sources qui marchent
    partout (PyPI), SANS dépendre du CDN des releases GitHub.

    Ordre : déjà présent -> torch/lib (si torch CUDA installé) -> wheels NVIDIA
    PyPI -> en dernier recours, torch CUDA puis copie.
    """
    target.mkdir(parents=True, exist_ok=True)
    if _cuda_runtime_present(target):
        print("Runtime CUDA déjà présent.")
        return True

    # 1) Réutiliser les DLL embarquées par un torch CUDA déjà installé.
    lib = _torch_lib_dir()
    if lib and (lib / "cudart64_12.dll").is_file():
        print("Copie des DLL CUDA depuis torch/lib…")
        if _copy_dlls_from(lib, target) >= 2:
            return True

    # 2) Wheels NVIDIA depuis PyPI (léger, et PyPI fonctionne sur votre réseau).
    try:
        print("Récupération du runtime CUDA via PyPI (nvidia-*-cu12)…")
        _pip("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")
        nv = _nvidia_pkg_dir()
        if nv and _copy_dlls_from(nv, target) >= 2:
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"   (wheels NVIDIA indisponibles : {exc})", flush=True)

    # 3) Dernier recours : torch CUDA (volumineux) puis copie des DLL.
    try:
        print("Installation de PyTorch CUDA 12.1 (fournit le runtime CUDA)…")
        # On évite --force-reinstall (verrouille tbb/mkl). On désinstalle juste
        # torch/torchvision puis on réinstalle la build CUDA.
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                         "torch", "torchvision"])
        _pip("--no-cache-dir", "torch==2.3.0", "torchvision==0.18.0",
             "--index-url", "https://download.pytorch.org/whl/cu121")
        lib = _torch_lib_dir()
        if lib and _copy_dlls_from(lib, target) >= 2:
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"   (échec torch : {exc})", flush=True)

    return _cuda_runtime_present(target)


def _force_ipv4():
    """Force la résolution IPv4 uniquement.

    Sur beaucoup de réseaux Windows, IPv6 est annoncé mais non routable : Python
    tente l'IPv6 du CDN GitHub et reste bloqué (pas de Happy-Eyeballs comme les
    navigateurs). On filtre getaddrinfo pour ne garder que l'IPv4.
    """
    _orig = socket.getaddrinfo

    def _ipv4_only(host, *args, **kwargs):
        res = _orig(host, *args, **kwargs)
        v4 = [r for r in res if r[0] == socket.AF_INET]
        return v4 or res

    socket.getaddrinfo = _ipv4_only


def _fetch_json(url: str):
    if requests is not None:
        r = requests.get(url, headers=_UA, timeout=60)
        r.raise_for_status()
        return r.json()
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _platform_tokens() -> list[str]:
    s = platform.system().lower()
    return {"windows": ["win"], "linux": ["linux", "ubuntu"],
            "darwin": ["darwin", "macos"]}.get(s, [])


def _is_archive(n: str) -> bool:
    return n.endswith((".zip", ".tar.gz", ".tgz"))


def default_variant() -> str:
    """Variante adaptée à la machine, sans avoir à la nommer.

    macOS ne propose que des builds Apple Silicon (Metal) : demander « cuda »
    n'y a aucun sens et ne donnerait aucun résultat. Ailleurs, CUDA reste le
    défaut historique."""
    return "metal" if platform.system() == "Darwin" else "cuda"


def _score_main(name: str, variant: str) -> int:
    """Note l'archive PRINCIPALE (qui contient sd-cli). Exclut le cudart."""
    n = name.lower()
    if not _is_archive(n) or "cudart" in n:
        return -1000
    toks = _platform_tokens()
    if toks and not any(t in n for t in toks):
        return -1000
    score = 10
    if variant == "metal":
        # Les archives macOS sont nommées « …-bin-Darwin-macOS-<ver>-arm64.zip ».
        # Il n'existe pas de build Intel : on écarte explicitement x86_64 pour
        # ne pas télécharger une archive inutilisable si elle apparaissait.
        if "arm64" in n or "aarch64" in n:
            score += 10
        if "x86_64" in n or "x64" in n:
            score -= 20
        if any(x in n for x in ("cuda", "rocm", "vulkan")):
            score -= 20
        return score
    if variant == "cuda":
        if "cuda12" in n or "cu12" in n:
            score += 10
        elif "cuda" in n:
            score += 8
        else:
            score -= 4              # pas une build CUDA
        if "rocm" in n or "vulkan" in n:
            score -= 20
    else:  # cpu
        if any(x in n for x in ("cuda", "rocm", "vulkan")):
            score -= 20
        if "avx2" in n:
            score += 6
        elif "avx512" in n:
            score += 3
        elif "avx" in n:
            score += 4
        elif "noavx" in n:
            score += 1
    if any(t in n for t in ("x64", "amd64", "x86_64")):
        score += 1
    return score


def _find_cudart(assets: list[dict]) -> dict | None:
    for a in assets:
        n = a["name"].lower()
        if "cudart" in n and "win" in n and _is_archive(n):
            return a
    return None


def _latest_release_with_assets() -> dict:
    data = _fetch_json(RELEASES)
    if isinstance(data, dict):  # message d'erreur (rate limit, etc.)
        sys.exit(f"API GitHub : {data.get('message', data)}")
    for rel in data:
        if rel.get("assets"):
            return rel
    sys.exit("Aucune release avec archives trouvée.")


def _progress(got: int, total: int, last: int) -> int:
    step = (total / 20) if total else 5_000_000
    if got - last >= step:
        if total:
            print(f"     {got/1e6:6.1f} / {total/1e6:.1f} Mo ({got*100//total}%)",
                  flush=True)
        else:
            print(f"     {got/1e6:6.1f} Mo téléchargés…", flush=True)
        return got
    return last


# Miroirs GitHub (essayés seulement si le téléchargement direct échoue).
# Utiles sur les réseaux qui filtrent/ralentissent le CDN des releases GitHub.
_MIRRORS = ["https://ghfast.top/", "https://ghproxy.net/", "https://gh.llkk.cc/"]


def _download(url: str) -> bytes:
    """Télécharge un fichier de façon ROBUSTE puis renvoie son contenu.

    Le CDN des releases GitHub coupe souvent la connexion en cours de route.
    On télécharge donc dans un fichier .part avec REPRISE (HTTP Range) : si la
    connexion lâche, on relance là où on s'était arrêté au lieu de tout refaire.
    Direct d'abord (avec reprise, nombreux essais), puis miroirs en secours.
    """
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in url.split("/")[-1] if c.isalnum() or c in "._-")
    tmp = BIN_DIR / ("._part_" + (safe or "download"))

    try:
        return _resumable(url, tmp, retries=10, resume=True)
    except Exception as exc:  # noqa: BLE001
        print(f"   Direct indisponible ({exc}). Essai via miroirs…", flush=True)

    for m in _MIRRORS:
        try:
            print(f"   Miroir : {m.split('/')[2]}", flush=True)
            if tmp.exists():
                tmp.unlink()
            return _resumable(m + url, tmp, retries=3, resume=True)
        except Exception as exc:  # noqa: BLE001
            print(f"     (miroir échoué : {exc})", flush=True)
    raise RuntimeError("téléchargement impossible (direct + miroirs)")


def _resumable(url: str, tmp: Path, retries: int, resume: bool) -> bytes:
    """Télécharge `url` dans `tmp` avec reprise, puis renvoie les octets."""
    import time
    if requests is None:  # repli minimal sans reprise
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    total = None
    last_print = 0
    for attempt in range(1, retries + 1):
        existing = tmp.stat().st_size if (resume and tmp.exists()) else 0
        headers = dict(_UA)
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=(10, 45)) as r:
                if r.status_code == 416:  # déjà complet
                    break
                r.raise_for_status()
                if r.status_code == 206:  # reprise acceptée
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        try:
                            total = int(cr.rsplit("/", 1)[1])
                        except ValueError:
                            pass
                    mode = "ab"
                else:  # 200 : pas de reprise -> on repart de zéro
                    existing = 0
                    cl = r.headers.get("Content-Length")
                    total = int(cl) if cl else None
                    mode = "wb"
                got = existing
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(262144):
                        if not chunk:
                            continue
                        f.write(chunk)
                        got += len(chunk)
                        last_print = _progress(got, total or 0, last_print)
            if total is None or tmp.stat().st_size >= total:
                break  # terminé
            raise IOError(f"interrompu à {tmp.stat().st_size}/{total} octets")
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                raise
            print(f"     (coupure : {exc} — reprise {attempt+1}/{retries}…)",
                  flush=True)
            time.sleep(min(2 * attempt, 8))

    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    print(f"     terminé ({len(data)/1e6:.1f} Mo).", flush=True)
    return data


def _restore_exec_bits(root: Path = BIN_DIR) -> None:
    """Rend le binaire exécutable après extraction (macOS / Linux).

    `zipfile.extractall` ne restitue PAS les permissions Unix : le sd-cli sorti
    d'un .zip arrive en 0644 et se solde par « Permission denied » au premier
    lancement. Les archives Windows n'en ont pas besoin, mais l'appel est sans
    effet là-bas."""
    if platform.system() == "Windows":
        return
    import stat
    for name in ("sd-cli", "sd"):
        for p in root.rglob(name):
            if not p.is_file():
                continue
            try:
                mode = p.stat().st_mode
                p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                print(f"     + exécutable : {p.name}", flush=True)
            except OSError:
                pass


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _extract_to(blob: bytes, name: str, target: Path) -> None:
    """Extraction sûre : refuse les chemins qui sortent du dossier cible."""
    target.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            if any(not _inside(target, target / member.filename)
                   for member in z.infolist()):
                raise RuntimeError("archive ZIP dangereuse (chemin hors dossier)")
            z.extractall(target)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            if any(not _inside(target, target / member.name)
                   for member in t.getmembers()):
                raise RuntimeError("archive TAR dangereuse (chemin hors dossier)")
            t.extractall(target, filter="data")
    _restore_exec_bits(target)


def _extract(blob: bytes, name: str) -> None:
    """Compatibilité interne : extraction directe, hors chemin de mise à jour."""
    _extract_to(blob, name, BIN_DIR)


def _find_sd_cli(root: Path) -> Path | None:
    names = ("sd-cli.exe", "sd.exe") if platform.system() == "Windows" \
        else ("sd-cli", "sd")
    for name in names:
        found = next((p for p in root.rglob(name) if p.is_file()), None)
        if found:
            return found
    return None


def _co_locate_cuda_runtime(root: Path) -> None:
    """Place les DLL à côté du .exe si l'archive utilise un sous-dossier."""
    cli = _find_sd_cli(root)
    if cli is None or platform.system() != "Windows":
        return
    for dll_name in _CUDA_DLLS:
        source = next((p for p in root.rglob(dll_name) if p.is_file()), None)
        dest = cli.parent / dll_name
        if source and source != dest and not dest.exists():
            shutil.copy2(source, dest)


def _validate_staged(root: Path) -> tuple[Path, list[str]]:
    """Smoke-test du nouveau binaire avant de remplacer le moteur courant."""
    cli = _find_sd_cli(root)
    if cli is None:
        raise RuntimeError("archive invalide : aucun sd-cli / sd trouvé")
    _restore_exec_bits(root)
    try:
        proc = subprocess.run([str(cli), "-h"], cwd=str(cli.parent),
                              capture_output=True, text=True, timeout=45,
                              encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"le nouveau moteur ne démarre pas : {exc}") from exc
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    options = sorted(set(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", text)))
    if "--mode" not in options and "--diffusion-model" not in options:
        tail = "\n".join(text.splitlines()[-10:])
        raise RuntimeError(
            "le smoke-test sd-cli -h n'a pas reconnu le moteur\n" + tail)
    return cli, options


def _read_embedded_metadata(root: Path) -> dict:
    path = next((p for p in root.rglob("engine-build.json") if p.is_file()), None)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _keep_what_the_archive_does_not_bring(old: Path, stage: Path) -> None:
    """Reporte dans le nouveau bin/ ce qui n'appartient pas au moteur.

    L'installation remplace le dossier bin/ EN ENTIER (c'est ce qui rend la
    mise à jour atomique et le rollback possible). Effet de bord : tout ce qui
    vivait là sans venir de l'archive disparaissait — à commencer par
    `bin/trellis/`, installé par un tout autre bouton et qui n'a rien à voir
    avec stable-diffusion.cpp. Mettre à jour le moteur d'images désinstallait
    silencieusement le moteur 3D.

    On COPIE au lieu de déplacer : l'ancien dossier devient la sauvegarde de
    rollback, et il doit rester complet lui aussi.
    """
    if not old.is_dir():
        return
    for item in old.iterdir():
        target = stage / item.name
        if target.exists():
            continue
        try:
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
            print(f"     - conservé : {item.name}")
        except OSError as exc:
            print(f"     [!] impossible de conserver {item.name} : {exc}")


def _transactional_install(blob: bytes, archive_name: str, metadata: dict,
                           needs_cuda_runtime: bool = False) -> None:
    """Installe après validation, garde l'ancien moteur pour rollback manuel."""
    stage = Path(tempfile.mkdtemp(prefix=".engine-stage-", dir=str(ROOT)))
    moved_old = False
    try:
        _extract_to(blob, archive_name, stage)
        if needs_cuda_runtime and platform.system() == "Windows" \
                and not _cuda_runtime_present(stage):
            print("Runtime CUDA absent du nouveau moteur — préparation en staging…")
            if not ensure_cuda_runtime(stage):
                raise RuntimeError("runtime CUDA 12 impossible à préparer")
        _co_locate_cuda_runtime(stage)
        cli, options = _validate_staged(stage)
        embedded = _read_embedded_metadata(stage)
        manifest = {
            "schema": 1,
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "archive": archive_name,
            "archive_sha256": hashlib.sha256(blob).hexdigest(),
            "cli": str(cli.relative_to(stage)),
            "supported_options": options,
            **metadata,
            **embedded,
        }
        (stage / ENGINE_MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        _keep_what_the_archive_does_not_bring(BIN_DIR, stage)

        # Une seule sauvegarde, celle qui précède immédiatement la mise à jour.
        if PREVIOUS_DIR.exists():
            shutil.rmtree(PREVIOUS_DIR)
        if BIN_DIR.exists():
            os.replace(BIN_DIR, PREVIOUS_DIR)
            moved_old = True
        os.replace(stage, BIN_DIR)
        try:
            _validate_staged(BIN_DIR)
        except Exception:
            # La validation finale couvre aussi les erreurs liées au changement
            # de chemin. L'ancien moteur revient automatiquement.
            broken = ROOT / ".engine-broken"
            if broken.exists():
                shutil.rmtree(broken)
            os.replace(BIN_DIR, broken)
            if moved_old and PREVIOUS_DIR.exists():
                os.replace(PREVIOUS_DIR, BIN_DIR)
            shutil.rmtree(broken, ignore_errors=True)
            raise
        print("Mise à jour validée. L'ancien moteur reste disponible pour rollback.")
    except Exception:
        # Avant le swap, BIN_DIR n'a pas bougé. Après un échec de swap, le bloc
        # ci-dessus l'a déjà restauré. Dans les deux cas, on ne purge rien.
        if moved_old and not BIN_DIR.exists() and PREVIOUS_DIR.exists():
            os.replace(PREVIOUS_DIR, BIN_DIR)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _rollback() -> None:
    if not PREVIOUS_DIR.exists():
        raise RuntimeError("aucun moteur précédent disponible")
    _validate_staged(PREVIOUS_DIR)
    swap = ROOT / ".engine-swap"
    if swap.exists():
        shutil.rmtree(swap)
    if BIN_DIR.exists():
        os.replace(BIN_DIR, swap)
    try:
        os.replace(PREVIOUS_DIR, BIN_DIR)
        _validate_staged(BIN_DIR)
        if swap.exists():
            os.replace(swap, PREVIOUS_DIR)
    except Exception:
        if BIN_DIR.exists():
            os.replace(BIN_DIR, PREVIOUS_DIR)
        if swap.exists():
            os.replace(swap, BIN_DIR)
        raise
    print("Rollback terminé : le moteur précédent est de nouveau actif.")


def _purge_old_binaries() -> None:
    for n in ("sd-cli.exe", "sd.exe", "sd-cli", "sd"):
        for p in BIN_DIR.rglob(n):
            try:
                p.unlink()
                print(f"     - ancien binaire retiré : {p.name}", flush=True)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cuda", "cpu", "metal"],
                    default=None,
                    help="cuda (Windows/Linux NVIDIA) · metal (macOS Apple "
                         "Silicon) · cpu. Par défaut : selon la machine.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-télécharger même si un binaire est déjà présent "
                         "(pour METTRE À JOUR le moteur)")
    ap.add_argument("--allow-ipv6", action="store_true",
                    help="ne pas forcer l'IPv4 (par défaut on force l'IPv4)")
    ap.add_argument("--rollback", action="store_true",
                    help="réactive le moteur sauvegardé avant la dernière MAJ")
    args = ap.parse_args()
    if args.rollback:
        try:
            _rollback()
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Rollback impossible : {exc}")
        return
    if args.variant is None:
        args.variant = default_variant()
        print(f"Variante retenue pour cette machine : {args.variant}")

    if not args.allow_ipv6:
        _force_ipv4()

    print("Recherche de la dernière release stable-diffusion.cpp…")
    rel = _latest_release_with_assets()
    assets = rel["assets"]
    print(f"Release : {rel.get('tag_name')}")
    # N'applique la borne qu'au format de build officiel connu. Un éventuel tag
    # sémantique futur (v1.2.3) ne doit pas être pris pour la build « 1 ».
    match = re.fullmatch(r"master[-_](\d+)(?:[-_].*)?",
                         rel.get("tag_name") or "", re.IGNORECASE)
    if match and int(match.group(1)) < MIN_OFFICIAL_BUILD:
        sys.exit(
            f"Release trop ancienne ({rel.get('tag_name')}) : build "
            f"{MIN_OFFICIAL_BUILD}+ requis pour les correctifs VRAM.")
    if args.list:
        for a in assets:
            print(" ", a["name"])
        return

    best = max(assets, key=lambda a: _score_main(a["name"], args.variant))
    if _score_main(best["name"], args.variant) <= 0:
        print("Aucune archive principale ne correspond. Disponibles :")
        for a in assets:
            print(" ", a["name"])
        sys.exit("Téléchargez-en une manuellement dans ./bin.")

    # Binaire principal (skip si déjà présent, utile en cas de relance).
    if _has_sd_cli() and not args.force:
        print("Binaire sd-cli déjà présent, on saute le téléchargement.")
    else:
        print(f"Téléchargement (binaire) : {best['name']}")
        blob = _download(best["browser_download_url"])
        _transactional_install(
            blob, best["name"],
            {"source": "official", "tag": rel.get("tag_name"),
             "release_url": rel.get("html_url"), "asset_id": best.get("id"),
             "published_at": rel.get("published_at")},
            needs_cuda_runtime=(args.variant == "cuda"))

    print(f"Installé dans {BIN_DIR}. Binaire sd-cli prêt.")


if __name__ == "__main__":
    main()
