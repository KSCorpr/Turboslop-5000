#!/usr/bin/env python3
"""Installe trellis.cpp (image → 3D) : binaire Windows CUDA + jeu de modèles GGUF.

- Binaire : release pré-compilée de pwilkin/trellis.cpp
    (trellis-cuda-windows-x64.zip) → bin/trellis/ (contient trellis-cli.exe).
- Modèles : dépôt Hugging Face ilintar/trellis2-gguf → models/trellis/
    (jeu complet : SS/SLAT flow, décodeurs, DINOv3, BiRefNet).

trellis-cli est un binaire NATIF (C++/GGML/CUDA, aucun PyTorch), invoqué en
one-shot (comme sd-cli) : le process se termine après chaque génération et
libère toute la VRAM — la stratégie « low-VRAM » (inspirée d'AISmith-3D) pour
faire tenir la 3D sur des cartes modestes (mode 512).

Usage :
    python scripts/get_trellis.py            # binaire + modèles
    python scripts/get_trellis.py --binary   # binaire seul
    python scripts/get_trellis.py --models   # modèles seuls
    python scripts/get_trellis.py --force    # re-télécharger le binaire
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

# Réutilise les téléchargements ROBUSTES de get_sdcpp (reprise, miroirs, IPv4).
import get_sdcpp  # noqa: E402

# SOURCE DE VÉRITÉ UNIQUE pour les emplacements : atelier.settings (qui honore
# la préférence « models_dir », donc un dossier externe type NVMe). Ne JAMAIS
# redéfinir les chemins ici : l'installeur écrirait ailleurs que là où l'app
# cherche, et rien ne le signalerait.
from atelier import settings as _settings  # noqa: E402

BIN_DIR = _settings.BIN_DIR
TRELLIS_BIN_DIR = BIN_DIR / "trellis"
MODELS_DIR = _settings.MODELS_DIR / "trellis"

GH_RELEASE = "https://api.github.com/repos/pwilkin/trellis.cpp/releases/latest"
HF_MODEL_REPO = "ilintar/trellis2-gguf"

# --------------------------------------------------------------------------- #
#  CHOIX DU PAQUET — et pourquoi ce n'est plus « Vulkan par défaut ».
#
#  Longtemps, la build CUDA de trellis.cpp ne tournait PAS sur toutes les cartes
#  NVIDIA : son CMakeLists ÉCRASAIT la liste d'architectures passée par la CI
#  (« set_target_properties(... CUDA_ARCHITECTURES "86;120") »), donc seules les
#  RTX 30xx et 50xx recevaient du code machine pour `deform_conv.cu` et
#  `decimate_qem.cu`. Ailleurs, « no kernel image is available » — et comme les
#  erreurs CUDA sont rémanentes, c'est l'opération ggml SUIVANTE qui la
#  rapportait, ce qui égarait le diagnostic. D'où Vulkan par défaut.
#
#  v0.6.0 (19 août 2026) corrige ça à la racine : l'écrasement est devenu un
#  simple défaut (« if(NOT CMAKE_CUDA_ARCHITECTURES) »), donc la liste de la CI
#  est enfin respectée, et une seconde archive vise les cartes anciennes.
#  Vérifié dans .github/workflows/release.yml au tag v0.6.0 :
#
#      cuda    (CUDA 13.1) -> archs 75;80;86;89;90;120   Turing et plus récent
#      cuda12  (CUDA 12.9) -> archs 60;61;70             Pascal et Volta
#
#  Concrètement : RTX 2080 Ti (7.5) et GTX 1080 Ti (6.1), jusqu'ici renvoyées
#  sur Vulkan, ont maintenant chacune leur paquet CUDA.
#
#  Vulkan reste le repli — rien n'y est compilé par architecture, la
#  convolution déformable passe par un shader de calcul — pour toute carte
#  qu'aucune des deux listes ne couvre (AMD, Intel, NVIDIA plus récente que la
#  CI amont).
# --------------------------------------------------------------------------- #
CUDA_ARCHS = frozenset({"7.5", "8.0", "8.6", "8.9", "9.0", "12.0"})
CUDA12_ARCHS = frozenset({"6.0", "6.1", "7.0"})


def cuda_build_supports(compute_cap: str) -> bool:
    """Une build CUDA amont couvre-t-elle cette carte ?"""
    cap = (compute_cap or "").strip()
    return cap in CUDA_ARCHS or cap in CUDA12_ARCHS


def preferred_backend(compute_cap: str | None = None) -> str:
    """« cuda », « cuda12 » ou « vulkan », selon la carte détectée.

    Sans capacité de calcul connue on choisit Vulkan : mieux vaut un paquet qui
    marche partout qu'un paquet plus rapide qui refuse de démarrer.
    """
    cap = (compute_cap or "").strip()
    if cap in CUDA_ARCHS:
        return "cuda"
    if cap in CUDA12_ARCHS:
        return "cuda12"
    return "vulkan"


def _is_binary(p: Path) -> bool:
    """Détecte un exécutable trellis utilisable (serveur OU cli, nom variable)."""
    if not p.is_file():
        return False
    n = p.name.lower()
    if platform.system() == "Windows" and not n.endswith(".exe"):
        return False
    stem = n[:-4] if n.endswith(".exe") else n
    # La release Windows fournit trellis-server ; on accepte aussi trellis-cli.
    return ("trellis" in stem and not any(
        x in stem for x in ("test", "studio", "bench", "convert")))


def _find_binary() -> Path | None:
    if not BIN_DIR.exists():
        return None
    hits = [p for p in BIN_DIR.rglob("*") if _is_binary(p)]
    if not hits:
        return None
    # Priorité au serveur (ce que fournit la release), sinon le premier trellis.
    srv = [p for p in hits if "server" in p.name.lower()]
    return srv[0] if srv else hits[0]


def has_cli() -> bool:
    return _find_binary() is not None


def _list_exes(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted(p.name for p in base.rglob("*")
                  if p.is_file() and p.name.lower().endswith(".exe"))


def has_models() -> bool:
    return MODELS_DIR.is_dir() and any(MODELS_DIR.rglob("*.gguf"))


def _backend_of(name: str) -> str:
    """Backend d'une archive, d'après son nom — sans confondre cuda et cuda12.

    « cuda » est un préfixe de « cuda12 » : un simple `in` ferait passer
    l'archive Pascal pour l'archive Turing+, et la carte récente hériterait
    d'un binaire compilé pour des architectures qu'elle n'a pas. On lit donc le
    segment ENTIER entre deux tirets (« trellis-<backend>-windows-x64.zip »).
    """
    parts = name.lower().replace(".zip", "").replace(".tar.gz", "").split("-")
    for token in parts:
        if token in ("cuda", "cuda12", "vulkan", "rocm"):
            return token
    return ""


def _pick_asset(assets: list[dict], backend: str = "cuda") -> dict | None:
    """Archive Windows du paquet demandé, avec repli sur Vulkan.

    Le repli compte : une release où un job de CI a échoué ne doit pas bloquer
    l'installation.
    """
    windows = {}
    for a in assets:
        n = a.get("name", "").lower()
        if not n.endswith(".zip") or "win" not in n or "studio" in n:
            continue
        kind = _backend_of(n)
        if kind and kind != "rocm" and kind not in windows:
            windows[kind] = a
    if backend in windows:
        return windows[backend]
    # Le SEUL repli valable est Vulkan. Passer de « cuda » à « cuda12 » (ou
    # l'inverse) donnerait un binaire compilé pour d'autres architectures que
    # celles de la carte : il se téléchargerait, s'installerait, et échouerait
    # au premier noyau par « no kernel image is available ».
    return windows.get("vulkan")


def install_binary(force: bool = False, log=print,
                   backend: str | None = None) -> bool:
    if has_cli() and not force:
        log("Binaire trellis-cli déjà présent, on saute (--force pour MAJ).")
        return True
    if backend is None:
        # Choix guidé par la carte : voir CUDA_ARCHS plus haut.
        cap = ""
        try:
            from atelier import hardware
            gpus = hardware.detect_gpus()
            if gpus:
                cap = max(gpus, key=lambda g: g.vram_gb).compute_cap
        except Exception:  # noqa: BLE001
            pass
        backend = preferred_backend(cap)
        sm = f"sm_{cap.replace('.', '')}" if cap else ""
        if backend == "vulkan":
            log("Paquet VULKAN retenu : "
                + (f"aucune des deux archives CUDA amont ne compile pour {sm}."
                   if cap else "votre carte n'a pas pu être identifiée.")
                + " Vulkan marche sur toutes les cartes.")
        elif backend == "cuda12":
            log(f"Paquet CUDA12 retenu ({sm} — Pascal/Volta, archive CUDA 12.9 "
                "prévue pour les cartes anciennes).")
        else:
            log(f"Paquet CUDA retenu ({sm} — Turing ou plus récent).")
    log("Recherche de la dernière release pwilkin/trellis.cpp…")
    rel = get_sdcpp._fetch_json(GH_RELEASE)
    if isinstance(rel, dict) and rel.get("message") and not rel.get("assets"):
        log(f"API GitHub : {rel.get('message')}")
        return False
    asset = _pick_asset(rel.get("assets", []), backend)
    if not asset:
        log("Aucune archive Windows trouvée dans la release. Disponibles :")
        for a in rel.get("assets", []):
            log("  " + a.get("name", "?"))
        return False
    log(f"Release : {rel.get('tag_name')} — {asset['name']} "
        f"({asset.get('size', 0) / 1e6:.0f} Mo)")
    # MISE À JOUR : on vide l'ancienne version avant d'extraire, sinon des DLL
    # obsolètes de la release précédente resteraient à côté des nouvelles.
    if force and TRELLIS_BIN_DIR.exists():
        import shutil
        shutil.rmtree(TRELLIS_BIN_DIR, ignore_errors=True)
        log("     (ancienne version du moteur retiree)")
    TRELLIS_BIN_DIR.mkdir(parents=True, exist_ok=True)
    blob = get_sdcpp._download(asset["browser_download_url"])
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(TRELLIS_BIN_DIR)
    found = _find_binary()
    if found is not None:
        log(f"[OK] Binaire installé : {found}")
        return True
    exes = _list_exes(TRELLIS_BIN_DIR)
    log("[!] trellis-cli introuvable après extraction. Exécutables trouvés :")
    for name in exes:
        log("    - " + name)
    if not exes:
        log("    (aucun .exe — l'archive n'a peut-être pas le binaire attendu)")
    return False


# Variantes de poids : f16 (défaut, dépôt racine) ou quantifiées (sous-dossiers
# q8/ et q4/ du dépôt HF). Tailles annoncées en amont.
VARIANTS = {
    "f16": ("racine du dépôt", "~16,5 Go — référence"),
    "q8": ("q8/", "~9,9 Go — quasi sans perte"),
    "q4": ("q4/", "~6 Go — léger grain de texture"),
}


def variant_dir(variant: str) -> Path:
    """Dossier local d'une variante (f16 = racine de models/trellis)."""
    return MODELS_DIR if variant == "f16" else MODELS_DIR / variant


def has_models(variant: str = "f16") -> bool:
    d = variant_dir(variant)
    if not d.is_dir():
        return False
    # En f16, ne PAS compter les .gguf des sous-dossiers q8//q4/.
    files = d.glob("*.gguf") if variant == "f16" else d.rglob("*.gguf")
    return any(files)


def install_models(variant: str = "f16", log=print) -> bool:
    if variant not in VARIANTS:
        log(f"Variante inconnue : {variant} (attendu : "
            f"{', '.join(VARIANTS)}).")
        return False
    if has_models(variant):
        log(f"Modèles trellis « {variant} » déjà présents, on saute.")
        return True
    try:
        import os
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        from huggingface_hub import snapshot_download
    except ImportError:
        log("huggingface_hub manquant (pip install huggingface_hub).")
        return False
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if variant == "f16":
        # Racine seulement : on EXCLUT les sous-dossiers quantifiés.
        allow = ["*.gguf", "*.json", "*.txt"]
        ignore = ["q8/*", "q4/*"]
    else:
        allow = [f"{variant}/*"]
        ignore = None
    log(f"Téléchargement des modèles trellis « {variant} » "
        f"({VARIANTS[variant][1]}) → {variant_dir(variant)} (reprise auto)…")
    try:
        # local_dir = racine : les fichiers q8/ et q4/ atterrissent
        # naturellement dans leur sous-dossier.
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(MODELS_DIR),
                          allow_patterns=allow, ignore_patterns=ignore)
    except Exception as exc:  # noqa: BLE001
        log(f"Échec du téléchargement des modèles : {exc}")
        return False
    if has_models(variant):
        log(f"[OK] Modèles trellis « {variant} » en place.")
        return True
    log("[!] Aucun .gguf après téléchargement — vérifiez le dépôt HF.")
    return False


def install_all(force: bool = False, variant: str = "f16", log=print,
                backend: str | None = None) -> bool:
    ok_bin = install_binary(force=force, log=log, backend=backend)
    ok_mdl = install_models(variant=variant, log=log)
    return ok_bin and ok_mdl


def main():
    # Console Windows en cp1252 : force l'UTF-8 pour ne pas planter sur un
    # caractère non-encodable (accents, symboles) dans les logs.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="store_true", help="binaire seul")
    ap.add_argument("--models", action="store_true", help="modèles seuls")
    ap.add_argument("--variant", choices=list(VARIANTS), default="f16",
                    help="variante de poids : f16 (défaut), q8 ou q4")
    ap.add_argument("--force", action="store_true",
                    help="re-télécharger le binaire même s'il est présent")
    ap.add_argument("--backend", choices=["auto", "cuda", "cuda12", "vulkan"],
                    default="auto",
                    help="paquet à installer. auto (défaut) = cuda pour "
                         "Turing et plus récent, cuda12 pour Pascal/Volta, "
                         "vulkan si aucune archive CUDA ne couvre la carte.")
    ap.add_argument("--allow-ipv6", action="store_true")
    args = ap.parse_args()
    if not args.allow_ipv6:
        get_sdcpp._force_ipv4()

    # Trace explicite : on voit tout de suite OÙ ça s'installe (et donc si un
    # dossier de modèles externe est bien pris en compte).
    print(f"Dossier des modèles : {MODELS_DIR}")
    print(f"Dossier du moteur   : {TRELLIS_BIN_DIR}")

    backend = None if args.backend == "auto" else args.backend
    if args.binary:
        ok = install_binary(force=args.force, backend=backend)
    elif args.models:
        ok = install_models(variant=args.variant)
    else:
        ok = install_all(force=args.force, variant=args.variant,
                         backend=backend)
    print("Terminé." if ok else "Terminé avec des erreurs (voir ci-dessus).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
