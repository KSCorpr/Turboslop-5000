"""Inventaire de tout ce qui s'installe sur le disque : moteurs, modèles,
add-ons du Toolkit, LoRA, sorties, temporaires.

Sert au gestionnaire « 🧹 Gestion & nettoyage » : lister ce qui occupe de la
place, avec sa taille, et permettre de le supprimer sélectivement. Aucun
chemin hors du dossier du projet n'est jamais listé ni supprimé.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import registry, settings
from .engine import tools, trellis


@dataclass
class Item:
    key: str
    label: str
    category: str
    paths: list[Path] = field(default_factory=list)
    note: str = ""
    protected: bool = False        # supprimable, mais on prévient (données perso)

    @property
    def size(self) -> int:
        return sum(path_size(p) for p in self.paths)

    @property
    def installed(self) -> bool:
        return any(p.exists() for p in self.paths)


def path_size(p: Path) -> int:
    """Taille d'un fichier ou d'un dossier (récursif), 0 s'il n'existe pas."""
    try:
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        pass
    return 0


def human(n: int) -> str:
    if n <= 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "To":
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} To"


def _engine_sdcpp_paths() -> list[Path]:
    """Binaire + DLL de sd.cpp dans bin/ (hors sous-dossier trellis)."""
    if not settings.BIN_DIR.is_dir():
        return []
    out = []
    for p in settings.BIN_DIR.iterdir():
        if p.name.lower() == "trellis":
            continue
        out.append(p)
    return out


def _model_dirs_for(model: registry.BaseModel) -> list[Path]:
    """Dossiers de dépôt HF utilisés par un modèle du catalogue."""
    dirs: list[Path] = []
    for comp in model.components:
        repo = getattr(comp, "repo", None)
        if not repo:
            continue
        d = settings.model_repo_dir(repo)
        if d not in dirs:
            dirs.append(d)
    return dirs


def items(prefs: dict | None = None) -> list[Item]:
    """Tout ce qui peut occuper de la place, groupé par catégorie."""
    prefs = prefs or settings.load_prefs()
    out: list[Item] = []

    # --- Moteurs -----------------------------------------------------------
    out.append(Item(
        "engine_sdcpp", "stable-diffusion.cpp engine (sd-cli + DLLs)", "Engines",
        _engine_sdcpp_paths(),
        note="Reinstallable: install.bat / update-engine.bat."))
    out.append(Item(
        "engine_trellis", "trellis.cpp engine (image → 3D)", "Engines",
        [trellis.TRELLIS_BIN_DIR],
        note="Reinstallable: the “Image → 3D” tab."))

    # --- Modèles du catalogue ---------------------------------------------
    # Deux modèles peuvent vivre dans les MÊMES dossiers de dépôt (variantes
    # d'une même famille partageant encodeur ou VAE). Un dossier n'est listé
    # qu'une fois — sinon on compterait sa taille deux fois — mais l'élément
    # porte alors le nom de TOUS les modèles concernés : supprimer les efface
    # tous, et il faut que ce soit écrit avant le clic, pas découvert après.
    seen: dict[Path, int] = {}          # dossier -> index de l'élément porteur
    shared_names: dict[int, list[str]] = {}
    for m in registry.load_base_models(prefs):
        model_dirs = _model_dirs_for(m)
        new = [d for d in model_dirs if d not in seen]
        if new:
            idx = len(out)
            out.append(Item(f"model_{m.id}", f"Model — {m.name}", "Models",
                            new,
                            note="Downloadable again: Model catalog."))
            shared_names[idx] = [m.name]
            for d in new:
                seen[d] = idx
        elif model_dirs:
            # Entièrement contenu dans des dossiers déjà listés : on l'ajoute au
            # nom de l'élément porteur au lieu de le faire disparaître.
            idx = seen[model_dirs[0]]
            shared_names[idx].append(m.name)
    for idx, names in shared_names.items():
        if len(names) > 1:
            out[idx].label = "Models — " + " + ".join(names)
            out[idx].note = ("⚠️ Folders SHARED by these models: deleting "
                             "them removes them all. Downloadable again: "
                             "Model catalog.")

    # --- Autres modèles ----------------------------------------------------
    out.append(Item("upscalers", "Upscalers ESRGAN (GGUF)", "Models",
                    [registry.upscalers_dir()],
                    note="Reinstallable: Toolkit → Enlarge."))
    out.append(Item("trellis_models", "Trellis 3D models (GGUF, ~10 GB)",
                    "Models", [trellis.MODELS_DIR],
                    note="Reinstallable: the “Image → 3D” tab."))

    # --- Add-ons du Toolkit (PyTorch) -------------------------------------
    out += [
        Item("tool_depth", "Toolkit — Depth", "Toolkit add-ons",
             [tools.DEPTH_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_bg", "Toolkit — Background removal", "Toolkit add-ons",
             [tools.BG_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_sam", "Toolkit — SAM cut-out", "Toolkit add-ons",
             [tools.SAM_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_enhance", "Prompt improver (LLM)", "Toolkit add-ons",
             [tools.ENHANCE_MODEL_DIR], note="Reinstallable in one click."),
        # `clip` et `describe` manquaient à cet inventaire : le modèle
        # image → prompt pèse 7,5 Go et n'apparaissait nulle part dans ce que
        # l'utilisateur peut voir ou libérer.
        Item("tool_clip", "Toolkit — CLIP labelling (layers)", "Toolkit add-ons",
             [tools.CLIP_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_describe", "Toolkit — Image → prompt (~7.5 GB)",
             "Toolkit add-ons", [tools.DESCRIBE_MODEL_DIR],
             note="Reinstallable in one click."),
        Item("tool_face", "Toolkit — Face restoration", "Toolkit add-ons",
             [tools.FACE_MODEL_DIR], note="Reinstallable in one click."),
        Item("tool_upscale", "Toolkit — Creative SDXL upscale", "Toolkit add-ons",
             [tools.UPSCALE_DIR], note="Includes ControlNet and custom checkpoints."),
        Item("tool_seedvr2", "Toolkit — SeedVR2 restoration", "Toolkit add-ons",
             [tools.SEEDVR2_DIR], note="An isolated Python + the Q8/Q4 models."),
    ]

    # --- Données utilisateur (prudence) -----------------------------------
    out += [
        Item("loras", "Installed LoRAs", "Your data", [settings.LORA_DIR],
             note="⚠️ Your LoRA files (Civitai imports included).",
             protected=True),
        Item("custom", "Custom models (models/custom)", "Your data",
             [settings.CUSTOM_DIR],
             note="⚠️ Files you dropped in or converted by hand.", protected=True),
        Item("outputs", "Generated images & 3D (outputs/)", "Your data",
             [settings.OUTPUT_DIR],
             note="⚠️ Your creations.", protected=True),
        Item("tmp", "Temporary files (tmp/)", "Your data",
             [settings.TMP_DIR],
             note="Preview caches, working files and the interface's image "
                  "cache. Safe with the application CLOSED; while it is "
                  "running, images already on screen turn into broken icons "
                  "until the page is reloaded."),
    ]
    return out


def by_key(key: str, prefs: dict | None = None) -> Item | None:
    return next((i for i in items(prefs) if i.key == key), None)


def total_size(prefs: dict | None = None) -> int:
    return sum(i.size for i in items(prefs))


def delete(keys: list[str], prefs: dict | None = None) -> tuple[list[str], int]:
    """Supprime les éléments désignés. Retourne (messages, octets libérés).

    Sécurité : on ne supprime QUE des chemins situés sous le dossier du projet.
    """
    msgs: list[str] = []
    freed = 0
    root = settings.ROOT.resolve()
    for key in keys or []:
        item = by_key(key, prefs)
        if item is None:
            msgs.append(f"• {key}: unknown, ignored.")
            continue
        if not item.installed:
            msgs.append(f"• {item.label}: nothing to delete.")
            continue
        size = item.size
        ok = 0
        for p in item.paths:
            try:
                rp = p.resolve()
            except OSError:
                continue
            if root not in rp.parents and rp != root:
                msgs.append(f"• {item.label}: path outside the project ignored ({p}).")
                continue
            if rp == root:
                continue
            try:
                if rp.is_dir():
                    shutil.rmtree(rp, ignore_errors=True)
                elif rp.exists():
                    rp.unlink()
                ok += 1
            except OSError as exc:
                msgs.append(f"• {item.label}: failed on {p.name} ({exc}).")
        if ok:
            freed += size
            msgs.append(f"✓ {item.label} — {human(size)} freed.")
    # Les dossiers de base doivent continuer d'exister.
    settings.ensure_dirs()
    return msgs, freed
