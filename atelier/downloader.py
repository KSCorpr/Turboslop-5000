"""Téléchargement à la demande des modèles depuis Hugging Face.

Pour un modèle de la bibliothèque, télécharge chaque composant (diffusion, vae,
encodeur...) en choisissant le bon fichier par correspondance souple sur le motif.
"""
from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Callable, Iterator

from . import quant, settings
from .registry import BaseModel, Component


def download_lora(repo: str, log: Callable[[str], None] | None = None) -> str:
    """Télécharge le premier .safetensors d'un dépôt LoRA dans loras/.

    Renvoie le nom (sans extension) à utiliser dans la syntaxe <lora:nom:poids>.
    """
    settings.configure_hf_env()
    settings.LORA_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo)
    cands = [f for f in files if f.lower().endswith(".safetensors")]
    if not cands:
        raise RuntimeError(f"No .safetensors in {repo}")
    chosen = sorted(cands, key=len)[0]
    dest = settings.LORA_DIR / Path(chosen).name
    if not dest.is_file():
        if log:
            log(f"Downloading the LoRA {repo}/{chosen}…")
        got = Path(hf_hub_download(repo_id=repo, filename=chosen,
                                   local_dir=str(settings.LORA_DIR)))
        if got.resolve() != dest.resolve() and got.is_file():
            shutil.copy(got, dest)
    return dest.stem


def download_lora_civitai(ref: str, token: str | None = None,
                          log: Callable[[str], None] | None = None) -> str:
    """Télécharge un LoRA depuis Civitai (URL ou ID de version) dans loras/.

    `ref` peut être une URL (…?modelVersionId=123) ou directement l'ID de version.
    Certains modèles exigent un token Civitai (Réglages). Renvoie le nom (sans
    extension) à utiliser dans la liste LoRA."""
    import re
    import requests

    settings.LORA_DIR.mkdir(parents=True, exist_ok=True)
    s = str(ref or "").strip()
    m = re.search(r"modelVersionId=(\d+)", s) or re.search(r"/(\d+)(?:[/?#]|$)", s)
    vid = m.group(1) if m else (s if s.isdigit() else None)
    if not vid:
        raise RuntimeError("Civitai version ID not found. Paste the URL "
                           "containing “modelVersionId=…”, or the numeric ID.")
    token = token or settings.load_prefs().get("civitai_token") or ""
    url = f"https://civitai.com/api/download/models/{vid}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if log:
        log(f"Civitai download (version {vid})…")
    r = requests.get(url, headers=headers, stream=True, allow_redirects=True,
                     timeout=120)
    ctype = r.headers.get("content-type", "")
    if r.status_code in (401, 403) or "text/html" in ctype:
        raise RuntimeError(
            "Civitai refused the download (this model needs a login or "
            "token). Add a Civitai token in Settings, or download the file by "
            "hand into the loras/ folder.")
    r.raise_for_status()
    cd = r.headers.get("content-disposition", "")
    fn = re.search(r'filename="?([^";]+)"?', cd)
    name = fn.group(1) if fn else f"civitai_{vid}.safetensors"
    if not name.lower().endswith((".safetensors", ".gguf", ".ckpt", ".pt")):
        name += ".safetensors"
    dest = settings.LORA_DIR / name
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    if log:
        log(f"  ✓ {name}")
    return dest.stem


def _fn(f: str, pattern: str) -> bool:
    return fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)


def _pick_file(comp: Component, files: list[str]) -> str | None:
    requested = comp.requested()
    if requested in files:
        return requested

    # 1) correspondance directe sur le motif demandé (nom ou chemin)
    direct = [f for f in files if _fn(f, requested)]
    if direct:
        return sorted(direct, key=len)[0]

    # 2) repli quant-tolérant : même motif de base, quant le plus proche
    if comp.token:
        cands = [f for f in files if _fn(f, comp.base_glob())]
        cands = [c for c in cands if "mmproj" not in c.lower()] or cands
        if cands and comp.quant:
            chosen = quant.best([Path(c).name for c in cands], comp.quant)
            return next((c for c in cands if Path(c).name == chosen), cands[0])
        if cands:
            return sorted(cands, key=len)[0]
    return None


def download_component(comp: Component,
                       log: Callable[[str], None] | None = None,
                       *, _repo_files: dict[str, list[str]] | None = None) -> Path:
    # Already-installed requested quant/component must work without the Hub.
    # Do not accept a lower-quant fallback here: a requested upgrade must still
    # download the requested weights when only an older variant is installed.
    from .registry import resolve_component_path
    installed = resolve_component_path(comp)
    local_dir = settings.model_repo_dir(comp.repo)
    if (installed is not None
            and _fn(installed.relative_to(local_dir).as_posix(), comp.requested())
            and installed.stat().st_size > 0):
        if log:
            log(f"  ✓ already there: {comp.role} ({installed.name})")
        return installed
    settings.configure_hf_env()
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub manquant : "
                           "pip install -r requirements.txt") from exc

    if _repo_files is not None and comp.repo in _repo_files:
        files = _repo_files[comp.repo]
    else:
        files = list_repo_files(comp.repo)
        if _repo_files is not None:
            _repo_files[comp.repo] = files
    chosen = _pick_file(comp, files)
    if not chosen:
        # On liste les fichiers pertinents pour diagnostiquer le vrai nommage.
        relevant = [f for f in files
                    if f.lower().endswith((".gguf", ".safetensors", ".sft"))]
        listing = "\n      - " + "\n      - ".join(sorted(relevant)[:40]) \
            if relevant else " (no .gguf/.safetensors found)"
        raise RuntimeError(
            f"No file in {comp.repo} matches neither "
            f"“{comp.requested()}” nor “{comp.base_glob()}”.\n"
            f"    Files available in the repository:{listing}")

    # Transparence : si le quant exact demandé n'existe pas dans le dépôt, on a
    # pris le plus proche EN DESSOUS (repli sûr). On le dit clairement.
    if log and comp.token == "{quant}" and comp.quant:
        got = quant.find_quant(Path(chosen).name)
        if got and got != comp.quant:
            sense = "≤" if quant._idx(got) is not None and quant._idx(comp.quant) \
                is not None and quant._idx(got) <= quant._idx(comp.quant) else "≥"
            log(f"  ⚠️ {comp.quant} unavailable in {comp.repo} → {got} "
                f"(repli, quant {sense} the closest one available)")

    local_dir = settings.model_repo_dir(comp.repo)
    dest = local_dir / chosen
    if dest.is_file() and dest.stat().st_size > 0:
        if log:
            log(f"  ✓ already there: {comp.role} ({chosen})")
        return dest
    if log:
        log(f"  ↓ {comp.role} : {comp.repo}/{chosen}")
    path = hf_hub_download(repo_id=comp.repo, filename=chosen,
                           local_dir=str(local_dir))
    return Path(path)


def download_model(model: BaseModel,
                   log: Callable[[str], None] | None = None) -> Iterator[str]:
    """Télécharge tous les composants manquants d'un modèle. Yields des messages."""
    settings.ensure_dirs()
    yield f"Downloading “{model.name} »…"
    repo_files: dict[str, list[str]] = {}
    for comp in model.components:
        try:
            download_component(comp, log=log, _repo_files=repo_files)
            yield f"  ✓ {comp.role}"
        except Exception as exc:  # noqa: BLE001
            yield f"  ✗ {comp.role} : {exc}"
            return
    yield f"« {model.name}” is ready. ✅"


def download_upscalers(log: Callable[[str], None] | None = None) -> Iterator[str]:
    """Télécharge tous les upscalers ESRGAN GGUF (wbruna/upscalers-sdcpp-gguf)."""
    from .registry import upscaler_config, upscalers_dir
    settings.configure_hf_env()
    settings.ensure_dirs()
    cfg = upscaler_config()
    repo = cfg.get("repo")
    files = cfg.get("files") or []
    if not repo or not files:
        yield "Upscalers not configured."
        return
    from huggingface_hub import hf_hub_download

    local_dir = upscalers_dir()
    local_dir.mkdir(parents=True, exist_ok=True)
    yield f"Downloading {len(files)} upscalers ESRGAN ({repo})…"
    ok = 0
    for fn in files:
        dest = local_dir / fn
        if dest.is_file():
            ok += 1
            yield f"  ✓ already there: {fn}"
            continue
        try:
            hf_hub_download(repo_id=repo, filename=fn, local_dir=str(local_dir))
            ok += 1
            yield f"  ↓ {fn}"
        except Exception as exc:  # noqa: BLE001
            yield f"  ✗ {fn} : {exc}"
    yield f"Upscalers ready ({ok}/{len(files)}). ✅"
