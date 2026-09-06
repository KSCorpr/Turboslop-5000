#!/usr/bin/env python3
"""Installe SeedVR2 standalone dans un environnement Python isolé.

Le Python principal de l'application reste intact. ``uv`` fournit un Python
3.12 local, puis les dépendances CUDA 12.6 compatibles Ampere/Pascal sont
installées dans ``tools_repo/seedvr2/venv``.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tools_repo" / "seedvr2"
SOURCE = BASE / "source"
VENV = BASE / "venv"
SEEDVR2_REF = "4490bd1f482e026674543386bb2a4d176da245b9"
SOURCE_URL = ("https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/"
              f"archive/{SEEDVR2_REF}.zip")


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32"
                   else "bin/python")


def _safe_extract(blob: bytes, destination: Path) -> Path:
    """Extrait le ZIP sans autoriser ``../`` ni chemins absolus."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Chemin dangereux dans l'archive : {member.filename}")
        archive.extractall(destination)
    dirs = [p for p in destination.iterdir() if p.is_dir()]
    if len(dirs) != 1:
        raise RuntimeError("Structure inattendue dans l'archive SeedVR2.")
    return dirs[0]


def _install_source() -> None:
    marker = SOURCE / ".turbo-slop-seedvr2-ref"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == SEEDVR2_REF:
        print("[OK] Sources SeedVR2 déjà présentes.")
        return
    print("Téléchargement des sources SeedVR2 officielles…", flush=True)
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Turbo-Slop/1"})
    with urllib.request.urlopen(req, timeout=180) as response:
        blob = response.read()
    staging = BASE / "source-new"
    if staging.exists():
        shutil.rmtree(staging)
    extracted = _safe_extract(blob, staging)
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    shutil.move(str(extracted), str(SOURCE))
    shutil.rmtree(staging, ignore_errors=True)
    marker.write_text(SEEDVR2_REF, encoding="utf-8")
    print(f"[OK] SeedVR2 épinglé au commit {SEEDVR2_REF[:12]}.")


def _install_python() -> Path:
    try:
        import uv  # noqa: F401
    except ImportError:
        print("Installation du gestionnaire d'environnement uv…")
        _run([sys.executable, "-m", "pip", "install", "uv>=0.8,<1"])
    py = _venv_python()
    if not py.is_file():
        print("Création du Python 3.12 isolé…")
        _run([sys.executable, "-m", "uv", "venv", "--python", "3.12",
              "--seed", str(VENV)])
    return py


def _install_dependencies(py: Path) -> None:
    print("Installation de PyTorch CUDA 12.6 (RTX 3060 + GTX 1080 Ti)…")
    _run([sys.executable, "-m", "uv", "pip", "install", "--python", str(py),
          "--index-url", "https://download.pytorch.org/whl/cu126",
          "torch==2.7.1", "torchvision==0.22.1"])
    print("Installation des dépendances SeedVR2…")
    deps = [
        "safetensors", "numpy>=1.26,<2", "tqdm", "psutil", "einops",
        "omegaconf>=2.3.0", "diffusers>=0.33.1,<0.36", "peft>=0.17,<0.19",
        "rotary-embedding-torch>=0.5.3", "opencv-python<5", "gguf",
        "matplotlib<4",
    ]
    _run([sys.executable, "-m", "uv", "pip", "install", "--python", str(py),
          *deps])


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    _install_source()
    py = _install_python()
    _install_dependencies(py)
    probe = "import torch; print(torch.__version__, torch.version.cuda)"
    _run([str(py), "-c", probe])
    print("\n[OK] SeedVR2 installé. Les poids Q8/Q4 seront téléchargés "
          "automatiquement au premier upscale.")


if __name__ == "__main__":
    main()
