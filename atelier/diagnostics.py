"""Rapport matériel/moteur partageable, sans secret utilisateur."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__, hardware, settings
from .engine import sdcpp


ENGINE_MANIFEST = settings.BIN_DIR / "engine-manifest.json"


def _sha256(path: Path | None) -> str:
    if not path or not path.is_file():
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def read_engine_manifest() -> dict[str, Any]:
    try:
        data = json.loads(ENGINE_MANIFEST.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def engine_report() -> dict[str, Any]:
    cli = settings.find_sd_cli()
    display_path = None
    if cli:
        try:
            display_path = str(cli.relative_to(settings.ROOT))
        except ValueError:
            # Ne pas exporter le dossier personnel complet dans un rapport
            # destiné à être partagé.
            display_path = cli.name
    report: dict[str, Any] = {
        "path": display_path,
        "present": bool(cli),
        "manifest": read_engine_manifest(),
    }
    if cli:
        try:
            report["size_bytes"] = cli.stat().st_size
            report["modified_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(cli.stat().st_mtime))
        except OSError:
            pass
        report["sha256"] = _sha256(cli)
        report["supported_options"] = sorted(sdcpp.supported_options(cli))
    return report


def system_report() -> dict[str, Any]:
    prefs = settings.load_prefs()
    # Seulement les choix qui influencent l'exécution. Ne jamais exporter les
    # jetons de téléchargement ni d'autres secrets de preferences.json.
    safe_pref_keys = (
        "gpu_index", "encoder_gpu_index", "text_gpu_index", "auto_fit",
        "split_mode", "params_backend", "auto_optimize", "quant",
        "enc_quant", "flags", "cache_mode", "cache_option",
        "cache_by_model", "max_vram", "stream_layers",
        "conv_direct_diffusion", "conv_direct_vae",
    )
    gpus = []
    free = {g.index: hardware.free_vram_gb(g.index)
            for g in hardware.detect_gpus()}
    used = hardware.used_vram_gb()
    for gpu in hardware.detect_gpus():
        item = asdict(gpu)
        item.update({"sm": gpu.sm, "pcie": gpu.pcie_label,
                     "free_vram_gb": free.get(gpu.index, 0.0),
                     "used_vram_gb": used.get(gpu.index, 0.0)})
        gpus.append(item)
    return {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "app_version": __version__,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": hardware.cpu_name(),
        "ram_gb": hardware.detect_ram_gb(),
        "gpus": gpus,
        "engine": engine_report(),
        "preferences": {k: prefs.get(k) for k in safe_pref_keys},
    }


def write_system_report() -> Path:
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = settings.OUTPUT_DIR / f"system-report-{stamp}.json"
    dest.write_text(json.dumps(system_report(), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return dest


def summary_markdown(report: dict[str, Any] | None = None) -> str:
    data = report or system_report()
    lines = [f"**Application :** {data['app_version']}",
             f"**CPU / RAM:** {data['cpu'] or 'unknown'} · {data['ram_gb']:.0f} GB"]
    for gpu in data.get("gpus", []):
        pcie = f" · {gpu.get('pcie')}" if gpu.get("pcie") else ""
        lines.append(
            f"- GPU #{gpu['index']}: {gpu['name']} · {gpu['vram_gb']:.0f} GB"
            f" · {gpu.get('sm') or gpu.get('arch')}{pcie}")
    engine = data.get("engine", {})
    manifest = engine.get("manifest") or {}
    version = manifest.get("tag") or manifest.get("sd_commit") or "not traced"
    lines.append(f"**sd.cpp engine:** {version}"
                 + ("" if engine.get("present") else " · absent"))
    return "\n".join(lines)
