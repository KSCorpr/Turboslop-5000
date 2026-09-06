"""Banc d'essai reproductible des placements GPU et des variantes Krea 2."""
from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import diagnostics, hardware, registry, settings
from .engine import generate


BENCHMARK_SEED = 424242
BENCHMARK_PROMPT = (
    "editorial photograph of a red enamel coffee maker on a blue workbench, "
    "soft window light, legible label TEST 42, realistic materials")


@dataclass(frozen=True)
class Placement:
    key: str
    label: str
    prefs_patch: dict


def _selected_gpu(prefs: dict, gpus: tuple[hardware.Gpu, ...]) -> hardware.Gpu | None:
    wanted = prefs.get("gpu_index")
    selected = next((g for g in gpus if g.index == wanted), None)
    return selected or (max(gpus, key=lambda g: g.vram_gb) if gpus else None)


def placement_candidates(prefs: dict | None = None,
                         gpus: tuple[hardware.Gpu, ...] | None = None
                         ) -> list[Placement]:
    """Scénarios réellement comparables sur la machine, sans Auto-Fit."""
    prefs = prefs or settings.load_prefs()
    gpus = gpus if gpus is not None else hardware.detect_gpus()
    main = _selected_gpu(prefs, gpus)
    if main is None:
        return []
    base_flags = hardware.auto_profile(main.index).flags()
    base_flags["vae_tiling"] = True
    staged = {**base_flags, "offload_to_cpu": True,
              "clip_on_cpu": False, "vae_on_cpu": False}
    resident = {**base_flags, "offload_to_cpu": False,
                "clip_on_cpu": False, "vae_on_cpu": False}
    common = {"auto_optimize": False, "gpu_index": main.index,
              "auto_fit": False, "split_mode": "layer"}
    out = [Placement(
        "single-staged", f"{main.name} alone · weights in RAM",
        {**common, "encoder_gpu_index": None, "params_backend": "",
         "flags": staged})]
    secondary = next((g for g in gpus if g.index != main.index), None)
    if secondary is not None:
        mapping = (f"diffusion=cuda{main.index},vae=cuda{main.index},"
                   f"te=cuda{secondary.index}")
        # `encoder_placement_forced` : la génération ordinaire refuse de faire
        # CALCULER l'encodeur sur une carte sans tensor cores. Ici c'est
        # justement ce qu'on veut mesurer — sans ce drapeau, le banc d'essai
        # exécuterait deux fois le même scénario et conclurait « aucune
        # différence ».
        out.extend([
            Placement(
                "dual-resident",
                f"{main.name} diffusion · {secondary.name} resident encoder",
                {**common, "encoder_gpu_index": secondary.index,
                 "encoder_placement_forced": True,
                 "params_backend": mapping, "flags": resident}),
            Placement(
                "dual-staged",
                f"{main.name} diffusion · {secondary.name} calcul · poids en RAM",
                {**common, "encoder_gpu_index": secondary.index,
                 "encoder_placement_forced": True,
                 "params_backend": "*=cpu", "flags": staged}),
        ])
    return out


def _merge_prefs(base: dict, patch: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if key == "flags" and isinstance(value, dict):
            merged["flags"] = {**merged.get("flags", {}), **value}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _benchmark_prefs(base: dict, patch: dict) -> dict:
    merged = _merge_prefs(base, patch)
    # Un benchmark mesure le placement, pas le cache entre pas.
    merged["cache_mode"] = ""
    merged["cache_option"] = ""
    merged["cache_by_model"] = {}
    merged["max_vram"] = ""
    merged["stream_layers"] = False
    # A previous user's resident model would otherwise bias placement timing
    # and baseline VRAM. This benchmark explicitly measures the CLI path.
    merged["resident_engine"] = False
    return merged


def _monitor_peak(stop: threading.Event, peak: dict[int, float]) -> None:
    while not stop.wait(0.2):
        for idx, used in hardware.used_vram_gb().items():
            peak[idx] = max(peak.get(idx, 0.0), used)


class Cancelled(RuntimeError):
    """Le test a été interrompu à la demande de l'utilisateur."""


# Un tir de chauffe JETÉ, puis N tirs mesurés. Le premier tir d'un profil lit
# ~9 Go de GGUF sur un disque froid ; les suivants tapent le cache du système.
# Sans chauffe, c'est le PREMIER profil testé qui est pénalisé, pas le plus
# lent — et le classement dit alors dans quel ordre on a lancé les tests.
WARMUP_RUNS = 1
MEASURED_RUNS = 3


def _one_run(model_id: str, prefs: dict, defaults: dict,
             log: Callable[[str], None] | None) -> tuple[float, list, dict]:
    """Un tir chronométré, avec le pic VRAM observé pendant CE tir."""
    # La ligne de base est relevée AVANT le tir et retranchée ensuite : ce qui
    # nous intéresse est ce que le tir consomme, pas ce que le bureau occupait
    # déjà. Sans ça, deux machines identiques donnent des chiffres différents
    # selon ce qui tourne à côté.
    from .engine import release_resident_engine
    release_resident_engine("hardware benchmark", log)
    baseline = hardware.used_vram_gb()
    peak = dict(baseline)
    stop = threading.Event()
    watcher = threading.Thread(target=_monitor_peak, args=(stop, peak), daemon=True)
    watcher.start()
    started = time.perf_counter()
    try:
        outputs = generate.generate(
            model_id=model_id, prompt=BENCHMARK_PROMPT, negative="",
            steps=max(1, int(defaults.get("steps", 4))),
            cfg_scale=float(defaults.get("cfg_scale", 1.0)),
            width=512, height=512, seed=BENCHMARK_SEED, batch_count=1,
            sampler=defaults.get("sampler", "euler"),
            schedule=defaults.get("scheduler", "auto"),
            flow_shift=float(defaults.get("flow_shift", 0.0) or 0.0),
            save_prompt=False, prefs_override=prefs, log=log)
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        watcher.join(timeout=1)
    delta = {idx: round(max(0.0, used - baseline.get(idx, 0.0)), 2)
             for idx, used in peak.items()}
    return elapsed, outputs, delta


def _run_case(model_id: str, prefs: dict, label: str,
              log: Callable[[str], None] | None = None,
              cancel: Callable[[], bool] | None = None) -> dict:
    model = registry.get_base_model(model_id, prefs)
    if model is None:
        return {"label": label, "ok": False, "error": f"Unknown model: {model_id}"}
    if not registry.model_is_ready(model):
        missing = [c.role for c in registry.missing_components(model)]
        return {"label": label, "ok": False,
                "error": "Missing files: " + ", ".join(missing)}
    d = model.defaults
    started = time.perf_counter()
    times: list[float] = []
    outputs: list = []
    peak: dict[int, float] = {}
    total = WARMUP_RUNS + MEASURED_RUNS
    try:
        for i in range(1, total + 1):
            if cancel and cancel():
                raise Cancelled("test interrompu")
            kind = "chauffe" if i <= WARMUP_RUNS else f"mesure {i - WARMUP_RUNS}"
            if log:
                log(f"[benchmark] {label} — {kind} ({i}/{total})")
            elapsed, outs, delta = _one_run(model_id, prefs, d, log)
            if i <= WARMUP_RUNS:
                continue
            times.append(elapsed)
            outputs = [str(p) for p in outs]
            for idx, val in delta.items():
                peak[idx] = max(peak.get(idx, 0.0), val)
        # MÉDIANE et non moyenne : un ralentissement ponctuel (le système qui
        # fait autre chose) déplace la moyenne, pas la médiane.
        times.sort()
        median = times[len(times) // 2]
        if log:
            log(f"[benchmark] {label} — median {median:.2f} s "
                f"(min {times[0]:.2f} · max {times[-1]:.2f})")
        return {"label": label, "ok": True, "seconds": round(median, 3),
                "steps": max(1, int(d.get("steps", 4))),
                "width": 512, "height": 512, "execution": "sd-cli",
                "runs_seconds": [round(x, 3) for x in times],
                "spread_seconds": round(times[-1] - times[0], 3),
                "peak_vram_gb_over_baseline": peak,
                "outputs": outputs}
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "ok": False,
                "seconds": round(time.perf_counter() - started, 3),
                "runs_seconds": [round(x, 3) for x in times],
                "peak_vram_gb_over_baseline": peak, "error": str(exc)}


def _pick_model(prefs: dict) -> str | None:
    models = {m.id: m for m in registry.load_base_models(prefs)}
    for wanted in ("krea2-turbo", "flux2-klein-9b"):
        model = models.get(wanted)
        if model and registry.model_is_ready(model):
            return wanted
    return next((m.id for m in models.values() if registry.model_is_ready(m)), None)


def _write_report(prefix: str, payload: dict) -> Path:
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = settings.OUTPUT_DIR / f"{prefix}-{stamp}.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return dest


def _winner(results: list[dict]) -> dict | None:
    """Le plus rapide — mais SEULEMENT si l'écart dépasse le bruit mesuré.

    Chaque profil rapporte sa dispersion (max - min sur les tirs mesurés). Si
    deux profils sont séparés par moins que ça, les départager serait tirer à
    pile ou face avec l'air sérieux. On garde alors le premier de la liste,
    c'est-à-dire le plus simple : `placement_candidates` les range du plus sûr
    au plus exotique.
    """
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None
    best = min(ok, key=lambda r: r["seconds"])
    noise = max(r.get("spread_seconds", 0.0) for r in ok)
    close = [r for r in ok if r["seconds"] - best["seconds"] <= noise]
    return close[0] if len(close) > 1 else best


def run_hardware_benchmark(model_id: str | None = None,
                           log: Callable[[str], None] | None = None,
                           cancel: Callable[[], bool] | None = None) -> Path:
    base = settings.load_prefs()
    selected = model_id or _pick_model(base)
    if not selected:
        raise RuntimeError("No model installed: download Krea 2 Turbo or Flux.2.")
    modes = placement_candidates(base)
    if not modes:
        raise RuntimeError("No NVIDIA GPU detected.")
    results = []
    for mode in modes:
        prefs = _benchmark_prefs(base, mode.prefs_patch)
        result = _run_case(selected, prefs, mode.label, log, cancel)
        result.update({"key": mode.key, "prefs_patch": mode.prefs_patch})
        results.append(result)
    winner = _winner(results)
    payload = {
        "schema": 2, "kind": "hardware-placement", "seed": BENCHMARK_SEED,
        "model_id": selected, "system": diagnostics.system_report(),
        "warmup_runs": WARMUP_RUNS, "measured_runs": MEASURED_RUNS,
        "results": results,
        "recommended_mode": winner.get("key") if winner else None,
        "recommended_prefs_patch": winner.get("prefs_patch") if winner else None,
    }
    return _write_report("hardware-benchmark", payload)


def compare_krea_variants(log: Callable[[str], None] | None = None,
                          cancel: Callable[[], bool] | None = None) -> Path:
    base = settings.load_prefs()
    results = []
    for model_id, label in (("krea2-turbo", "Krea 2 Turbo GGUF"),
                            ("krea2-turbo-int8", "Krea 2 Turbo INT8 ConvRot")):
        results.append({"model_id": model_id,
                        **_run_case(model_id, _benchmark_prefs(base, {}), label, log, cancel)})
    fastest = _winner(results)
    payload = {
        "schema": 2, "kind": "krea-quant-comparison", "seed": BENCHMARK_SEED,
        "system": diagnostics.system_report(),
        "warmup_runs": WARMUP_RUNS, "measured_runs": MEASURED_RUNS,
        "results": results,
        "fastest_model": fastest.get("model_id") if fastest else None,
        "quality_review_required": True,
    }
    return _write_report("krea-gguf-vs-int8", payload)


def apply_recommendation(report_path: str | Path) -> str:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    patch = data.get("recommended_prefs_patch")
    if not isinstance(patch, dict) or not patch:
        raise RuntimeError("This report holds no valid profile to apply.")
    prefs = _merge_prefs(settings.load_prefs(), patch)
    settings.save_prefs(prefs)
    return str(data.get("recommended_mode") or "measured profile")
