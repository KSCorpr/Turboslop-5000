"""Outils PyTorch installés à la demande (profondeur, détourage, upscale créatif).

Installation et exécution en sous-process (Python embarqué), pour ne pas
verrouiller les DLL de torch dans le process Gradio.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from .. import hardware, settings
from . import release_resident_engine
from ..i18n import t

TOOLS_DIR = settings.ROOT / "tools_repo"
DEPTH_MODEL_DIR = TOOLS_DIR / "depth" / "model"
BG_MODEL_DIR = TOOLS_DIR / "bg" / "model"
SAM_MODEL_DIR = TOOLS_DIR / "sam" / "model"
CLIP_MODEL_DIR = TOOLS_DIR / "clip" / "model"
ENHANCE_MODEL_DIR = TOOLS_DIR / "enhance" / "model"
DESCRIBE_MODEL_DIR = TOOLS_DIR / "describe" / "model"
FACE_MODEL_DIR = TOOLS_DIR / "face" / "model"
UPSCALE_DIR = TOOLS_DIR / "upscale"
UPSCALE_CKPT_DIR = UPSCALE_DIR / "checkpoints"   # checkpoints SDXL perso (.safetensors)
SEEDVR2_DIR = TOOLS_DIR / "seedvr2"
SEEDVR2_SOURCE_DIR = SEEDVR2_DIR / "source"
SEEDVR2_MODEL_DIR = SEEDVR2_DIR / "models"

_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")

# Registre des sous-process d'outils en cours (pour le bouton « Annuler »).
_ACTIVE: set[subprocess.Popen] = set()
_LOCK = threading.Lock()
_CANCELLED = False


def cancel() -> str:
    """Termine le(s) sous-process d'outil en cours (upscale, etc.)."""
    global _CANCELLED
    with _LOCK:
        procs = list(_ACTIVE)
    if not procs:
        return "No task running."
    _CANCELLED = True
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    return "⏹️ Task cancelled."


class ToolError(RuntimeError):
    pass


def _model_present(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))


def depth_is_installed() -> bool:
    return _model_present(DEPTH_MODEL_DIR)


def bg_is_installed() -> bool:
    return _model_present(BG_MODEL_DIR)


def sam_is_installed() -> bool:
    return _model_present(SAM_MODEL_DIR)


def clip_is_installed() -> bool:
    return _model_present(CLIP_MODEL_DIR)


def describe_is_installed() -> bool:
    return _model_present(DESCRIBE_MODEL_DIR)


def enhance_is_installed() -> bool:
    return _model_present(ENHANCE_MODEL_DIR)


# Détection et segmentation : COMMUNES à tous les restaurateurs, et aussi
# indispensables qu'eux. Sans elles, le premier clic échouerait au milieu du
# traitement au lieu d'afficher le bouton « Installer ».
FACE_SHARED_FILES = ("detection_Resnet50_Final.pth", "parsing_parsenet.pth")

# Les restaurateurs proposés : (fichier, libellé, licence).
#
# La LICENCE est affichée avec le modèle, pas enterrée dans un avertissement de
# bas de page. Elle décide de ce qu'on a le droit de faire du résultat, et deux
# des trois sont libres de toute restriction commerciale — c'est une
# information de premier plan pour qui vend ses images, pas une note de bas de
# page juridique.
FACE_MODELS: tuple[tuple[str, str, str], ...] = (
    ("GFPGANv1.4.pth",
     "GFPGAN v1.4 — preserves identity best",
     "Apache-2.0 · usage commercial libre"),
    ("RestoreFormer++.ckpt",
     "RestoreFormer++ — better on badly damaged photos",
     "Apache-2.0 · usage commercial libre"),
    ("codeformer.pth",
     "CodeFormer — adjustable fidelity dial",
     "S-Lab 1.0 · NON COMMERCIAL"),
)
FACE_MODEL_FILES = frozenset(f for f, _, _ in FACE_MODELS)
# Par défaut : le modèle libre qui garde le mieux le visage de la personne.
FACE_DEFAULT = "GFPGANv1.4.pth"
# Seul CodeFormer expose le « w » de fidélité ; pour les autres le curseur
# n'aurait aucun effet, et l'interface le dit plutôt que de le laisser croire.
FACE_FIDELITY_MODELS = frozenset({"codeformer.pth"})


def face_models_installed() -> list[tuple[str, str, str]]:
    """Restaurateurs réellement présents sur le disque."""
    return [m for m in FACE_MODELS if (FACE_MODEL_DIR / m[0]).is_file()]


def face_is_installed() -> bool:
    """Les briques communes ET au moins un restaurateur.

    « Au moins un » et pas « tous » : une installation interrompue après le
    premier modèle reste utilisable, et proposer de tout réinstaller pour un
    fichier manquant serait disproportionné.
    """
    shared = all((FACE_MODEL_DIR / n).is_file() for n in FACE_SHARED_FILES)
    return shared and bool(face_models_installed())


def upscale_is_installed() -> bool:
    base = UPSCALE_DIR / "sd_xl_base_1.0.safetensors"
    vae = UPSCALE_DIR / "vae"
    return base.is_file() and vae.is_dir() and any(vae.glob("*.safetensors"))


def upscale_cn_is_installed() -> bool:
    """ControlNet Tile présent (optionnel — verrouille la structure)."""
    cn = UPSCALE_DIR / "controlnet"
    return cn.is_dir() and any(cn.glob("*.safetensors"))


def _seedvr2_python() -> Path:
    return SEEDVR2_DIR / ("venv/Scripts/python.exe" if sys.platform == "win32"
                          else "venv/bin/python")


def seedvr2_is_installed() -> bool:
    return (_seedvr2_python().is_file()
            and (SEEDVR2_SOURCE_DIR / "inference_cli.py").is_file())


# Poids SeedVR2 : (libellé, fichier). Le CLI amont les télécharge lui-même et
# vérifie leur SHA-256 (dépôt AInVFX/SeedVR2_comfyUI) — cette liste est donc à
# la fois le menu de l'interface et l'unique liste blanche du sous-process.
SEEDVR2_MODELS: tuple[tuple[str, str], ...] = (
    ("3B Q8 — safe default, fastest", "seedvr2_ema_3b-Q8_0.gguf"),
    ("3B Q4 — fallback when memory runs short", "seedvr2_ema_3b-Q4_K_M.gguf"),
    ("7B Q4 — more detail, about 2× slower",
     "seedvr2_ema_7b-Q4_K_M.gguf"),
    ("7B Q4 “sharp” — sharpest (can harden grain)",
     "seedvr2_ema_7b_sharp-Q4_K_M.gguf"),
)
SEEDVR2_MODEL_FILES = frozenset(f for _, f in SEEDVR2_MODELS)


def _seedvr2_max_blocks(model: str) -> int:
    """Le 7B a 36 blocs de transformeur, le 3B en a 32."""
    return 36 if "_7b" in model else 32


def _seedvr2_site_packages() -> Path | None:
    lib = SEEDVR2_DIR / ("venv/Lib/site-packages" if sys.platform == "win32"
                         else "venv/lib")
    if sys.platform == "win32":
        return lib if lib.is_dir() else None
    if not lib.is_dir():
        return None
    for child in sorted(lib.glob("python3.*/site-packages")):
        return child
    return None


def seedvr2_attention_mode() -> str:
    """Meilleur noyau d'attention réellement installable ici.

    ``sdpa`` (PyTorch) marche partout. ``flash_attn_2`` / ``sageattn_2``
    exigent Ampère ou mieux (RTX 3060 oui, RTX 2080 Ti et GTX 1080 Ti non) ET
    le paquet correspondant dans le venv isolé de SeedVR2. On ne demande le
    noyau rapide que si les deux conditions sont vraies : le CLI amont saurait
    retomber sur ``sdpa``, mais autant ne pas polluer le journal d'un
    avertissement à chaque lancement.
    """
    site = _seedvr2_site_packages()
    if site is None:
        return "sdpa"
    index = _gen_gpu_index()
    gpus = {g.index: g for g in hardware.detect_gpus()}
    gpu = gpus.get(index) if index is not None else None
    cc = (gpu.compute_cap if gpu else "") or ""
    try:
        ampere_or_newer = float(cc) >= 8.0
    except ValueError:
        # Pilote trop ancien pour rapporter la capacité : l'architecture déduite
        # du nom reste un indice suffisant pour ne PAS tenter le noyau rapide.
        ampere_or_newer = bool(gpu) and gpu.arch in {
            "ampere", "ada", "hopper", "blackwell"}
    if not ampere_or_newer:
        return "sdpa"
    if (site / "flash_attn").is_dir():
        return "flash_attn_2"
    if (site / "sageattention").is_dir():
        return "sageattn_2"
    return "sdpa"


def list_upscale_checkpoints() -> list[tuple[str, str]]:
    """Checkpoints SDXL disponibles pour l'upscale créatif : (libellé, chemin).
    Le modèle de base + tout .safetensors déposé dans tools_repo/upscale/checkpoints/."""
    out: list[tuple[str, str]] = []
    base = UPSCALE_DIR / "sd_xl_base_1.0.safetensors"
    if base.is_file():
        out.append(("SDXL Base 1.0 (default)", str(base)))
    if UPSCALE_CKPT_DIR.is_dir():
        for p in sorted(UPSCALE_CKPT_DIR.glob("*.safetensors")):
            out.append((p.stem, str(p)))
    return out


def _install_stream(tool: str):
    """Installe un outil (depth|bg) en streamant le journal (pour l'UI)."""
    setup = settings.ROOT / "scripts" / "setup_tools.py"
    cmd = [sys.executable, str(setup), tool]
    buf: list[str] = [f"$ {' '.join(cmd)}", ""]
    yield "\n".join(buf)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT),
                            env=settings.child_env(),
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line.rstrip("\n"))
        yield "\n".join(buf[-500:])
    code = proc.wait()
    buf.append("")
    buf.append("✅ Installation complete." if code == 0
               else f"❌ Failed (code {code}). See the log above.")
    yield "\n".join(buf[-500:])


def install_depth_stream():
    yield from _install_stream("depth")


def install_bg_stream():
    yield from _install_stream("bg")


def install_sam_stream():
    yield from _install_stream("sam")


def install_clip_stream():
    yield from _install_stream("clip")


def install_enhance_stream():
    yield from _install_stream("enhance")


def install_describe_stream():
    yield from _install_stream("describe")


def install_face_stream():
    yield from _install_stream("face")


def install_upscale_stream():
    yield from _install_stream("upscale")


def install_seedvr2_stream():
    """Installe SeedVR2 dans son Python 3.12 isolé."""
    setup = settings.ROOT / "scripts" / "setup_seedvr2.py"
    cmd = [sys.executable, str(setup)]
    buf: list[str] = [f"$ {' '.join(cmd)}", ""]
    yield "\n".join(buf)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT),
                            env=settings.child_env(), encoding="utf-8",
                            errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line.rstrip("\n"))
        yield "\n".join(buf[-500:])
    code = proc.wait()
    buf += ["", "✅ SeedVR2 installation complete." if code == 0
            else f"❌ SeedVR2 failed (code {code}). See the log."]
    yield "\n".join(buf[-500:])


def _gen_gpu_index() -> int | None:
    """GPU de GÉNÉRATION d'images (Flux/Krea, upscale SDXL, depth/bg/SAM).
    Jamais le GPU secondaire dédié au texte."""
    prefs = settings.load_prefs()
    if prefs.get("gpu_index") is not None:
        return prefs["gpu_index"]
    prof = hardware.auto_profile()
    return prof.gpu.index if prof.gpu else None


def _text_gpu_index() -> int | None:
    """GPU pour le TEXTE (améliorateur de prompt). GPU secondaire si défini
    (ex. 1080 Ti), sinon le GPU de génération."""
    prefs = settings.load_prefs()
    if prefs.get("text_gpu_index") is not None:
        return prefs["text_gpu_index"]
    return _gen_gpu_index()


def _to_src(image: Image.Image | str | Path, prefix: str) -> Path:
    settings.ensure_dirs()
    if isinstance(image, (str, Path)):
        return Path(image)
    src = settings.TMP_DIR / f"{prefix}_src_{int(time.time()*1000)}.png"
    image.save(src)
    return src


def _run_tool(cmd: list[str], log: Callable[[str], None] | None,
              err_msg: str, gpu_index: int | None = None,
              cwd: Path | None = None, env: dict | None = None) -> None:
    global _CANCELLED
    # Un outil PyTorch qui démarre pendant que le moteur résident garde 8 Go de
    # modèle en VRAM, c'est un OOM. Le serveur rend la place ; il se rechargera
    # tout seul à la prochaine image.
    release_resident_engine("a Toolkit tool needs the GPU", log)
    run_env = env if env is not None else settings.child_env(gpu_index)
    if log:
        log("$ " + " ".join(cmd))
    _CANCELLED = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            cwd=str(cwd or settings.ROOT), env=run_env,
                            encoding="utf-8", errors="replace")
    with _LOCK:
        _ACTIVE.add(proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if log:
                log(line.rstrip("\n"))
        code = proc.wait()
    finally:
        with _LOCK:
            _ACTIVE.discard(proc)
    if _CANCELLED:
        raise ToolError("Cancelled by the user.")
    if code != 0:
        raise ToolError(err_msg)


def _collect(out_dir: Path, final_prefix: str, stamp: str) -> Path:
    produced = sorted(p for p in out_dir.rglob("*") if p.suffix.lower() in _IMG_EXT)
    if not produced:
        raise ToolError("No image produced (see the log).")
    final = settings.OUTPUT_DIR / f"{final_prefix}-{stamp}.png"
    Image.open(produced[0]).save(final)  # conserve l'alpha (RGBA) si présent
    return final


def depth_map(image, log: Callable[[str], None] | None = None) -> Path:
    if not depth_is_installed():
        raise ToolError("The depth tool is not installed (“Install” button in "
                        "the Toolkit).")
    src = _to_src(image, "depth")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"depth_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_depth.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(DEPTH_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir)]
    _run_tool(cmd, log, "Depth estimation failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "depth", stamp)


def face_restore(image, fidelity: float = 0.5, only_center: bool = False,
                 model: str = FACE_DEFAULT,
                 log: Callable[[str], None] | None = None) -> Path:
    """Restaure les visages d'une image (CodeFormer), sans toucher au reste.

    `fidelity` est le « w » de CodeFormer : 0 laisse le modèle reconstruire
    librement (visage très abîmé, mais le résultat peut ne plus être tout à
    fait la même personne), 1 colle au pixel d'origine. 0,5 est le réglage de
    référence. À passer APRÈS l'upscale : le visage est alors plus grand, donc
    mieux détecté et mieux recollé.
    """
    if not face_is_installed():
        raise ToolError("Face restoration is not installed (“Install” button "
                        "in the Toolkit).")
    if model not in FACE_MODEL_FILES:
        raise ToolError(f"Unknown restoration model: {model}")
    weights = FACE_MODEL_DIR / model
    if not weights.is_file():
        raise ToolError(f"The model “{model}” is not downloaded. "
                        "Run the installation again from the Toolkit.")
    src = _to_src(image, "face")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"face_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_face.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(FACE_MODEL_DIR),
           "--weights", weights.name,
           "--input", str(src), "--output-dir", str(out_dir),
           "--fidelity", f"{min(1.0, max(0.0, float(fidelity))):.2f}"]
    if only_center:
        cmd.append("--only-center")
    _run_tool(cmd, log, "Face restoration failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "face", stamp)


def bg_remove(image, log: Callable[[str], None] | None = None) -> Path:
    if not bg_is_installed():
        raise ToolError("The background removal tool is not installed "
                        "(“Install” button in the Toolkit).")
    src = _to_src(image, "nobg")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"nobg_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_rembg.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(BG_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir)]
    _run_tool(cmd, log, "Background removal failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "nobg", stamp)


def sam_segment(image, x: int, y: int,
                log: Callable[[str], None] | None = None) -> tuple[Path, Path | None]:
    """Segment Anything au point (x, y). Renvoie (découpage PNG transparent,
    aperçu overlay) — l'overlay montre la zone sélectionnée en surbrillance."""
    if not sam_is_installed():
        raise ToolError("Segment Anything is not installed (“Install” button "
                        "in the Toolkit).")
    src = _to_src(image, "sam")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"sam_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = settings.TMP_DIR / f"sam_overlay_{stamp}.png"
    runner = settings.ROOT / "scripts" / "tools" / "run_sam.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(SAM_MODEL_DIR),
           "--input", str(src), "--output-dir", str(out_dir),
           "--x", str(int(x)), "--y", str(int(y)),
           "--overlay-path", str(overlay)]
    _run_tool(cmd, log, "Segmentation failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "sam", stamp), (overlay if overlay.exists() else None)


def _layers_to_files(src: Path, masks: list, names: list[str], stamp: str,
                     want_psd: bool, want_png: bool,
                     log: Callable[[str], None] | None = None) -> list[Path]:
    """Assemble des masques en PSD et/ou en PNG transparents séparés.

    Volontairement HORS du sous-process : l'écriture PSD n'a besoin ni de torch
    ni de transformers, et la faire ici évite d'imposer l'add-on à qui veut
    seulement rassembler des calques déjà découpés (mode manuel).
    """
    import numpy as np
    from . import masks as mask_utils
    from . import psd as psd_writer

    rgb = np.asarray(Image.open(src).convert("RGB"))
    h, w = rgb.shape[:2]
    out: list[Path] = []

    # Le fond, c'est l'image entière : même si une zone est détourée par-dessus,
    # on ne laisse jamais un trou dans le fichier final.
    layers = [(t("Background (full image)"),
               np.dstack([rgb, np.full((h, w), 255, "uint8")]))]
    for name, m in zip(names, masks):
        # Bord ADOUCI : un masque binaire collé tel quel donne ce contour en
        # marches d'escalier qui trahit le détourage automatique. Un pixel de
        # transition suffit à le faire disparaître, sans manger la zone.
        alpha = mask_utils.feather_alpha(m, radius=1)
        layers.append((name, np.dstack([rgb, alpha])))

    if want_psd:
        dest = settings.OUTPUT_DIR / f"layers-{stamp}.psd"
        psd_writer.write_psd(dest, rgb, layers)
        size = dest.stat().st_size / (1024 * 1024)
        if log:
            log(f"[layers] PSD written: {dest.name} ({size:.1f} MB, "
                f"{len(layers)} layers)")
        out.append(dest)
    if want_png:
        folder = settings.OUTPUT_DIR / f"layers-{stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        for i, (name, rgba) in enumerate(layers):
            safe = "".join(c if (c.isalnum() or c in " -_") else "_"
                           for c in name).strip() or f"layer{i}"
            Image.fromarray(rgba, "RGBA").save(folder / f"{i:02d}_{safe}.png")
        if log:
            log(f"[layers] {len(layers)} transparent PNGs: {folder.name}/")
        out.append(folder)
    return out


def image_to_layers(image, points_per_side: int = 12, max_layers: int = 24,
                    min_area_pct: float = 0.4, want_psd: bool = True,
                    want_png: bool = False,
                    log: Callable[[str], None] | None = None) -> list[Path]:
    """AUTOMATIQUE : SAM balaie l'image, on en tire des calques triés en
    profondeur, puis on assemble le PSD.

    Ce que ça ne fait PAS, et qu'il faut savoir avant de cliquer : les calques
    sont des DÉCOUPES à plat. Déplacer un objet révèle un trou, parce que le
    fond derrière lui n'a jamais existé. C'est utile pour masquer, retoucher une
    zone ou exporter un élément — pas pour recomposer la scène.
    """
    if not sam_is_installed():
        raise ToolError("Segment Anything is not installed (“Install” button "
                        "in the Toolkit).")
    import numpy as np

    src = _to_src(image, "layers")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = settings.TMP_DIR / f"layers_{stamp}"
    runner = settings.ROOT / "scripts" / "tools" / "run_layers.py"
    cmd = [sys.executable, str(runner),
           "--sam-dir", str(SAM_MODEL_DIR),
           "--input", str(src), "--output-dir", str(work),
           "--points-per-side", str(int(points_per_side)),
           "--max-layers", str(int(max_layers)),
           "--min-area", f"{max(0.0005, float(min_area_pct) / 100.0):g}"]
    # La profondeur n'est PAS requise : sans elle on trie par surface, et le
    # runner le dit. L'exiger transformerait un add-on optionnel en dépendance.
    if depth_is_installed():
        cmd += ["--depth-dir", str(DEPTH_MODEL_DIR)]
    # CLIP : étiquetage sémantique, fusion des morceaux d'un même objet, et
    # rejet des zones qui ne ressemblent à rien. Facultatif comme la profondeur.
    if clip_is_installed():
        cmd += ["--clip-dir", str(CLIP_MODEL_DIR)]
    _run_tool(cmd, log, "Splitting into layers failed (see the log).",
              gpu_index=_gen_gpu_index())

    manifest = work / "layers.json"
    if not manifest.is_file():
        raise ToolError("No layer produced (see the log).")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    from . import masks as mask_utils
    rgb = np.asarray(Image.open(src).convert("RGB"))
    entries = data.get("masks", [])
    masks, names = [], []
    for i, entry in enumerate(entries):
        m = np.asarray(Image.open(work / entry["file"]).convert("L")) > 127
        masks.append(m)
        # Nom SÉMANTIQUE quand CLIP a pu étiqueter la zone (« véhicule »,
        # « ciel »), descriptif sinon (plan, position, couleur). Dans les deux
        # cas on garde plan et taille : ce sont eux qui situent le calque dans
        # la pile.
        base = mask_utils.describe(m, rgb, i, len(entries))
        lab = (entry.get("label") or "").strip()
        names.append(f"{lab} · {base}" if lab else base)
    if not masks:
        raise ToolError(
            "No usable region found. Try more probe points, or a lower "
            "minimum area.")
    return _layers_to_files(src, masks, names, stamp, want_psd, want_png, log)


def masks_to_layers(image, masks: list, names: list[str] | None = None,
                    want_psd: bool = True, want_png: bool = False,
                    log: Callable[[str], None] | None = None) -> list[Path]:
    """MANUEL : assemble des zones choisies à la main (clics SAM successifs).

    Les masques arrivent déjà segmentés : aucun modèle n'est chargé ici.
    """
    if not masks:
        raise ToolError("No region selected — click on the image first to "
                        "create layers.")
    src = _to_src(image, "layers")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    names = names or [f"Zone {i + 1}" for i in range(len(masks))]
    return _layers_to_files(src, masks, names, stamp, want_psd, want_png, log)


ENHANCE_STYLES = ("generic", "krea2", "xanax")


# --------------------------------------------------------------------------- #
#  Améliorateur de prompt (petit LLM instruct)
# --------------------------------------------------------------------------- #
def enhance_prompt_variants(prompt: str, style: str = "generic",
                            level: str = "medium", variants: int = 1,
                            style_constraint: str = "",
                            log: Callable[[str], None] | None = None) -> list[str]:
    """Améliore un prompt brut via un petit LLM instruct (transformers).

    `style` choisit le system prompt : "krea2" (guide Krea) ou "generic".
    `variants` demande plusieurs PROPOSITIONS en un seul chargement du modèle.
    `style_constraint` transmet le préréglage de style actif, pour que le LLM
    écrive AVEC lui. S'exécute en sous-process (chargé puis déchargé : aucun
    conflit VRAM avec sd.cpp)."""
    if not enhance_is_installed():
        raise ToolError("The prompt improver is not installed (the “✨ "
                        "Improve” section of the generation tab).")
    if not (prompt or "").strip():
        raise ToolError("Enter a prompt to enhance first.")
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = settings.TMP_DIR / f"enhance_{stamp}.json"
    runner = settings.ROOT / "scripts" / "tools" / "run_enhance.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(ENHANCE_MODEL_DIR),
           "--prompt", prompt, "--output", str(out_file),
           "--style", style if style in ENHANCE_STYLES else "generic",
           "--level", level if level in ("light", "medium", "strong") else "medium",
           "--variants", str(max(1, min(8, int(variants or 1))))]
    # Préréglage de style actif : le LLM doit écrire AVEC lui, pas contre lui.
    if (style_constraint or "").strip():
        cmd += ["--style-constraint", style_constraint.strip()]
    # Améliorateur = TEXTE → GPU secondaire dédié au texte (ex. 1080 Ti).
    _run_tool(cmd, log, "Prompt improvement failed (see the log).",
              gpu_index=_text_gpu_index())
    try:
        raw = out_file.read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    out: list[str] = []
    if raw:
        try:
            data = json.loads(raw)
            out = [str(x).strip() for x in data if str(x).strip()]
        except (ValueError, TypeError):
            out = [raw]           # repli : ancien format texte brut
    if not out:
        raise ToolError("The improver returned no text (see the log).")
    return out


# --------------------------------------------------------------------------- #
#  Image → prompt (modèle de vision-langage)
# --------------------------------------------------------------------------- #
# Trois intentions RÉELLEMENT différentes, pas trois réglages du même curseur :
#   full  -> refaire cette image ailleurs (sujet + style)
#   style -> appliquer CE rendu à un AUTRE sujet (style seul, sujet interdit)
#   plain -> savoir ce qu'il y a dedans, en français courant côté lecture
DESCRIBE_MODES = ("full", "style", "plain")


def image_to_prompt(image, mode: str = "full", variants: int = 1,
                    log: Callable[[str], None] | None = None) -> list[str]:
    """Lit une image et en écrit un PROMPT (pas une légende).

    Tourne en sous-process, comme l'améliorateur : le modèle est chargé puis
    déchargé, donc rien ne reste en VRAM pendant la génération sd.cpp. Et comme
    l'améliorateur, c'est du TEXTE : il part sur le GPU dédié au texte quand il
    y en a un (la 1080 Ti, par exemple), ce qui laisse la carte de génération
    tranquille.
    """
    if not describe_is_installed():
        raise ToolError("The “Image → prompt” module is not installed "
                        "(install button in its own tab).")
    src = _to_src(image, "describe")
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = settings.TMP_DIR / f"describe_{stamp}.json"
    runner = settings.ROOT / "scripts" / "tools" / "run_describe.py"
    cmd = [sys.executable, str(runner), "--model-dir", str(DESCRIBE_MODEL_DIR),
           "--image", str(src), "--output", str(out_file),
           "--mode", mode if mode in DESCRIBE_MODES else "full",
           "--variants", str(max(1, min(4, int(variants or 1))))]
    _run_tool(cmd, log, "Reading the image failed (see the log).",
              gpu_index=_text_gpu_index())
    try:
        raw = out_file.read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    out: list[str] = []
    if raw:
        try:
            out = [str(x).strip() for x in json.loads(raw) if str(x).strip()]
        except (ValueError, TypeError):
            out = [raw]
    if not out:
        raise ToolError("The model returned no text (see the log).")
    return out


def enhance_prompt(prompt: str, style: str = "generic", level: str = "medium",
                   log: Callable[[str], None] | None = None) -> str:
    """Une seule proposition (compatibilité)."""
    return enhance_prompt_variants(prompt, style=style, level=level,
                                   variants=1, log=log)[0]


def ultimate_upscale(image, scale: float = 2.0, prompt: str = "",
                     negative: str = "",
                     denoise: float = 0.35, steps: int = 24, cfg: float = 6.0,
                     tile: int = 1024, overlap: int = 128,
                     use_controlnet: bool = False, cn_scale: float = 0.6,
                     base_model: str | None = None, integrated_vae: bool = False,
                     esrgan_model: str | None = None,
                     preview_path: Path | None = None,
                     log: Callable[[str], None] | None = None) -> Path:
    """Upscale créatif tuilé « Ultimate SD Upscale » (SDXL img2img résident).

    Pré-agrandit puis raffine tuile par tuile à faible débruitage. Options :
    `base_model` (checkpoint SDXL ; défaut = base 1.0), `integrated_vae` (utiliser
    la VAE du checkpoint au lieu de la fp16-fix externe), `esrgan_model` (pré-
    agrandir avec un ESRGAN GGUF plutôt qu'en Lanczos), `use_controlnet`."""
    if not upscale_is_installed():
        raise ToolError("The creative SDXL upscale is not installed "
                        "(“Install” button in Toolkit → Upscale).")
    base = Path(base_model) if base_model else UPSCALE_DIR / "sd_xl_base_1.0.safetensors"
    if not base.is_file():
        raise ToolError(f"Checkpoint SDXL introuvable : {base}")
    from PIL import Image as _PILImage
    src = _to_src(image, "usdu")
    with _PILImage.open(src) as _im:
        ow, oh = _im.size
    tw = max(8, int(round(ow * scale / 8)) * 8)
    th = max(8, int(round(oh * scale / 8)) * 8)

    # Pré-agrandissement ESRGAN optionnel (sd.cpp) : base plus nette que Lanczos.
    #
    # UNE SEULE passe, jamais deux. Enchaîner un ESRGAN sur sa propre sortie est
    # le meilleur moyen de fabriquer les artefacts et l'aliasing qu'on cherche à
    # éviter : la 2ᵉ passe prend les hautes fréquences INVENTÉES par la 1ʳᵉ pour
    # du détail réel et les ré-accentue, ce qui transforme un léger ringing en
    # marches d'escalier franches sur les diagonales. Si une passe ne suffit pas
    # à atteindre la cible, le runner complète en Lanczos : c'est plus doux,
    # mais c'est propre — et c'est précisément le rôle du raffinage SDXL qui
    # suit de redonner le détail. Une base molle se rattrape, une base crénelée
    # non : SDXL fige les créneaux au lieu de les corriger.
    #
    # Quand le facteur du modèle DÉPASSE la cible (un ×4 pour un ×2), la
    # réduction faite ensuite par le runner est un suréchantillonnage : c'est le
    # cas le plus propre possible, l'aliasing est moyenné à la baisse.
    inp = src
    if esrgan_model:
        from .. import registry
        from . import generate as gen_engine
        factor = registry.upscaler_factor(esrgan_model)
        try:
            if log:
                log(f"ESRGAN pre-enlargement “{esrgan_model}” (×{factor}, "
                    "one pass)…")
                if factor > float(scale):
                    log(f"[usdu] ×{factor} for a ×{scale:g} target: the "
                        "downscale that follows acts as supersampling "
                        "(free anti-aliasing).")
                elif factor < float(scale):
                    # Honnêteté : le runner complète en Lanczos, donc la base
                    # sera plus douce — mais pas crénelée, et SDXL la reprend.
                    log(f"[usdu] ×{factor} < target ×{scale:g}: the rest is "
                        "filled in with Lanczos (a softer base, which the "
                        "SDXL refine pass re-details). For crisp line art "
                        "from the start, pick a ×"
                        f"{int(-(-float(scale) // 1))} model "
                        "or aim for a lower factor.")
            inp = gen_engine.upscale_image(src, esrgan_model, repeats=1,
                                           log=log)
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"[usdu] ESRGAN failed ({exc}) → repli Lanczos.")
            inp = src

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"usdu_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = settings.ROOT / "scripts" / "tools" / "run_ultimate_upscale.py"
    cmd = [sys.executable, str(runner),
           "--base-model", str(base),
           "--input", str(inp), "--output-dir", str(out_dir),
           "--width", str(tw), "--height", str(th),
           "--scale", str(float(scale)), "--denoise", str(float(denoise)),
           "--steps", str(int(steps)), "--cfg", str(float(cfg)),
           "--tile", str(int(tile)), "--overlap", str(int(overlap)),
           "--prompt", prompt or ""]
    # Négatif : vide = le défaut PHOTO du runner. Le préréglage dessin en
    # fournit un autre (anti-grain, anti-photoréalisme sur les aplats).
    if negative:
        cmd += ["--negative", negative]
    # VAE : externe fp16-fix (défaut) sauf si on veut celle intégrée au checkpoint.
    if not integrated_vae and (UPSCALE_DIR / "vae").is_dir():
        cmd += ["--vae", str(UPSCALE_DIR / "vae")]
    if use_controlnet and upscale_cn_is_installed():
        cmd += ["--controlnet", str(UPSCALE_DIR / "controlnet"),
                "--cn-scale", str(float(cn_scale))]
    if preview_path:
        cmd += ["--preview-path", str(preview_path)]
    # VRAM serrée (< 12 Go) → offload CPU du modèle pour éviter l'OOM.
    prof = hardware.auto_profile(settings.load_prefs().get("gpu_index"))
    if prof.gpu and prof.gpu.vram_gb < 12:
        cmd.append("--low-vram")
    # Upscale SDXL = génération d'IMAGES → GPU de génération (jamais le secondaire).
    _run_tool(cmd, log, "The creative SDXL upscale failed (see the log).",
              gpu_index=_gen_gpu_index())
    return _collect(out_dir, "usdu", stamp)


SPANDREL_DIR = TOOLS_DIR / "spandrel"


def spandrel_models():
    return sorted(p.name for p in SPANDREL_DIR.glob("*")
                  if p.suffix.lower() in {".pth", ".safetensors"})


def install_spandrel_stream():
    yield from _install_stream("spandrel")


def import_spandrel_model(path):
    import shutil
    import uuid
    source = Path(path)
    if source.suffix.lower() not in {".pth", ".safetensors"}:
        raise ToolError("Choose a .pth or .safetensors model.")
    SPANDREL_DIR.mkdir(parents=True, exist_ok=True)
    dest = SPANDREL_DIR / source.name
    partial = SPANDREL_DIR / f"{uuid.uuid4().hex}.part"
    try:
        shutil.copyfile(source, partial)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest.name


def spandrel_upscale(image, model_name, full_precision=False, log=None):
    import tempfile
    import uuid
    from ..imaging.upscale import read_source
    if model_name not in spandrel_models():
        raise ToolError("Install or import a native x4 model first.")
    settings.ensure_dirs()
    output = settings.OUTPUT_DIR / f"upscale4x-{uuid.uuid4().hex}.png"
    runner = settings.ROOT / "scripts" / "tools" / "run_spandrel.py"
    with tempfile.TemporaryDirectory(prefix="spandrel-", dir=settings.TMP_DIR) as job:
        src = Path(job) / "input.png"
        source = read_source(image)
        source.save(src, icc_profile=source.info.get("icc_profile"))
        cmd = [sys.executable, str(runner), "--input", str(src),
               "--output", str(output), "--model", str(SPANDREL_DIR / model_name)]
        if full_precision:
            cmd.append("--full-precision")
        _run_tool(cmd, log, "Spandrel failed. See the log; install the optional "
                  "engine first. For invalid pixels, enable full precision.",
                  gpu_index=_gen_gpu_index())
    with Image.open(output) as result:
        if result.size != (source.width*4, source.height*4):
            output.unlink(missing_ok=True)
            raise ToolError("Invalid x4 output dimensions.")
    return output


def seedvr2_upscale(image, resolution: int = 2048,
                    model: str = "seedvr2_ema_3b-Q8_0.gguf",
                    blocks_to_swap: int = 16, tile: int = 1024,
                    overlap: int = 128, offload: str = "secondary",
                    color_correction: str = "wavelet",
                    log: Callable[[str], None] | None = None) -> Path:
    """Restauration/upscale SeedVR2 pour image unique.

    Le calcul reste sur le GPU principal. Avec ``secondary``, CUDA remappe les
    cartes en ``[principal, secondaire]`` : SeedVR2 calcule sur cuda:0 et stocke
    ses blocs/VAE sur cuda:1, sans lancer son mode multi-GPU vidéo.
    """
    if not seedvr2_is_installed():
        raise ToolError("SeedVR2 is not installed (Install button in the Toolkit).")
    if model not in SEEDVR2_MODEL_FILES:
        raise ToolError(f"SeedVR2 model not allowed: {model}")
    if color_correction not in {"wavelet", "lab", "wavelet_adaptive", "none"}:
        color_correction = "wavelet"

    src = _to_src(image, "seedvr2")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = settings.OUTPUT_DIR / f"seedvr2-{stamp}.png"
    py = _seedvr2_python()
    cli = SEEDVR2_SOURCE_DIR / "inference_cli.py"
    SEEDVR2_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prefs = settings.load_prefs()
    main_gpu = _gen_gpu_index()
    secondary = prefs.get("encoder_gpu_index")
    if secondary is None or secondary == main_gpu:
        candidate = prefs.get("text_gpu_index")
        secondary = candidate if candidate != main_gpu else None

    run_env = settings.child_env()
    offload_device = "cpu"
    if offload == "secondary" and main_gpu is not None and secondary is not None:
        # cuda:0 = GPU principal, cuda:1 = GPU secondaire dans le sous-process.
        run_env["CUDA_VISIBLE_DEVICES"] = f"{main_gpu},{secondary}"
        run_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        offload_device = "1"
        if log:
            log(f"SeedVR2 : calcul GPU #{main_gpu}, offload GPU #{secondary}.")
    elif main_gpu is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = str(main_gpu)
        offload_device = "none" if offload == "none" else "cpu"
    blocks = max(0, min(_seedvr2_max_blocks(model), int(blocks_to_swap)))
    if offload_device == "none":
        blocks = 0
    attention = seedvr2_attention_mode()
    if log and attention != "sdpa":
        log(f"SeedVR2: accelerated attention ({attention}).")

    cmd = [
        str(py), str(cli), str(src), "--output", str(output),
        "--output_format", "png", "--model_dir", str(SEEDVR2_MODEL_DIR),
        "--dit_model", model, "--resolution", str(max(512, int(resolution))),
        "--max_resolution", str(max(512, int(resolution))), "--batch_size", "1",
        "--color_correction", color_correction,
        "--dit_offload_device", offload_device,
        "--vae_offload_device", offload_device,
        "--tensor_offload_device", offload_device,
        "--blocks_to_swap", str(blocks),
        "--vae_encode_tiled", "--vae_decode_tiled",
        "--vae_encode_tile_size", str(max(512, int(tile))),
        "--vae_decode_tile_size", str(max(512, int(tile))),
        "--vae_encode_tile_overlap", str(max(64, int(overlap))),
        "--vae_decode_tile_overlap", str(max(64, int(overlap))),
        "--attention_mode", attention, "--debug",
    ]
    if blocks:
        cmd.append("--swap_io_components")
    _run_tool(cmd, log, "SeedVR2 failed (see the log).",
              cwd=SEEDVR2_SOURCE_DIR, env=run_env)
    if not output.is_file() or output.stat().st_size == 0:
        raise ToolError("SeedVR2 produced no image.")
    return output


def seedvr2_batch(images, resolution: int = 2048,
                  model: str = "seedvr2_ema_3b-Q8_0.gguf",
                  blocks_to_swap: int = 16, tile: int = 1024,
                  overlap: int = 128, offload: str = "secondary",
                  color_correction: str = "wavelet",
                  log: Callable[[str], None] | None = None) -> list[Path]:
    """Restaure plusieurs images dans UNE invocation SeedVR2.

    Le CLI amont sait traiter un dossier avec `--cache_dit --cache_vae` : le
    modèle est chargé une fois puis réutilisé, au lieu de payer son chargement
    pour chaque image. Les originaux ne sont jamais modifiés.
    """
    if not seedvr2_is_installed():
        raise ToolError("SeedVR2 is not installed (Install button in the Toolkit).")
    if model not in SEEDVR2_MODEL_FILES:
        raise ToolError(f"SeedVR2 model not allowed: {model}")
    if color_correction not in {"wavelet", "lab", "wavelet_adaptive", "none"}:
        color_correction = "wavelet"

    raw = list(images or [])
    sources: list[Path] = []
    for item in raw:
        # pathlib.Path possède lui aussi un attribut ``name`` (le basename) :
        # ne pas le confondre avec le chemin temporaire porté par UploadedFile.
        p = item if isinstance(item, Path) else Path(getattr(item, "name", item))
        if p.is_file() and p.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            sources.append(p)
    if not sources:
        raise ToolError("No usable image in the batch.")

    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    batch_root = settings.TMP_DIR / f"seedvr2-batch-{stamp}-{int(time.time()*1000)%1000:03d}"
    input_dir = batch_root / "input"
    output_dir = settings.OUTPUT_DIR / f"seedvr2-batch-{stamp}"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for index, src in enumerate(sources, 1):
        # Préfixe stable : deux dossiers peuvent contenir le même nom de fichier.
        shutil.copy2(src, input_dir / f"{index:04d}-{src.name}")

    py = _seedvr2_python()
    cli = SEEDVR2_SOURCE_DIR / "inference_cli.py"
    SEEDVR2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prefs = settings.load_prefs()
    main_gpu = _gen_gpu_index()
    secondary = prefs.get("encoder_gpu_index")
    if secondary is None or secondary == main_gpu:
        candidate = prefs.get("text_gpu_index")
        secondary = candidate if candidate != main_gpu else None

    run_env = settings.child_env()
    offload_device = "cpu"
    if offload == "secondary" and main_gpu is not None and secondary is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = f"{main_gpu},{secondary}"
        run_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        offload_device = "1"
        if log:
            log(f"SeedVR2 lot : calcul GPU #{main_gpu}, spare GPU #{secondary}.")
    elif main_gpu is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = str(main_gpu)
        offload_device = "none" if offload == "none" else "cpu"
    # Les caches de modèle du mode dossier exigent un backend d'offload.
    if offload_device == "none":
        offload_device = "cpu"
        if log:
            log("SeedVR2 batch: RAM offload enabled so the model stays cached.")

    blocks = max(0, min(_seedvr2_max_blocks(model), int(blocks_to_swap)))
    attention = seedvr2_attention_mode()
    if log and attention != "sdpa":
        log(f"SeedVR2: accelerated attention ({attention}).")
    cmd = [
        str(py), str(cli), str(input_dir), "--output", str(output_dir),
        "--output_format", "png", "--model_dir", str(SEEDVR2_MODEL_DIR),
        "--dit_model", model, "--resolution", str(max(512, int(resolution))),
        "--max_resolution", str(max(512, int(resolution))), "--batch_size", "1",
        "--color_correction", color_correction,
        "--dit_offload_device", offload_device,
        "--vae_offload_device", offload_device,
        "--tensor_offload_device", offload_device,
        "--blocks_to_swap", str(blocks), "--cache_dit", "--cache_vae",
        "--vae_encode_tiled", "--vae_decode_tiled",
        "--vae_encode_tile_size", str(max(512, int(tile))),
        "--vae_decode_tile_size", str(max(512, int(tile))),
        "--vae_encode_tile_overlap", str(max(64, int(overlap))),
        "--vae_decode_tile_overlap", str(max(64, int(overlap))),
        "--attention_mode", attention, "--debug",
    ]
    if blocks:
        cmd.append("--swap_io_components")
    try:
        if log:
            log(f"SeedVR2 : {len(sources)} image(s), a single model load.")
        _run_tool(cmd, log, "The SeedVR2 batch failed (see the log).",
                  cwd=SEEDVR2_SOURCE_DIR, env=run_env)
    finally:
        shutil.rmtree(batch_root, ignore_errors=True)
    outputs = sorted(p for p in output_dir.rglob("*")
                     if p.is_file() and p.suffix.lower() == ".png")
    if not outputs:
        raise ToolError("SeedVR2 produced no image for this batch.")
    return outputs
