"""Exécuteur stable-diffusion.cpp : construction et lancement des commandes sd-cli."""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .. import settings
from . import release_resident_engine


class EngineError(RuntimeError):
    pass


class VramError(EngineError):
    """Échec par manque de mémoire GPU.

    Distinguée du reste parce que c'est le SEUL échec qu'il vaille la peine de
    retenter : à l'identique il se reproduirait, mais en plus petit il peut
    passer. Les appelants qui savent réduire (la passe HD) s'appuient dessus."""


# Registre des process sd-cli en cours, pour pouvoir les annuler.
_ACTIVE: set[subprocess.Popen] = set()
_LOCK = threading.Lock()
_CANCELLED = False


def cancel_active() -> str:
    """Termine tous les process sd-cli en cours (bouton « Annuler »)."""
    global _CANCELLED
    with _LOCK:
        procs = list(_ACTIVE)
    if not procs:
        return "No generation running."
    _CANCELLED = True
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    return "⏹️ Generation cancelled."


def was_cancelled() -> bool:
    """Le dernier `run()` s'est-il arrêté parce que l'utilisateur a annulé ?

    Sert aux appelants qui réessaient après un échec : une annulation ne doit
    évidemment pas déclencher une nouvelle tentative."""
    return _CANCELLED


@dataclass
class HiresParams:
    """« Highres fix » natif de sd.cpp : agrandir puis re-débruiter EN ENTIER.

    Le second passage se fait sur l'image complète, pas tuile par tuile — donc
    aucune couture possible, et le modèle voit la scène entière au lieu d'un
    carré de 1024 px. C'est la différence de fond avec l'upscale créatif SDXL.

    `upscaler` : « Latent », « Lanczos », « Nearest » (intégrés à sd.cpp) ou le
    NOM DE FICHIER d'un ESRGAN présent dans `upscalers_dir`.
    """
    scale: float = 2.0
    upscaler: str = "Latent"
    upscalers_dir: Path | None = None
    denoise: float = 0.4
    steps: int = 0             # 0 = réutiliser --steps
    tile_size: int = 0         # 0 = laisser le défaut sd.cpp (ESRGAN seulement)
    # Taille finale EXPLICITE. Renseignée, elle prime sur `scale` côté sd.cpp —
    # et surtout elle garantit que la taille annoncée dans le journal est celle
    # qui sort réellement, au lieu d'un arrondi calculé deux fois.
    target_width: int = 0
    target_height: int = 0


@dataclass
class GenRequest:
    diffusion_model: Path | None = None   # modèle de diffusion seul (GGUF flow)
    vae: Path | None = None
    model_path: Path | None = None        # checkpoint complet -> -m
    text_encoder: Path | None = None       # --llm (modèles à encodeur LLM)
    # --llm_vision : projecteur vision (mmproj) de l'encodeur — permet à
    # Qwen3-VL de « voir » l'image de référence (édition Krea 2 / Ostris Edit).
    llm_vision: Path | None = None
    t5xxl: Path | None = None              # --t5xxl (FLUX.1, etc.)
    clip_l: Path | None = None             # --clip_l
    uncond_model: Path | None = None
    extra_flags: list[str] = field(default_factory=list)
    prompt: str = ""
    negative: str = ""
    steps: int = 8
    cfg_scale: float = 1.0
    sampler: str = "euler"
    schedule: str = ""          # vide = laisser le scheduler par défaut du modèle
    flow_shift: float = 0.0     # 0 = auto (ne pas passer --flow-shift)
    width: int = 1024
    height: int = 1024
    seed: int = -1
    batch_count: int = 1
    init_image: Path | None = None     # img2img classique (-i + --strength)
    strength: float = 0.6
    # Masque d'inpainting (blanc = à régénérer, noir = à conserver). L'option
    # est DÉTECTÉE sur le binaire (mask_flag) : ignorée s'il ne la connaît pas.
    mask_image: Path | None = None
    # édition (-r / --ref-image, Flux.2) : un chemin OU une liste (multi-référence)
    ref_image: "Path | list[Path] | None" = None
    lora_dir: Path | None = None       # --lora-model-dir
    preview_path: Path | None = None   # aperçu temps réel (--preview proj)
    flags: dict[str, bool] = field(default_factory=dict)
    gpu_index: int | None = None
    # EXPÉRIMENTAL : place l'encodeur de texte sur un autre GPU (ex. 1080 Ti)
    # via --backend te=cudaX. None = encodeur sur le GPU principal / RAM.
    encoder_gpu_index: int | None = None
    # Backend de RÉSIDENCE des poids, distinct du backend de CALCUL ci-dessus.
    # Ex. "diffusion=cuda0,vae=cuda0,te=cuda1".
    params_backend: str = ""
    # EXPÉRIMENTAL : répartition auto du modèle sur tous les GPU (--auto-fit).
    # Prioritaire sur le split d'encodeur (auto-fit remplace --backend).
    auto_fit: bool = False
    split_mode: str = ""               # --split-mode : "layer" | "row" (vide=défaut)
    # Accélération par cache (docs/caching.md) : réutilise les calculs entre pas.
    cache_mode: str = ""               # easycache | dbcache | taylorseer | …
    cache_option: str = ""             # ex. "threshold=0.2"
    hires: "HiresParams | None" = None  # passe HD native (voir HiresParams)
    # Budget VRAM pour l'exécution SEGMENTÉE (voir max_vram_arg). Vide = option
    # non envoyée, sd.cpp alloue son graphe d'un bloc comme avant.
    max_vram: str = ""
    stream_layers: bool = False


# --------------------------------------------------------------------------- #
#  Découverte des options réellement supportées par LE binaire installé.
#
#  L'orthographe des options bouge d'une version de sd.cpp à l'autre (et notre
#  build maison peut différer de l'officielle). Plutôt que de coder en dur un
#  nom d'option et d'échouer à l'exécution, on lit « sd-cli -h » une fois et on
#  s'adapte. Coût : un lancement de quelques millisecondes, mis en cache.
# --------------------------------------------------------------------------- #
_HELP_CACHE: dict[tuple, str] = {}


def binary_help(sd_cli: Path | None) -> str:
    """Read help once per binary revision; a failed probe can be retried."""
    if not sd_cli or not Path(sd_cli).is_file():
        return ""
    p = Path(sd_cli)
    try:
        stat = p.stat()
        key = (str(p), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns)
    except OSError:
        return ""
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    text = ""
    try:
        # -h sort parfois sur stderr et/ou avec un code de retour non nul.
        r = subprocess.run([str(p), "-h"], capture_output=True, text=True,
                           timeout=30, errors="replace")
        text = (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:  # noqa: BLE001
        text = ""
    if "--" in text:
        # Updates should not accumulate old help/version entries forever.
        for old in list(_HELP_CACHE):
            if old[0] == str(p):
                _HELP_CACHE.pop(old, None)
        _HELP_CACHE[key] = text
    return text


def supported_options(sd_cli: Path | None) -> frozenset:
    """Long options advertised by the installed binary, not an assumed release."""
    return frozenset(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", binary_help(sd_cli)))


def auto_fit_args(binary: Path, enabled: bool) -> list[str]:
    """Support both historical boolean and current on|off auto-fit syntax."""
    known = supported_options(binary)
    if "--auto-fit" not in known:
        if enabled:
            raise EngineError("Auto-fit needs a newer engine. Run update-engine.bat.")
        return []
    help_text = binary_help(binary)
    modern = bool(re.search(r"--auto-fit[^\n]*(?:on\|off|'on' or 'off')", help_text))
    if modern:
        return ["--auto-fit", "on" if enabled else "off"]
    return ["--auto-fit"] if enabled else []


def memory_args(binary: Path, req: GenRequest) -> list[str]:
    """Shared CLI/server placement; CPU computation is distinct from residency."""
    known = supported_options(binary)
    args: list[str] = []
    flags = dict(req.flags)
    params = req.params_backend if "--params-backend" in known else ""
    if req.auto_fit:
        params = ""
        for flag in ("offload_to_cpu", "clip_on_cpu", "vae_on_cpu"):
            flags[flag] = False
    elif params:
        args += ["--params-backend", params]
        flags["offload_to_cpu"] = False
    if req.max_vram and "--max-vram" in known:
        args += ["--max-vram", req.max_vram]
    if req.stream_layers and stream_layers_possible(binary, flags, params):
        args.append("--stream-layers")
    args += _flag_args(flags, binary)
    args += auto_fit_args(binary, req.auto_fit)
    if req.auto_fit:
        # Older engines could split compute here. Current auto-fit only places
        # weights; its documentation explicitly says split-mode has no effect.
        if req.split_mode and "--split-mode" in known and "--disable-segmented-compute" not in known:
            args += ["--split-mode", req.split_mode]
    elif req.encoder_gpu_index is not None and req.encoder_gpu_index != req.gpu_index:
        if "--backend" not in known:
            raise EngineError("Encoder GPU placement needs a newer engine. Run update-engine.bat.")
        g = req.gpu_index if req.gpu_index is not None else 0
        te = "cpu" if flags.get("clip_on_cpu") else f"cuda{req.encoder_gpu_index}"
        vae = "cpu" if flags.get("vae_on_cpu") else f"cuda{g}"
        args += ["--backend", f"diffusion=cuda{g},vae={vae},te={te}"]
    return args


# Orthographes possibles de l'option « image de masque », par ordre de préférence.
_MASK_CANDIDATES = ("--mask-image", "--mask-img", "--mask")


def mask_flag(sd_cli: Path | None) -> str | None:
    """Nom de l'option de masque supportée, ou None si le binaire n'en a pas.

    Le masque de sd.cpp est appliqué PENDANT l'échantillonnage : blanc = zone à
    (re)générer, noir = zone conservée. Sans masque, sd.cpp en fabrique un tout
    blanc — c'est-à-dire « repeins tout », le comportement img2img normal."""
    opts = supported_options(sd_cli)
    if not opts:
        return None
    return next((c for c in _MASK_CANDIDATES if c in opts), None)


# Options qui n'existent que sur les binaires récents. On ne les envoie que si
# CE binaire les connaît : sinon sd-cli s'arrête sur un argument inconnu, et
# l'utilisateur récolte une erreur illisible au lieu d'une image.
_OPTIONAL_FLAGS = ("--diffusion-conv-direct", "--vae-conv-direct")


def _flag_args(flags: Mapping[str, bool],
               sd_cli: Path | None = None) -> list[str]:
    mapping = {
        "diffusion_fa": "--diffusion-fa",
        "offload_to_cpu": "--offload-to-cpu",
        "vae_tiling": "--vae-tiling",
        "clip_on_cpu": "--clip-on-cpu",
        "vae_on_cpu": "--vae-on-cpu",
        # Convolution directe : évite le gros tampon intermédiaire d'im2col.
        "conv_direct_diffusion": "--diffusion-conv-direct",
        "conv_direct_vae": "--vae-conv-direct",
    }
    wanted = [opt for key, opt in mapping.items() if flags.get(key)]
    risky = [o for o in wanted if o in _OPTIONAL_FLAGS]
    if risky:
        known = supported_options(sd_cli)
        if known:
            wanted = [o for o in wanted
                      if o not in _OPTIONAL_FLAGS or o in known]
        else:
            # Binaire non interrogeable : on s'abstient plutôt que de parier.
            wanted = [o for o in wanted if o not in _OPTIONAL_FLAGS]
    return wanted


def _require(*paths: Path | None) -> None:
    for p in paths:
        if p is not None and not Path(p).is_file():
            raise EngineError(
                f"Required file not found: {p}\n"
                "Download the model from the Model catalog tab.")


def _ref_list(ref) -> list[Path]:
    """Normalise ref_image (chemin unique ou liste) en liste de chemins."""
    if ref is None:
        return []
    if isinstance(ref, (list, tuple)):
        return [Path(r) for r in ref if r]
    return [Path(ref)]


# Agrandisseurs intégrés à sd.cpp pour la passe HD (pas de fichier à installer).
# Le reste des valeurs acceptées par --hires-upscaler est un NOM DE FICHIER
# d'ESRGAN cherché dans --hires-upscalers-dir.
HIRES_BUILTIN = ("Latent", "Latent (antialiased)", "Latent (bicubic)",
                 "Latent (bicubic antialiased)", "Latent (nearest)",
                 "Latent (nearest-exact)", "Lanczos", "Nearest")


def hires_supported(sd_cli: Path | None) -> bool:
    """Ce binaire connaît-il la passe HD native (`--hires`) ?"""
    return "--hires" in supported_options(sd_cli)


# --------------------------------------------------------------------------- #
#  Exécution SEGMENTÉE (« graph cut »).
#
#  Par défaut, sd.cpp alloue le graphe de calcul d'un seul bloc : si le bloc ne
#  tient pas, cudaMalloc échoue et toute la génération tombe — c'est l'OOM de la
#  passe HD. `--max-vram` lui donne un budget et l'autorise à DÉCOUPER le graphe
#  pour tenir dedans, au lieu de tenter le tout-ou-rien.
#
#  Une valeur NÉGATIVE est la plus intéressante : sd.cpp détecte alors la VRAM
#  réellement libre et en réserve la valeur indiquée
#  (resolve_auto_max_vram_bytes). « -1 » = « prends tout le libre sauf 1 Gio » —
#  ce qui s'adapte tout seul à la carte ET à ce qui l'occupe déjà, là où un
#  chiffre en dur ne s'adapte à rien.
# --------------------------------------------------------------------------- #
MAX_VRAM_AUTO = "auto"


def max_vram_supported(sd_cli: Path | None) -> bool:
    return "--max-vram" in supported_options(sd_cli)


def max_vram_arg(setting: str, spare_gib: float = 1.0) -> str:
    """Traduit la préférence utilisateur en valeur pour `--max-vram`.

    « auto » -> « -1 » (VRAM libre moins `spare_gib`). Un nombre est repris tel
    quel : l'utilisateur qui écrit « 6 » veut un plafond ferme de 6 Gio, et
    « cuda0=6,cuda1=4 » reste possible pour les machines multi-cartes.
    """
    s = (setting or "").strip()
    if not s or s in ("off", "0"):
        return ""
    if s == MAX_VRAM_AUTO:
        return f"-{spare_gib:g}"
    return s


def hires_args(h: "HiresParams") -> list[str]:
    """Options CLI de la passe HD. Appelant responsable de `hires_supported`."""
    args = ["--hires",
            "--hires-denoising-strength", f"{float(h.denoise):g}"]
    if h.target_width > 0 and h.target_height > 0:
        args += ["--hires-width", str(int(h.target_width)),
                 "--hires-height", str(int(h.target_height))]
    else:
        args += ["--hires-scale", f"{float(h.scale):g}"]
    if h.upscaler:
        args += ["--hires-upscaler", h.upscaler]
    # Le dossier n'est utile que pour un agrandisseur À MODÈLE : sd.cpp y
    # cherche le fichier nommé par --hires-upscaler.
    if h.upscalers_dir and h.upscaler not in HIRES_BUILTIN:
        args += ["--hires-upscalers-dir", str(h.upscalers_dir)]
        if h.tile_size and int(h.tile_size) > 0:
            args += ["--hires-upscale-tile-size", str(int(h.tile_size))]
    if h.steps and int(h.steps) > 0:
        args += ["--hires-steps", str(int(h.steps))]
    return args


def stream_layers_possible(sd_cli: "Path | None", flags: dict,
                           params_backend: str = "") -> bool:
    """`--stream-layers` aura-t-il un effet dans cette configuration ?

    Le moteur est formel (docs/performance.md) : l'option ne prend effet que si
    le backend de PARAMÈTRES de la diffusion est le CPU ; sinon elle est ignorée
    avec un avertissement. Elle ne dépend PAS de `--max-vram`, contrairement à
    ce que le code d'origine supposait.

    La condition est extraite ici parce qu'elle sert deux fois : à la
    construction de la commande, et en amont — la passe HD doit savoir si
    activer le streaming vaut une tentative avant de baisser son facteur.
    """
    low = (params_backend or "").lower().replace(" ", "")
    diffusion_on_cpu = (bool(flags.get("offload_to_cpu"))
                        or "diffusion=cpu" in low or "*=cpu" in low)
    return diffusion_on_cpu and "--stream-layers" in supported_options(sd_cli)


def build_gen_cmd(sd_cli: Path, req: GenRequest, output: Path) -> list[str]:
    refs = _ref_list(req.ref_image)
    _require(req.model_path, req.diffusion_model, req.vae, req.text_encoder,
             req.llm_vision, req.t5xxl, req.clip_l, req.uncond_model,
             req.init_image, req.mask_image, *refs)

    cmd: list[str] = [str(sd_cli), "--mode", "img_gen"]
    if req.model_path:
        # Checkpoint complet : CLIP + VAE inclus.
        cmd += ["-m", str(req.model_path)]
        if req.vae:
            cmd += ["--vae", str(req.vae)]
    else:
        cmd += ["--diffusion-model", str(req.diffusion_model)]
        if req.uncond_model:
            cmd += ["--uncond-diffusion-model", str(req.uncond_model)]
        if req.vae:
            cmd += ["--vae", str(req.vae)]
        if req.text_encoder:
            cmd += ["--llm", str(req.text_encoder)]
        if req.llm_vision:
            cmd += ["--llm_vision", str(req.llm_vision)]
        if req.t5xxl:
            cmd += ["--t5xxl", str(req.t5xxl)]
        if req.clip_l:
            cmd += ["--clip_l", str(req.clip_l)]

    cmd += list(req.extra_flags)
    cmd += ["-p", req.prompt]
    if req.negative and req.cfg_scale > 1.0:
        cmd += ["-n", req.negative]

    cmd += [
        "--cfg-scale", f"{req.cfg_scale}",
        "--steps", f"{req.steps}",
        "--sampling-method", req.sampler,
        "-W", f"{req.width}", "-H", f"{req.height}",
        "-s", f"{req.seed}", "-b", f"{req.batch_count}",
    ]
    if req.schedule:
        cmd += ["--scheduler", req.schedule]
    if req.flow_shift and req.flow_shift > 0:
        cmd += ["--flow-shift", f"{req.flow_shift}"]
    if req.init_image:
        cmd += ["-i", str(req.init_image), "--strength", f"{req.strength}"]
        # Masque : uniquement si CE binaire connaît l'option (sinon sd.cpp
        # planterait sur un argument inconnu — on dégrade en img2img simple).
        if req.mask_image:
            mf = mask_flag(sd_cli)
            if mf:
                cmd += [mf, str(req.mask_image)]
    for r in refs:
        # Édition d'image (Flux.2) : pilotée par le prompt, sans strength.
        # Plusieurs « -r » = édition multi-référence (combine les images).
        cmd += ["-r", str(r)]
    if req.lora_dir:
        cmd += ["--lora-model-dir", str(req.lora_dir)]

    known = supported_options(sd_cli)
    if req.hires and hires_supported(sd_cli):
        cmd += hires_args(req.hires)
    if req.preview_path:
        cmd += ["--preview", "proj", "--preview-path", str(req.preview_path),
                "--preview-interval", "1"]
    # Accélération par cache (opt-in) : saute des calculs quasi identiques entre
    # pas. Nécessite un sd-cli récent (update-engine.bat si flag inconnu).
    if req.cache_mode:
        if "--cache-mode" not in known or (req.cache_option and "--cache-option" not in known):
            raise EngineError("This cache setting needs a newer engine. Run update-engine.bat.")
        cmd += ["--cache-mode", req.cache_mode]
        if req.cache_option:
            cmd += ["--cache-option", req.cache_option]
    # Résidence explicite des poids. `--offload-to-cpu` est un ancien raccourci
    # équivalent à params *=cpu : l'envoyer en même temps rendrait le résultat
    # dépendant de l'ordre de parsing. Quand l'option moderne existe, elle est
    # seule à décider de la résidence.
    cmd += memory_args(sd_cli, req)
    cmd += ["-o", str(output), "-v"]
    return cmd


def build_convert_cmd(sd_cli: Path, input_model: Path, output_model: Path,
                      qtype: str) -> list[str]:
    """Conversion/quantification d'un modèle en GGUF (sd.cpp --mode convert).

    Lit un checkpoint/safetensors/diffusion et le ré-écrit dans la quant `qtype`
    (ex. « q4_k », « q8_0 »). 100% CPU, aucune diffusion : c'est une simple
    transformation des poids. Voir docs/quantization_and_gguf.md."""
    return [str(sd_cli), "-M", "convert", "-m", str(input_model),
            "-o", str(output_model), "--type", qtype, "-v"]


# Taille de tuile de l'upscaler ESRGAN, par défaut dans sd.cpp : 128 px.
#
# C'est la cause principale des artefacts et de l'aliasing d'un agrandissement
# ESRGAN. sd.cpp découpe l'image en tuiles de 128 px (recouvrement 25%, fondu
# smootherstep) et fait tourner le réseau RRDB sur chaque tuile SÉPARÉMENT. Un
# RRDB décide de son accentuation d'après le contenu qu'il voit : sur 128 px il
# ne voit presque rien, donc deux tuiles voisines traitent le même trait
# différemment. Le fondu adoucit la couture mais ne recolle pas deux décisions
# contradictoires — d'où les micro-marches sur les diagonales et le grain qui
# change de nature d'un carré à l'autre.
#
# La correction est de laisser le réseau voir l'image entière (ou de très
# grandes tuiles). Coût : de la VRAM, d'où le plafond par carte ci-dessous.
SDCPP_DEFAULT_UPSCALE_TILE = 128

# (VRAM minimale en Go, tuile maximale en px). Volontairement prudent : un OOM
# est pire qu'un artefact, et un repli automatique existe de toute façon.
_UPSCALE_TILE_BY_VRAM = ((16, 1024), (11, 832), (8, 640), (0, 512))


def upscale_tile_size(width: int, height: int,
                      vram_gb: float | None = None) -> int:
    """Tuile ESRGAN à demander pour une image de `width`×`height`.

    Renvoie `max(width, height)` quand l'image tient sous le plafond de la carte
    — sd.cpp prend alors son chemin SANS découpage (`tile_size <= 0 ||
    l'image tient dans la tuile`), c'est-à-dire zéro couture par construction.
    Sinon on renvoie le plafond : moins de tuiles, plus de contexte, donc moins
    d'artefacts qu'avec les 128 px d'origine.
    """
    cap = 512
    if vram_gb:
        cap = next(px for gb, px in _UPSCALE_TILE_BY_VRAM if vram_gb >= gb)
    longest = max(int(width or 0), int(height or 0), 1)
    return longest if longest <= cap else cap


def build_upscale_cmd(sd_cli: Path, init_image: Path, upscale_model: Path,
                      output: Path, repeats: int = 1,
                      offload: bool = True, tile_size: int = 0) -> list[str]:
    """Upscale ESRGAN natif sd.cpp (--mode upscale). Déterministe, 100% GPU.

    `repeats` applique le modèle plusieurs fois (ex. un modèle ×2 appliqué 2 fois
    = ×4). Aucun prompt/diffusion : c'est un réseau ESRGAN GGUF.

    `tile_size` > 0 élargit les tuiles du réseau (voir `upscale_tile_size`).
    L'option n'est envoyée que si CE binaire la connaît : sur un sd-cli plus
    ancien elle n'existe pas et l'argument inconnu ferait échouer la commande.
    """
    _require(upscale_model, init_image)
    cmd = [str(sd_cli), "--mode", "upscale", "-i", str(init_image),
           "--upscale-model", str(upscale_model)]
    if repeats and repeats > 1:
        cmd += ["--upscale-repeats", str(int(repeats))]
    if tile_size and int(tile_size) > 0 and \
            "--upscale-tile-size" in supported_options(sd_cli):
        cmd += ["--upscale-tile-size", str(int(tile_size))]
    if offload:
        cmd.append("--offload-to-cpu")
    cmd += ["-o", str(output), "-v"]
    return cmd


# Vitesse de lecture des poids, déduite des traces de sd-cli. Deux lignes
# consécutives suffisent : le temps de lecture, puis la taille du tampon obtenu.
#   « loading tensors completed, taking 80.54s (read: 79.86s, memcpy: … ) »
#   « prepared params backend buffer (8410.71 MB, 430 tensors, VRAM) »
_READ_SECONDS = re.compile(r"loading tensors completed, taking [\d.]+s\s*"
                           r"\(read:\s*([\d.]+)s")
_PARAMS_MB = re.compile(r"prepared params backend buffer \(([\d.]+)\s*MB")
# Un SSD SATA lit à ~500 Mo/s, un NVMe à plusieurs Go/s. En dessous de 300, ce
# n'est plus un SSD — et sur un modèle de 8 Go ça se paie en minutes.
SLOW_DISK_MB_S = 300.0
# Sous ce temps de lecture, le rapport n'est pas fiable (fichiers minuscules,
# cache disque) et ne vaut de toute façon pas un avertissement.
_MIN_READ_S = 4.0


class DiskWatch:
    """Repère un stockage lent à partir du journal de sd-cli.

    L'information existe déjà dans le journal, mais éclatée sur deux lignes et
    exprimée en secondes et en mégaoctets séparément : personne ne fait la
    division de tête. On la fait, et on ne dit rien tant qu'il n'y a rien à
    dire.
    """

    def __init__(self) -> None:
        self._read_s: float | None = None
        self._said = False

    def note(self, line: str) -> str | None:
        m = _READ_SECONDS.search(line)
        if m:
            self._read_s = float(m.group(1))
            return None
        m = _PARAMS_MB.search(line)
        if m is None or self._read_s is None:
            return None
        read_s, self._read_s = self._read_s, None
        size_mb = float(m.group(1))
        if self._said or read_s < _MIN_READ_S or size_mb <= 0:
            return None
        speed = size_mb / read_s
        if speed >= SLOW_DISK_MB_S:
            return None
        self._said = True
        ssd = size_mb / 550.0          # SSD SATA, l'hypothèse la plus prudente
        return (f"⚠️ Model read at {speed:.0f} MB/s "
                f"({size_mb / 1024:.1f} GB in {read_s:.0f} s): that is "
                f"mechanical-disk speed. On an SSD the same read would take "
                f"~{ssd:.0f} s, i.e. ~{read_s - ssd:.0f} s less ON EVERY "
                f"image. Move the models/ folder.")


def child_env_for(gpu_index: int | None,
                  all_gpus: bool) -> "dict[str, str] | None":
    """Environnement CUDA d'un sous-process moteur (None = celui du parent).

    Deux régimes seulement, et ils ne se mélangent pas : soit une seule carte
    est visible et elle devient cuda0, soit toutes le sont et gardent leurs
    index nvidia-smi. Le second impose l'ordre par bus PCI, sinon « cuda1 »
    dans un mapping de résidence ne désigne pas la carte attendue.
    """
    if all_gpus:
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        return env
    if gpu_index is not None:
        return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    return None


def run(cmd: list[str], log: Callable[[str], None] | None = None,
        gpu_index: int | None = None, all_gpus: bool = False) -> None:
    global _CANCELLED
    # Le moteur résident garde son modèle en VRAM. Une commande sd-cli lancée
    # par-dessus (LoRA, passe HD, upscale ESRGAN — tout ce que le serveur ne
    # sert pas) tomberait sur une carte déjà pleine. Il rend la place ici et se
    # rechargera à la prochaine image qu'il sait servir.
    release_resident_engine("a command needs the whole card", log)
    env = child_env_for(gpu_index, all_gpus)
    if log:
        log("$ " + " ".join(_q(c) for c in cmd))
    _CANCELLED = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT), env=env,
                            encoding="utf-8", errors="replace")
    with _LOCK:
        _ACTIVE.add(proc)
    # On garde la fin de la sortie pour diagnostiquer les crashs de sd-cli
    # (l'assert GGML n'apparaît que quelques lignes avant la mort du process).
    tail: deque[str] = deque(maxlen=100)
    disk = DiskWatch()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            s = line.rstrip("\n")
            tail.append(s)
            if log:
                log(s)
                note = disk.note(s)
                if note:
                    log(note)
        code = proc.wait()
    finally:
        with _LOCK:
            _ACTIVE.discard(proc)
    if _CANCELLED:
        raise EngineError("Interrupted by the user.")
    if code != 0:
        raise _failure_error(code, cmd, tail)


_OOM_MARKERS = ("cudaMalloc failed: out of memory",
                "failed to allocate the compute buffer",
                "alloc compute buffer failed",
                "ggml_gallocr_reserve_n_impl: failed to allocate",
                "out of memory")

# « allocating 4731.82 MiB on device 0 » → on récupère la taille demandée pour
# pouvoir la citer : c'est le chiffre qui rend le message actionnable.
_OOM_SIZE = re.compile(r"allocating\s+([\d.]+)\s*MiB")


def _failure_error(code: int, cmd: list[str],
                   tail: "deque[str]") -> EngineError:
    """Exception TYPÉE pour un échec de sd-cli.

    Le type porte l'information « on peut réessayer plus petit » ; le message
    porte l'explication pour l'utilisateur. Les deux séparément, plutôt qu'un
    marqueur planqué dans le texte que le premier reformatage effacerait.
    """
    if any(m in ln for ln in tail for m in _OOM_MARKERS):
        want = ""
        for ln in reversed(tail):
            m = _OOM_SIZE.search(ln)
            if m:
                want = (f" It was short by a block of "
                        f"{float(m.group(1)) / 1024:.1f} GB.")
                break
        return VramError(
            "❌ Not enough GPU memory." + want + "\n"
            + ("The HD pass re-denoises the WHOLE image: its cost climbs with "
               "the pixel count, and it adds to the model weights already on "
               "the card.\n→ Lower the enlargement factor, or go through “🔼 "
               "Enlarge (ESRGAN)” and then “✨ Creative upscale (SDXL)”, which "
               "works in tiles and fits in far less VRAM."
               if "--hires" in cmd else
               "→ Lower the resolution, or pick a lighter quantization in "
               "Settings (the model will take up less VRAM)."))
    return EngineError(_diagnose_failure(code, cmd, tail))


def _diagnose_failure(code: int, cmd: list[str], tail: "deque[str]") -> str:
    """Transforme un code de sortie brut de sd-cli en message actionnable.

    L'assert GGML de reshape — dimensions de tenseur incompatibles — est presque
    toujours un LoRA entraîné pour une autre base, ou une résolution hors grille.
    """
    reshape_assert = any("GGML_ASSERT(ggml_nelements(a) ==" in ln for ln in tail)
    if reshape_assert:
        has_lora = "--lora-model-dir" in cmd or any("<lora:" in c for c in cmd)
        if has_lora:
            return (
                "❌ Crash while applying a LoRA (incompatible tensor "
                "shapes).\nThis LoRA is not compatible with the selected "
                "model — usually a LoRA trained for a different base (e.g. "
                "Krea 2 “full” while you are using Krea 2 Turbo).\n→ Try "
                "again without that LoRA, or use the model it was trained "
                "for.")
        return (
            "❌ sd-cli crashed on a tensor reshape (incompatible "
            "dimensions).\n"
            "Check that the resolution follows the model's grid "
            "(a multiple of 64 px for Krea, 32 px for Flux.2).\n"
            f"(exit code {code})")
    return f"sd-cli exited with code {code}."


def _q(s: str) -> str:
    return f'"{s}"' if " " in s else s


def unique_output(prefix: str, ext: str = "png") -> Path:
    settings.ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ms = int(time.time() * 1000) % 1000
    return settings.OUTPUT_DIR / f"{prefix}-{stamp}-{ms:03d}.{ext}"


def collect_outputs(output: Path, batch_count: int) -> list[Path]:
    if batch_count <= 1 and output.is_file():
        return [output]
    found = sorted(output.parent.glob(f"{output.stem}_*{output.suffix}"))
    return found or ([output] if output.is_file() else [])
