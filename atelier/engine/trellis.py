"""Moteur trellis.cpp : image → maillage 3D (GLB) via le binaire natif.

La release Windows prête fournit **`trellis-server.exe`** (serveur HTTP,
C++/GGML/CUDA, aucun PyTorch), pas de CLI one-shot. On le pilote donc de façon
**TRANSITOIRE** : on le démarre, on attend `/health`, on poste l'image sur
`/generate`, on récupère le GLB, puis on **arrête le serveur** → toute la VRAM
est libérée (stratégie low-VRAM, comme AISmith-3D).

API serveur (POST /generate, multipart) :
  - image        : fichier image (requis)
  - resolution   : 512 | 1024 | 1536   (512 « light » tient sur ≤ 12 Go)
  - seed         : entier (optionnel)
  - bg_removal   : threshold | birefnet
  Réponse : octets GLB bruts (model/gltf-binary).
"""
from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from .. import settings
from . import release_resident_engine
from . import sdcpp

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

TRELLIS_BIN_DIR = settings.BIN_DIR / "trellis"
MODELS_DIR = settings.MODELS_DIR / "trellis"
DEFAULT_PORT = 8000

# Résolutions proposées. Seul le 512 tient sur 11-12 Go (les autres ~16 Go+).
RESOLUTIONS = [
    ("512 — light (recommended, ≤ 12 GB)", 512),
    ("1024 — cascade (≥ 16 GB)", 1024),
    ("1536 — high (≥ 16 GB+)", 1536),
]


def _is_trellis_exe(p: Path) -> bool:
    if not p.is_file():
        return False
    n = p.name.lower()
    if platform.system() == "Windows" and not n.endswith(".exe"):
        return False
    stem = n[:-4] if n.endswith(".exe") else n
    return "trellis" in stem and not any(
        x in stem for x in ("test", "bench", "studio", "convert"))


def find_server() -> Path | None:
    """Localise le binaire serveur trellis (nom variable selon la release)."""
    if settings.BIN_DIR.exists():
        exes = [p for p in settings.BIN_DIR.rglob("*") if _is_trellis_exe(p)]
        # Priorité à un binaire « server » ; sinon le premier exécutable trellis.
        srv = [p for p in exes if "server" in p.name.lower()]
        if srv:
            return srv[0]
        if exes:
            return exes[0]
    for name in ("trellis-server", "trellis"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


# Variantes de poids (dépôt ilintar/trellis2-gguf) : f16 à la racine, q8/ et
# q4/ en sous-dossiers. Quantifier réduit fortement l'empreinte mémoire, ce qui
# peut rendre les modes 1024/1536 atteignables sur une carte modeste.
VARIANTS = [
    ("f16 — reference (~16.5 GB)", "f16"),
    ("q8 — near lossless (~9.9 GB)", "q8"),
    ("q4 — slight grain, lighter (~6 GB)", "q4"),
]


def variant_dir(variant: str = "f16") -> Path:
    return MODELS_DIR if variant in (None, "", "f16") else MODELS_DIR / variant


def models_ready(variant: str = "f16") -> bool:
    d = variant_dir(variant)
    if not d.is_dir():
        return False
    # En f16, les .gguf des sous-dossiers q8//q4/ ne comptent pas.
    return any(d.glob("*.gguf") if variant in (None, "", "f16")
               else d.rglob("*.gguf"))


def installed_variants() -> list[str]:
    return [v for _, v in VARIANTS if models_ready(v)]


def diagnose() -> str:
    """État lisible : OÙ l'app cherche, et ce qu'elle y trouve.

    Rend visible le cas « installé ailleurs que là où on regarde » (dossier de
    modèles externe), au lieu d'un silencieux « pas installé ».
    """
    srv = find_server()
    lines = [f"- Engine: {'`' + str(srv) + '`' if srv else '**not found**'}",
             f"- Models folder: `{MODELS_DIR}`"]
    if not MODELS_DIR.exists():
        lines.append("  - ⚠️ **this folder does not exist** — nothing was "
                     "installed here.")
    for lbl, v in VARIANTS:
        d = variant_dir(v)
        if models_ready(v):
            n = len(list(d.glob("*.gguf")))
            lines.append(f"- ✅ **{v}** — {n} file(s) in `{d}`")
        else:
            lines.append(f"- — {v}: absent (`{d}`)")
    # Un jeu de modèles présent dans le dossier PROJET alors qu'on regarde
    # ailleurs = installation faite avant/hors du déplacement des modèles.
    proj = settings.DEFAULT_MODELS_DIR / "trellis"
    if proj.resolve() != MODELS_DIR.resolve() and proj.is_dir() \
            and any(proj.rglob("*.gguf")):
        lines.append(f"\n⚠️ **Trellis models also exist inside the project "
                     f"folder** (`{proj}`) while the app is looking at "
                     f"`{MODELS_DIR}`. Move them to the folder above (or run "
                     "the installation again) so they are seen.")
    return "\n".join(lines)


def is_ready() -> bool:
    return find_server() is not None and bool(installed_variants())


def _wait_health(port: int, proc: subprocess.Popen, deadline: float) -> bool:
    """Attend que GET /health réponde « ok », tant que le serveur vit."""
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False                       # serveur mort pendant le chargement
        try:
            r = requests.get(url, timeout=2)
            if r.ok and "ok" in r.text.lower():
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return False


# --------------------------------------------------------------------------
#  Serveur RÉSIDENT (optionnel) : garde le serveur en vie entre les générations
#  → plus de rechargement des modèles (~30 s gagnées par objet), mais la VRAM
#  reste occupée. À réserver aux séries de 3D ; à arrêter avant de générer des
#  images. Par défaut le mode transitoire (start/stop) reste actif.
# --------------------------------------------------------------------------
# « sig » = signature des flags de LANCEMENT (res, decim, atlas, gpu…). Ils ne
# sont PAS renégociables par requête : si elle change, il faut relancer le
# serveur, sinon les nouveaux réglages seraient silencieusement ignorés.
# « log » = journal de la génération EN COURS : le lecteur de sortie y écrit
# dynamiquement, sinon les logs resteraient attachés à la 1re génération.
# --------------------------------------------------------------------------- #
#  Diagnostic des morts subites du serveur
#
#  Quand trellis plante en pleine génération (erreur CUDA, assert GGML), la
#  requête HTTP en cours est coupée net et « requests » remonte un banal
#  ConnectionResetError. C'est le SYMPTÔME : la cause est passée dans la sortie
#  du serveur quelques lignes plus haut. On la retient au vol pour pouvoir la
#  rejouer à l'utilisateur au lieu de lui montrer une erreur de socket.
# --------------------------------------------------------------------------- #
_FATAL_MARKERS = ("CUDA error", "no kernel image", "out of memory",
                  "GGML_ASSERT", "cudaMalloc", "device kernel image",
                  "invalid device function", "insufficient")
_CRASH: "deque[str]" = deque(maxlen=25)

# Capacité de calcul CUDA par architecture, pour nommer précisément ce qui
# manque dans le binaire (« sm_75 ») plutôt que de rester vague.
# La table vit désormais dans hardware, alimentée par la capacité de calcul que
# rapporte le pilote — plus fiable qu'une déduction depuis le nom commercial.
# Alias conservé : d'anciennes traces d'exécution y font référence.
from ..hardware import _SM_BY_ARCH  # noqa: E402,F401


def _note_if_fatal(line: str) -> None:
    if any(m.lower() in line.lower() for m in _FATAL_MARKERS):
        _CRASH.append(line.strip())


def _gpu_hint(gpu_index: "int | None") -> str:
    """« RTX 2080 Ti (turing, sm_75) » — pour un message qui parle DE SA carte."""
    try:
        from .. import hardware
        gpus = hardware.detect_gpus()
    except Exception:  # noqa: BLE001
        return ""
    g = next((x for x in gpus if x.index == gpu_index), None) if \
        gpu_index is not None else (max(gpus, key=lambda x: x.vram_gb)
                                    if gpus else None)
    if g is None:
        return ""
    # `g.sm` vient de la capacité de calcul rapportée par le pilote quand elle
    # est disponible : c'est EXACTEMENT le sm_XX que cite l'erreur CUDA.
    return f"{g.name} ({g.arch}{', ' + g.sm if g.sm else ''})"


def _diagnose_crash(gpu_index: "int | None" = None) -> str:
    """Message actionnable à partir de ce que le serveur a craché avant de mourir."""
    lines = list(_CRASH)
    joined = " ".join(lines).lower()
    card = _gpu_hint(gpu_index)
    who = f" on **{card}**" if card else ""
    if "no kernel image" in joined or "invalid device function" in joined:
        return (
            f"❌ **The installed trellis binary has no machine code for "
            f"your card**{who}.\n\n"
            "This is neither a VRAM shortage nor a setting: the binary was "
            "compiled for architectures other than yours. Because CUDA errors "
            "are sticky, it is the *next* ggml operation (often `IM2COL`) "
            "that reports it — which sends the diagnosis astray.\n\n"
            "**For a long time there was no way out**: the upstream "
            "`CMakeLists.txt` overwrote its own CI's architecture list and "
            "compiled only for the RTX 30xx and 50xx. **v0.6.0 (August 2026) "
            "fixed that** and now publishes two CUDA archives:\n\n"
            "- `cuda` — Turing and newer (RTX 20xx, 30xx, 40xx, 50xx);\n"
            "- `cuda12` — Pascal and Volta (GTX 10xx, Titan V).\n\n"
            "**The fix: “⬆️ Update the binary”** in this tab. The archive is "
            "chosen from your card, and Vulkan remains the fallback if "
            "neither list covers it. The ~10 GB of models are not downloaded "
            "again.")
    if "out of memory" in joined or "cudamalloc" in joined:
        return ("❌ **Not enough GPU memory** — the trellis server was killed "
                "mid-computation.\n\nLower the resolution (512), close other "
                "3D/AI applications, and check that no resident server is "
                "already holding the VRAM.")
    if lines:
        return ("❌ **The trellis server stopped during generation.**\n\nLast "
                "lines before it stopped:\n```\n"
                + "\n".join(lines[-6:]) + "\n```")
    return ("❌ **The trellis server stopped during generation** with no "
            "usable message. See the full log below.")


_RESIDENT: dict = {"proc": None, "port": None, "res": None, "sig": None,
                   "log": None}
_RES_LOCK = threading.Lock()


def resident_is_running() -> bool:
    p = _RESIDENT.get("proc")
    return p is not None and p.poll() is None


def resident_status() -> str:
    if resident_is_running():
        return f"🟢 Resident server running (port {_RESIDENT['port']}, "\
               f"res {_RESIDENT['res']}) — VRAM still occupied."
    return "⚪ Resident server stopped (transient mode: start/stop on every generation, VRAM freed)."


def resident_stop() -> str:
    with _RES_LOCK:
        p = _RESIDENT.get("proc")
        if p is not None:
            try:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
            except Exception:  # noqa: BLE001
                pass
        _RESIDENT.update({"proc": None, "port": None, "res": None,
                          "sig": None, "log": None})
    return "⏹️ Resident server stopped — VRAM freed."


def build_server_args(res: int, decim: int = 0, atlas: int = 0,
                      no_texture: bool = False, box_uv: bool = False,
                      require_gpu: bool = True, f32: bool = False,
                      no_fa: bool = False,
                      gpu: int | None = None, variant: str = "f16",
                      band: float = 0.0) -> list[str]:
    """Flags de lancement du serveur trellis (voir README trellis.cpp)."""
    args = ["--models", str(variant_dir(variant)), "--res", str(int(res))]
    if band and float(band) > 0:
        # Surcharge l'offset de remaillage « narrow-band » (v0.5.4 : il s'adapte
        # désormais à la résolution, ce qui corrige les speckles en 1024).
        args += ["--band", str(band)]
    if gpu is not None:
        # Flag OFFICIEL de trellis (« --gpu N », N<0 = CPU) : plus propre que
        # CUDA_VISIBLE_DEVICES, et l'index correspond à celui affiché au démarrage.
        args += ["--gpu", str(int(gpu))]
    if decim and int(decim) > 0:
        args += ["--decim", str(int(decim))]       # cible de décimation (faces)
    if atlas and int(atlas) > 0:
        args += ["--atlas", str(int(atlas))]       # taille de l'atlas UV (px)
    if no_texture:
        args.append("--no-texture")                # géométrie seule (plus rapide)
    if box_uv:
        args.append("--box-uv")
    if require_gpu:
        # Évite un repli CPU SILENCIEUX (des heures de calcul) si la VRAM manque.
        args.append("--require-gpu")
    if f32:
        args.append("--f32")                       # précision 32 bits (défaut f16)
    if no_fa:
        args.append("--no-fa")                     # désactive FlashAttention
    return args


def generate(image_path: Path, out_path: Path, res: int = 512,
             seed: int | None = None, bg_removal: str = "birefnet",
             port: int = DEFAULT_PORT, extra: str = "",
             log: Callable[[str], None] | None = None,
             gpu_index: int | None = None,
             resident: bool = False,
             decim: int = 0, atlas: int = 0, no_texture: bool = False,
             box_uv: bool = False, require_gpu: bool = True,
             f32: bool = False, no_fa: bool = False,
             variant: str = "f16", band: float = 0.0,
             meta: dict | None = None) -> Path:
    """Génère un GLB 3D à partir d'une image via le serveur trellis.

    `resident=False` (défaut) : serveur démarré puis ARRÊTÉ (VRAM libérée).
    `resident=True` : serveur gardé en vie pour les générations suivantes.
    `gpu_index` : carte de calcul, via le flag officiel « --gpu N » (N<0 = CPU).
      Choisir la carte avec le plus de VRAM : le cascade 1024 est documenté
      pour ~16 Go, en dessous la géométrie sort dégradée (« blobs ») au lieu
      d'échouer proprement.
    """
    if requests is None:
        raise sdcpp.EngineError("Module « requests » manquant (pip install requests).")
    server = find_server()
    if server is None:
        raise sdcpp.EngineError(
            "trellis server not found — install trellis.cpp (3D tab).")
    if not models_ready(variant):
        raise sdcpp.EngineError(
            f"Trellis models “{variant}” missing — install that variant "
            "(3D tab).")
    settings.ensure_dirs()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info: dict = meta if meta is not None else {}

    def _log(m: str) -> None:
        # Le serveur annonce la graine RÉELLEMENT utilisée (« … with seed N
        # (auto) » quand on lui laisse le choix) : on la capture pour pouvoir
        # rejouer l'objet et la consigner dans le .txt.
        mt = re.search(r"with seed\s+(\d+)", m)
        if mt:
            info["seed"] = int(mt.group(1))
        if log:
            log(m)

    launch_args = build_server_args(
        res, decim=decim, atlas=atlas, no_texture=no_texture, box_uv=box_uv,
        require_gpu=require_gpu, f32=f32, no_fa=no_fa, gpu=gpu_index,
        variant=variant, band=band)
    if extra and extra.strip():
        launch_args += shlex.split(extra)
    sig = tuple(launch_args)
    # Réglages consignés dans le .txt à côté du GLB (la graine réelle y est
    # injectée après coup : le serveur la choisit quand on passe seed = auto).
    params = {"image": Path(image_path).name, "res": int(res),
              "bg_removal": bg_removal, "decim": decim, "atlas": atlas,
              "no_texture": no_texture, "box_uv": box_uv, "f32": f32,
              "no_fa": no_fa, "gpu": gpu_index, "variant": variant}

    # Réutilise le serveur résident SEULEMENT si TOUS les flags de lancement
    # sont identiques. decim/atlas/no-texture/gpu… ne sont pas renégociables
    # par requête : sans ce contrôle, les changer resterait sans effet.
    if resident and resident_is_running():
        if _RESIDENT.get("sig") == sig:
            use_port = _RESIDENT["port"]
            with _RES_LOCK:
                _RESIDENT["log"] = _log      # journal de CETTE génération
            _log(f"♻️ Reusing the resident server (port {use_port}) — "
                 "no model reload.")
            _CRASH.clear()
            _post_generate(image_path, out_path, res, seed, bg_removal,
                           use_port, _log, gpu_index)
            _sidecar(out_path, {**params, "seed": info.get("seed", seed)})
            return out_path
        _log("🔄 Launch settings changed (resolution/decimation/atlas/GPU…) — "
             "restarting the server to apply them.")

    # Un résident aux réglages différents doit céder la place.
    if resident_is_running():
        _log(resident_stop())
    # Le moteur d'images résident aussi : deux serveurs qui gardent chacun
    # leur modèle sur la même carte, c'est un OOM avec deux coupables.
    release_resident_engine("3D needs the GPU", _log)

    cmd = [str(server)] + launch_args
    # NB : on n'utilise PAS CUDA_VISIBLE_DEVICES ici — le flag « --gpu N » de
    # trellis fait le travail, et masquer les cartes en plus décalerait les
    # index (la carte N deviendrait la 0 pour le process).
    env = None

    _log("$ " + " ".join(cmd))
    _CRASH.clear()          # diagnostic propre à CE démarrage
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, cwd=str(settings.ROOT), env=env,
        encoding="utf-8", errors="replace")

    state: dict = {"port": None}

    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            s = line.rstrip("\n")
            # En résident, ce thread survit à la génération : il doit écrire
            # dans le journal COURANT, pas celui (mort) de la 1re génération.
            sink = _RESIDENT.get("log") if _RESIDENT.get("proc") is proc else None
            (sink or _log)(s)
            _note_if_fatal(s)
            if state["port"] is None:
                m = re.search(r"https?://[^\s:]+:(\d{2,5})", s) \
                    or re.search(r"(?:listen|port)\D{0,12}(\d{4,5})", s, re.I)
                if m:
                    state["port"] = int(m.group(1))

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()

    try:
        # Attente du chargement des modèles (~10 Go) puis de /health.
        _log("⏳ Starting the trellis server (loading the models)…")
        deadline = time.time() + 600
        use_port = port
        # Laisse au serveur le temps d'annoncer son port dans ses logs.
        for _ in range(15):
            if state["port"]:
                use_port = state["port"]
                break
            if proc.poll() is not None:
                break
            time.sleep(1.0)
        if state["port"]:
            use_port = state["port"]
        if not _wait_health(use_port, proc, deadline):
            if _CRASH:
                # Il a parlé avant de mourir : autant le citer.
                raise sdcpp.EngineError(_diagnose_crash(gpu_index))
            raise sdcpp.EngineError(
                f"The trellis server did not answer on port {use_port} "
                "(see the log: a different port, not enough VRAM, or "
                "incomplete models?).")

        _log(f"✅ Server ready (port {use_port}) — sending the image…")
        _post_generate(image_path, out_path, res, seed, bg_removal, use_port,
                       _log, gpu_index)
    finally:
        if resident:
            # Mode résident : on GARDE le serveur en vie pour la suite. On
            # mémorise la signature des flags pour détecter tout changement.
            with _RES_LOCK:
                _RESIDENT.update({"proc": proc, "port": use_port,
                                  "res": int(res), "sig": sig, "log": _log})
            _log("♻️ Server kept resident (VRAM occupied — use “Stop the "
                 "resident server” to free it).")
        else:
            # Mode transitoire : arrêt → libération VRAM (stratégie low-VRAM).
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:  # noqa: BLE001
                pass
    _sidecar(out_path, {**params, "seed": info.get("seed", seed)})
    return out_path


def _sidecar(out_path: Path, params: dict) -> None:
    """Écrit un .txt à côté du GLB (même principe que les images) : on retrouve
    la graine réellement utilisée et tous les réglages dans outputs/."""
    lines = [f"Source image: {params.get('image', '')}",
             f"Resolution: {params.get('res')}",
             f"Seed: {params.get('seed', '?')}",
             f"Background removal: {params.get('bg_removal')}",
             f"Weights: {params.get('variant', 'f16')}"]
    if params.get("decim"):
        lines.append(f"Decimation target: {params['decim']}")
    if params.get("atlas"):
        lines.append(f"UV atlas: {params['atlas']}")
    flags = [k for k in ("no_texture", "box_uv", "f32", "no_fa")
             if params.get(k)]
    if flags:
        lines.append("Flags: " + ", ".join(flags))
    if params.get("gpu") is not None:
        lines.append(f"GPU: {params['gpu']}")
    lines.append(f"Engine: trellis.cpp (TRELLIS.2)")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        out_path.with_suffix(".txt").write_text("\n".join(lines),
                                                encoding="utf-8")
    except OSError:
        pass


def _post_generate(image_path: Path, out_path: Path, res: int,
                   seed: int | None, bg_removal: str, port: int,
                   _log: Callable[[str], None],
                   gpu_index: "int | None" = None) -> Path:
    """POST /generate (multipart) → écrit les octets GLB reçus."""
    data = {"resolution": str(int(res)), "bg_removal": bg_removal or "birefnet"}
    if seed is not None:
        data["seed"] = str(int(seed))
    try:
        with open(image_path, "rb") as fh:
            r = requests.post(f"http://127.0.0.1:{port}/generate",
                              files={"image": fh}, data=data, timeout=3600)
    except requests.exceptions.RequestException as exc:
        # Connexion coupée = le serveur est mort en cours de route. L'erreur
        # de socket ne dit rien d'utile ; la vraie cause a été capturée dans
        # sa sortie standard juste avant.
        raise sdcpp.EngineError(_diagnose_crash(gpu_index)) from exc
    if not r.ok:
        raise sdcpp.EngineError(
            f"trellis /generate failed (HTTP {r.status_code}) : "
            f"{r.text[:300]}")
    out_path.write_bytes(r.content)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise sdcpp.EngineError(
            "No GLB produced — see the log (not enough VRAM even at 512?).")
    _log(f"✅ GLB received: {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path
