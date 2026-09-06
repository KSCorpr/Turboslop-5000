"""Détection matérielle (GPU NVIDIA + RAM) et profil d'optimisation.

Mono-GPU par choix : le profil automatique cible la meilleure carte NVIDIA
détectée (quant de diffusion selon la VRAM, encodeur selon la RAM,
flash-attention dès Turing, offload CPU, VAE tiling).
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

from .i18n import t


# --------------------------------------------------------------------------- #
#  Détection
# --------------------------------------------------------------------------- #
@dataclass
class Gpu:
    index: int
    name: str
    vram_gb: float
    # maxwell | pascal | volta | turing | ampere | ada | hopper | blackwell |
    # apple | unknown
    arch: str
    tensor_cores: bool
    # Capacité de calcul CUDA telle que RAPPORTÉE par le pilote (« 7.5 »), vide
    # si le pilote est trop ancien pour l'exposer. C'est la seule source fiable
    # : le nom commercial ne dit pas tout (une même série mélange des puces) et
    # c'est ce chiffre — pas le nom — qui décide si un binaire CUDA tournera.
    compute_cap: str = ""
    driver: str = ""
    bus_id: str = ""
    pcie_gen: str = ""
    pcie_width: str = ""

    @property
    def is_apple(self) -> bool:
        return self.arch == "apple"

    @property
    def sm(self) -> str:
        """« sm_75 » — l'identifiant que réclament les erreurs CUDA."""
        if self.compute_cap and "." in self.compute_cap:
            major, minor = self.compute_cap.split(".", 1)
            return f"sm_{major}{minor}"
        return _SM_BY_ARCH.get(self.arch, "")

    def label(self) -> str:
        """Libellé lisible et complet, pour les menus et les diagnostics."""
        bits = [f"{self.vram_gb:.0f} GB", self.arch]
        if self.sm:
            bits.append(self.sm)
        return f"{self.name} ({', '.join(bits)})"

    @property
    def pcie_label(self) -> str:
        """Lien PCIe courant, quand le pilote NVIDIA sait le rapporter."""
        if not (self.pcie_gen or self.pcie_width):
            return ""
        gen = f"Gen{self.pcie_gen}" if self.pcie_gen else "PCIe"
        width = f" x{self.pcie_width}" if self.pcie_width else ""
        return gen + width


# Capacité de calcul -> architecture. Table officielle NVIDIA ; c'est elle qui
# fait autorité, le nom commercial ne sert que de repli.
_ARCH_BY_CC = {
    (5, 0): "maxwell", (5, 2): "maxwell", (5, 3): "maxwell",
    (6, 0): "pascal", (6, 1): "pascal", (6, 2): "pascal",
    (7, 0): "volta", (7, 2): "volta",
    (7, 5): "turing",
    (8, 0): "ampere", (8, 6): "ampere", (8, 7): "ampere",
    (8, 9): "ada",
    (9, 0): "hopper",
    (10, 0): "blackwell", (10, 1): "blackwell", (10, 3): "blackwell",
    (12, 0): "blackwell", (12, 1): "blackwell",
}

# Architecture -> sm, pour les cartes dont le pilote ne rapporte pas la
# capacité de calcul (repli seulement).
_SM_BY_ARCH = {"maxwell": "sm_52", "pascal": "sm_61", "volta": "sm_70",
               "turing": "sm_75", "ampere": "sm_86", "ada": "sm_89",
               "hopper": "sm_90", "blackwell": "sm_120"}

# Architectures dotées de tensor cores (donc où flash-attention vaut le coup).
TENSOR_CORE_ARCHS = frozenset(
    {"volta", "turing", "ampere", "ada", "hopper", "blackwell"})


def _arch_from_cc(cc: str) -> tuple[str, bool] | None:
    """(architecture, tensor cores) depuis « 7.5 ». None si illisible."""
    try:
        major, minor = (int(x) for x in cc.strip().split(".", 1))
    except (ValueError, AttributeError):
        return None
    arch = _ARCH_BY_CC.get((major, minor))
    if arch is None:
        # Puce plus récente que cette table : au-delà de Volta, NVIDIA n'a
        # jamais retiré les tensor cores. Mieux vaut un nom d'architecture
        # inconnu qu'un profil dégradé sur une carte neuve.
        arch = "blackwell" if major >= 10 else "unknown"
    return arch, (arch in TENSOR_CORE_ARCHS)


def _arch_from_name(name: str) -> tuple[str, bool]:
    """Déduit l'architecture et la présence de tensor cores depuis le nom.

    REPLI uniquement : utilisé quand le pilote ne rapporte pas la capacité de
    calcul. Un nom commercial est une heuristique, pas une donnée."""
    n = name.upper()
    # RTX 50xx
    if re.search(r"RTX\s?50\d\d", n):
        return "blackwell", True
    # RTX 40xx
    if re.search(r"RTX\s?40\d\d", n):
        return "ada", True
    # RTX 30xx
    if re.search(r"RTX\s?30\d\d", n):
        return "ampere", True
    # RTX 20xx / TITAN RTX / Quadro RTX
    if re.search(r"RTX\s?20\d\d", n) or "TITAN RTX" in n:
        return "turing", True
    # GTX 16xx (Turing sans tensor cores grand public)
    if re.search(r"GTX\s?16\d\d", n):
        return "turing", False
    # GTX 10xx / TITAN X(p)
    if re.search(r"GTX\s?10\d\d", n) or "TITAN X" in n:
        return "pascal", False
    # Datacenter récents
    if any(x in n for x in ("H100", "H200", "B100", "B200", "GB200")):
        return "blackwell", True
    if any(x in n for x in ("A100", "A40", "A6000", "A5000", "A4000")):
        return "ampere", True
    if any(x in n for x in ("L40", "L4", "RTX 6000 ADA", "RTX 5000 ADA")):
        return "ada", True
    return "unknown", True


# Part de la mémoire unifiée qu'un Mac Apple Silicon laisse au GPU. macOS
# n'expose pas de « VRAM » : le GPU adresse la même mémoire que le CPU, et le
# plafond de travail conseillé tourne autour de 75 % du total. On garde cette
# fraction plutôt que d'annoncer la RAM entière, sinon le profil automatique
# choisirait des quantifications qui font swapper la machine.
_APPLE_GPU_SHARE = 0.75


@lru_cache(maxsize=1)
def _apple_gpu() -> "Gpu | None":
    """GPU intégré d'un Mac Apple Silicon, vu comme un GPU normal.

    Le reste de l'app raisonne en « une carte, tant de VRAM » : plutôt que de
    semer des cas particuliers partout, on présente la mémoire unifiée sous la
    même forme. `arch = "apple"` suffit ensuite à décider ce qui a du sens
    (pas d'offload : il n'y a qu'une seule mémoire)."""
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    name = "Apple Silicon"
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        if out:
            name = out          # ex. « Apple M3 Pro »
    except (OSError, subprocess.SubprocessError):
        pass
    ram = detect_ram_gb()
    return Gpu(0, name, round(ram * _APPLE_GPU_SHARE, 1), "apple", True)


def _nvidia_smi(fields: str) -> str | None:
    """`nvidia-smi --query-gpu=<fields>`, ou None s'il n'est pas exploitable."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=1)
def detect_gpus() -> tuple[Gpu, ...]:
    """Cartes détectées, avec leur architecture RÉELLE quand le pilote la donne.

    `compute_cap` n'existe que depuis les pilotes 510+ : sur un pilote plus
    ancien la requête échoue en bloc, donc on retente sans ce champ et on
    retombe sur la déduction par le nom. Un vieux pilote doit dégrader la
    finesse du diagnostic, pas empêcher l'application de démarrer.
    """
    apple = _apple_gpu()
    if apple is not None:
        return (apple,)
    full = "index,name,memory.total,compute_cap,driver_version"
    out, has_cc = _nvidia_smi(full), True
    if out is None:
        out, has_cc = _nvidia_smi("index,name,memory.total"), False
    if out is None:
        return ()
    # Le lien PCIe est interrogé séparément : certains pilotes anciens ne
    # connaissent pas ces champs. Une requête combinée ferait alors perdre
    # aussi compute_cap et driver_version, pourtant disponibles.
    pci: dict[int, tuple[str, str, str]] = {}
    pci_out = _nvidia_smi(
        "index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current")
    if pci_out:
        for row in pci_out.strip().splitlines():
            cols = [p.strip() for p in row.split(",")]
            if len(cols) >= 4:
                try:
                    pci[int(cols[0])] = (cols[1], cols[2], cols[3])
                except ValueError:
                    pass

    gpus: list[Gpu] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            vram = float(parts[2]) / 1024.0  # Mio -> Gio
        except ValueError:
            continue
        name = parts[1]
        cc = parts[3] if (has_cc and len(parts) > 3) else ""
        driver = parts[4] if (has_cc and len(parts) > 4) else ""
        found = _arch_from_cc(cc) if cc else None
        if found is None:
            arch, tc = _arch_from_name(name)
            cc = ""
        else:
            arch, tc = found
            # Seule exception à l'autorité de la capacité de calcul : les GTX
            # 16xx sont en 7.5 comme les RTX 20xx mais n'ont PAS de tensor
            # cores. Le nom est ici la seule façon de les distinguer.
            if re.search(r"GTX\s?16\d\d", name.upper()):
                tc = False
        bus, gen, width = pci.get(idx, ("", "", ""))
        gpus.append(Gpu(idx, name, round(vram, 1), arch, tc,
                        compute_cap=cc, driver=driver, bus_id=bus,
                        pcie_gen=gen, pcie_width=width))
    return tuple(gpus)


def free_vram_gb(index: int | None = None) -> float:
    """VRAM RÉELLEMENT libre, en Gio. 0 si indéterminable.

    Volontairement NON mise en cache, contrairement à `detect_gpus` : c'est une
    mesure d'instant, qui change selon ce que fait le reste de la machine. La
    budgéter à partir de la VRAM *totale* est ce qui fait planter une passe HD
    quand un navigateur ou un jeu occupe déjà la carte.
    """
    if _apple_gpu() is not None:
        return 0.0                      # mémoire unifiée : la notion n'a pas cours
    out = _nvidia_smi("index,memory.free")
    if out is None:
        return 0.0
    best = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx, free = int(parts[0]), float(parts[1]) / 1024.0
        except ValueError:
            continue
        if index is None:
            best = max(best, free)
        elif idx == index:
            return round(free, 1)
    return round(best, 1)


def used_vram_gb() -> dict[int, float]:
    """Mémoire GPU utilisée à l'instant T, pour le banc d'essai matériel."""
    out = _nvidia_smi("index,memory.used")
    if out is None:
        return {}
    used: dict[int, float] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used[int(parts[0])] = round(float(parts[1]) / 1024.0, 2)
        except ValueError:
            pass
    return used


@lru_cache(maxsize=1)
def cpu_name() -> str:
    """Modèle de processeur, pour les diagnostics (et le mode CPU)."""
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Windows":
            return (os.environ.get("PROCESSOR_IDENTIFIER") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or ""


@lru_cache(maxsize=1)
def detect_ram_gb() -> float:
    """RAM système totale en Gio (Windows + Linux, sans dépendance externe)."""
    import platform
    try:
        if platform.system() == "Windows":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullTotalPhys / (1024 ** 3), 1)
        if platform.system() == "Darwin":
            # macOS n'a pas /proc : sysctl donne la mémoire physique en octets.
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                          text=True, timeout=10).strip()
            return round(int(out) / (1024 ** 3), 1)
        # Linux / autres POSIX
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


# --------------------------------------------------------------------------- #
#  Profil d'optimisation
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    gpu: Gpu | None
    ram_gb: float
    quant: str                 # quant recommandé pour le modèle de diffusion
    enc_quant: str             # quant recommandé pour l'encodeur de texte
    diffusion_fa: bool
    offload_to_cpu: bool
    vae_tiling: bool
    clip_on_cpu: bool
    vae_on_cpu: bool
    notes: list[str] = field(default_factory=list)

    def flags(self) -> dict[str, bool]:
        return {
            "diffusion_fa": self.diffusion_fa,
            "offload_to_cpu": self.offload_to_cpu,
            "vae_tiling": self.vae_tiling,
            "clip_on_cpu": self.clip_on_cpu,
            "vae_on_cpu": self.vae_on_cpu,
        }


def _quant_for_vram(vram: float) -> str:
    if vram < 8:
        return "Q4_K_S"
    if vram < 12:
        return "Q4_K_M"
    if vram < 16:
        return "Q5_K_M"
    if vram < 24:
        return "Q6_K"
    return "Q8_0"


def _enc_quant_for_ram(ram: float) -> str:
    if ram < 16:
        return "Q4_K_M"
    if ram < 32:
        return "Q6_K"
    return "Q8_0"


# Échelle des quantifications, de la plus légère (rapide) à la plus lourde (qualité).
QUANT_LADDER = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
                "Q6_K", "Q8_0"]


def _shift_quant(q: str, delta: int) -> str:
    """Décale une quant de `delta` crans dans l'échelle (borné)."""
    try:
        i = QUANT_LADDER.index(q)
    except ValueError:
        return q
    return QUANT_LADDER[max(0, min(len(QUANT_LADDER) - 1, i + delta))]


# Générations RTX grand public. `bias` décale la quant : < 0 privilégie la
# VITESSE (cartes anciennes / VRAM serrée), > 0 la QUALITÉ (cartes récentes).
GENERATIONS: dict[str, dict] = {
    "gtx10": {"label": "GTX 10xx (Pascal)", "arch": "pascal", "bias": -1,
              "typical_vram": 8.0,
              "note": "Pascal (GTX 10xx / 1080 Ti): no tensor cores, "
                      "flash-attention disabled (barely helps). A light "
                      "quantization to compensate; the encoder offloaded to "
                      "RAM."},
    "rtx20": {"label": "RTX 20xx (Turing)", "arch": "turing", "bias": 0,
              "typical_vram": 8.0,
              "note": "Turing: flash-attention OK, no fp8 acceleration "
                      "(sd.cpp computes in fp16). VRAM often tight → light "
                      "quant to stay fast."},
    "rtx30": {"label": "RTX 30xx (Ampere)", "arch": "ampere", "bias": 0,
              "typical_vram": 12.0,
              "note": "Ampere: native bf16, well balanced. Quant by VRAM."},
    "rtx40": {"label": "RTX 40xx (Ada)", "arch": "ada", "bias": 1,
              "typical_vram": 16.0,
              "note": "Ada: very fast, large VRAM headroom → bump up one "
                      "quality step."},
    "rtx50": {"label": "RTX 50xx (Blackwell)", "arch": "blackwell", "bias": 1,
              "typical_vram": 16.0,
              "note": "Blackwell: recent architecture + large VRAM → high "
                      "quality."},
}


def generation_profile(gen_key: str, vram_gb: float | None = None,
                       ram_gb: float | None = None) -> Profile:
    """Profil d'optimisation CURATÉ pour une génération de carte RTX.

    S'appuie sur la VRAM réelle (si fournie) pour la quant et les flags mémoire,
    et applique un léger biais qualité/vitesse propre à la génération.
    """
    spec = GENERATIONS.get(gen_key) or GENERATIONS["rtx30"]
    vram = vram_gb if (vram_gb and vram_gb > 0) else spec["typical_vram"]
    ram = ram_gb if (ram_gb and ram_gb > 0) else detect_ram_gb()
    quant = _shift_quant(_quant_for_vram(vram), spec["bias"])
    enc_quant = _enc_quant_for_ram(ram or 16.0)
    return Profile(
        gpu=None, ram_gb=ram or 0.0, quant=quant, enc_quant=enc_quant,
        # Flash-attention : à partir de Turing. Désactivé sur Pascal (GTX 10xx).
        diffusion_fa=(spec.get("arch") != "pascal"),
        offload_to_cpu=(vram < 16),
        vae_tiling=(vram <= 12),
        clip_on_cpu=(vram < 8),
        vae_on_cpu=(vram < 6),
        notes=[t(spec["note"]),
               t("VRAM {vram} GB → diffusion {quant}, encoder {enc}.").format(
                   vram=f"{vram:.0f}", quant=quant, enc=enc_quant)],
    )


def auto_profile(gpu_index: int | None = None) -> Profile:
    """Construit un profil d'optimisation à partir du matériel détecté."""
    gpus = detect_gpus()
    ram = detect_ram_gb()
    notes: list[str] = []

    gpu: Gpu | None = None
    if gpus:
        if gpu_index is not None:
            gpu = next((g for g in gpus if g.index == gpu_index), None)
        if gpu is None:
            gpu = max(gpus, key=lambda g: g.vram_gb)  # par défaut : la plus grosse
        if len(gpus) > 1:
            notes.append(
                t("{n} GPUs detected — compute pinned to #{idx} ({name}). "
                  "Changeable in Settings.").format(
                    n=len(gpus), idx=gpu.index, name=gpu.name))

    if gpu is None:
        notes.append(t("No NVIDIA GPU detected: CPU mode (very slow). Check "
                       "drivers / nvidia-smi."))
        return Profile(None, ram, "Q4_K_M", "Q4_K_M",
                       diffusion_fa=False, offload_to_cpu=True, vae_tiling=True,
                       clip_on_cpu=True, vae_on_cpu=True, notes=notes)

    vram = gpu.vram_gb
    quant = _quant_for_vram(vram)
    enc_quant = _enc_quant_for_ram(ram)

    # Mac Apple Silicon : mémoire UNIFIÉE. « Décharger en RAM » n'y veut rien
    # dire — c'est la même mémoire que celle du GPU, donc l'offload n'économise
    # rien et ne fait qu'ajouter des copies. On le désactive, et on garde le
    # VAE tiling qui, lui, réduit vraiment le pic. Flash-attention est laissé de
    # côté : le chemin Metal de sd.cpp ne l'utilise pas comme les kernels CUDA.
    if gpu.is_apple:
        # L'échelle de l'encodeur suppose une RAM SÉPARÉE de la VRAM : sur PC
        # l'encodeur est déchargé côté système et ne dispute rien au modèle de
        # diffusion. Ici les deux tirent sur la même réserve, donc appliquer la
        # règle PC telle quelle ferait swapper la machine. On budgète l'encodeur
        # sur ce qui reste une fois la diffusion logée, soit environ la moitié.
        enc_quant = _enc_quant_for_ram(vram * 0.5)
        profile = Profile(
            gpu=gpu, ram_gb=ram, quant=quant, enc_quant=enc_quant,
            diffusion_fa=False, offload_to_cpu=False, vae_tiling=True,
            clip_on_cpu=False, vae_on_cpu=False, notes=notes,
        )
        notes.append(t("{name} — {ram} GB unified memory, ~{vram} GB usable "
                       "by the GPU -> {quant} diffusion.").format(
            name=gpu.name, ram=f"{ram:.0f}", vram=f"{vram:.0f}", quant=quant))
        notes.append(t("Unified memory: RAM offload is disabled (it saves "
                       "nothing here) and compute goes through Metal."))
        return profile

    # Flash-attention : on suit les TENSOR CORES, pas une liste de noms
    # d'architectures. C'est ce qui règle enfin le cas des GTX 16xx, en 7.5
    # comme les RTX 20xx mais dépourvues de tensor cores — elles héritaient
    # jusqu'ici du réglage « Turing » et de sa flash-attention inutile.
    fa = gpu.tensor_cores
    if not fa:
        notes.append(t("{name}: no tensor cores → flash-attention disabled "
                       "(it gains nothing here), slower generation.").format(name=gpu.name))

    profile = Profile(
        gpu=gpu, ram_gb=ram, quant=quant, enc_quant=enc_quant,
        diffusion_fa=fa,
        offload_to_cpu=(vram < 16),
        vae_tiling=(vram <= 12),
        clip_on_cpu=(vram < 8),
        vae_on_cpu=(vram < 6),
        notes=notes,
    )

    notes.append(t("VRAM {vram} GB ({arch}) -> diffusion in {quant}.").format(
        vram=f"{vram:.0f}", arch=gpu.arch, quant=quant))
    if ram:
        notes.append(t("RAM {ram} GB -> text encoder in {enc} (offloaded to "
                       "RAM, no VRAM cost).").format(
                        ram=f"{ram:.0f}", enc=enc_quant))
    if vram < 10:
        notes.append(t("Tight VRAM: prefer resolutions ≤ 768 px and a lower "
                       "quantization (generation will be slower)."))
    return profile


def rtx3060_1080ti_combo() -> tuple[Gpu, Gpu] | None:
    """Détecte le duo ciblé par le preset : RTX 3060 12 Go + GTX 1080 Ti.

    Renvoie toujours ``(gpu_generation, gpu_secondaire)``. On vérifie aussi la
    VRAM pour ne pas confondre la 3060 12 Go avec une 3060 Ti 8 Go : les deux
    noms sont proches, mais le profil mémoire n'est pas interchangeable.
    """
    gpus = detect_gpus()
    rtx = next((g for g in gpus
                if re.search(r"RTX\s*3060(?!\s*TI)", g.name.upper())
                and g.vram_gb >= 11.5), None)
    pascal = next((g for g in gpus
                   if re.search(r"GTX\s*1080\s*TI", g.name.upper())
                   and g.vram_gb >= 10.0), None)
    return (rtx, pascal) if rtx is not None and pascal is not None else None


def rtx3060_1080ti_prefs() -> dict:
    """Préférences sûres et mesurables pour le duo 3060 12 Go / 1080 Ti.

    Ampere exécute diffusion, encodeur et VAE (tensor cores, Flash Attention).
    Pascal garde le LLM d'amélioration de prompt et sert de réserve de poids à
    SeedVR2. Pas d'auto-fit/row split : la seconde carte est souvent sur un
    port PCIe x4, donc les échanges à chaque matmul peuvent coûter plus qu'ils
    ne rapportent.

    L'encodeur de texte était sur la 1080 Ti — une erreur mesurée à 38 s par
    image sur Krea 2. Le raisonnement d'origine ne valait que pour du
    STOCKAGE : la carte y calculait aussi, or le GP102 exécute le fp16 à 1/64
    de sa vitesse fp32 (c'est la puce, pas un réglage), et l'encodage de prompt
    est précisément un gros matmul fp16.
    """
    combo = rtx3060_1080ti_combo()
    if combo is None:
        raise ValueError("The RTX 3060 12 GB + GTX 1080 Ti pair was not detected.")
    main, secondary = combo
    return {
        "auto_optimize": False,
        "gpu_index": main.index,
        # LLM d'amélioration de prompt : il tourne SEUL, quand aucune image
        # n'est en cours, donc l'occuper là ne prend rien à personne.
        "text_gpu_index": secondary.index,
        # Encodeur sd.cpp : plus jamais sur la Pascal (cf. docstring).
        "encoder_gpu_index": main.index,
        # Ses POIDS restent en RAM, son CALCUL se fait sur la carte principale :
        # --params-backend décide de la résidence, pas du lieu d'exécution, et
        # sd.cpp transfère les poids au moment de s'en servir. Les 4 Go de
        # l'encodeur ne prennent donc rien à une carte de 12 Go qui doit déjà
        # loger 8,4 Go de diffusion.
        # Mono-GPU : CUDA_VISIBLE_DEVICES remappe la carte choisie en cuda0,
        # donc on écrit cuda0 et pas l'index nvidia-smi.
        "params_backend": "diffusion=cuda0,vae=cuda0,te=cpu",
        "auto_fit": False,
        "split_mode": "layer",
        "quant": "Q5_K_M",
        "enc_quant": "Q8_0",
        "cache_mode": "",
        "cache_option": "",
        "flags": {
            "diffusion_fa": True,
            # Les poids résident sur les cartes indiquées ci-dessus. Le mode
            # RAM reste disponible dans le benchmark comme solution de repli.
            "offload_to_cpu": False,
            "vae_tiling": True,
            "clip_on_cpu": False,
            "vae_on_cpu": False,
        },
    }


# --------------------------------------------------------------------------- #
#  Le SEUL arbitrage que l'utilisateur ait à rendre
# --------------------------------------------------------------------------- #
# Tout le reste (quant, offload, tiling, flash-attention) se déduit du matériel :
# l'application sait le faire, et la personne devant l'écran n'a aucun moyen de
# mieux répondre. Il reste UNE question à laquelle le matériel ne répond pas,
# parce qu'elle porte sur un goût : préférez-vous de la marge (ça passe toujours,
# c'est un peu moins fin) ou du détail (c'est plus beau, ça sature plus tôt) ?
#
# D'où un curseur à trois crans, et un seul. « bias » décale la quantification
# dans QUANT_LADDER et resserre ou relâche les options mémoire.
BIASES: dict[str, dict] = {
    "memory": {
        "shift": -1,
        "label": "🪶 More memory headroom",
        "why": "If you get out-of-memory errors, or if you generate at large "
               "sizes. The model is compressed one notch further and the app "
               "saves memory wherever it can.",
    },
    "balanced": {
        "shift": 0,
        "label": "⚖️ Balanced (recommended)",
        "why": "What your card can hold without a fight. This is the right "
               "choice as long as nothing bothers you.",
    },
    "quality": {
        "shift": 1,
        "label": "🎨 More detail",
        "why": "One notch less compression: the image gains a little "
               "fineness, and the card has less headroom. Take it if "
               "everything already fits comfortably.",
    },
}


def biased_profile(bias: str, gpu_index: int | None = None) -> Profile:
    """Le profil automatique, décalé d'un cran vers la mémoire ou la qualité."""
    prof = auto_profile(gpu_index)
    spec = BIASES.get(bias) or BIASES["balanced"]
    shift = spec["shift"]
    if not shift:
        return prof
    vram = prof.gpu.vram_gb if prof.gpu else 0.0
    prof.quant = _shift_quant(prof.quant, shift)
    if shift < 0:
        # Côté mémoire, on ne se contente pas de compresser : on rapatrie ce
        # qu'on peut hors de la carte. L'encodeur de texte est le meilleur
        # candidat — il ne sert qu'au début de la génération.
        prof.offload_to_cpu = True
        prof.vae_tiling = True
        prof.clip_on_cpu = prof.clip_on_cpu or vram < 12
    else:
        # Côté qualité, on relâche le tiling du VAE (qui coûte un peu de
        # qualité aux jointures) uniquement s'il reste vraiment de la place.
        # On NE touche PAS à l'offload : sur PC il libère de la VRAM pour un
        # coût de vitesse négligeable — le désactiver ne gagnerait rien et
        # ferait saturer plus tôt.
        if vram >= 12:
            prof.vae_tiling = False
    return prof


def bias_from_prefs(prefs: dict) -> str:
    """Retrouve le cran choisi en comparant la quant enregistrée à l'auto."""
    if prefs.get("auto_optimize", True):
        return "balanced"
    saved = prefs.get("quant")
    if not saved:
        return "balanced"
    auto = auto_profile(prefs.get("gpu_index")).quant
    try:
        delta = QUANT_LADDER.index(saved) - QUANT_LADDER.index(auto)
    except ValueError:
        return "balanced"
    if delta <= -1:
        return "memory"
    if delta >= 1:
        return "quality"
    return "balanced"


def summary_text() -> str:
    """Petit résumé lisible du matériel détecté (pour l'UI)."""
    gpus = detect_gpus()
    ram = detect_ram_gb()
    if not gpus:
        return t("⚠️ No NVIDIA GPU detected · RAM {ram} GB").format(
            ram=f"{ram:.0f}")
    if gpus[0].is_apple:
        # Une seule mémoire : parler de « VRAM » induirait en erreur, on dit
        # explicitement que c'est la part de la mémoire unifiée.
        g = gpus[0]
        return "\n".join([
            t("Unified memory: **{ram} GB**").format(ram=f"{ram:.0f}"), "",
            t("**Detected GPU:**"),
            f"- {g.name} · Metal · "
            + t("~{vram} GB addressable by the GPU").format(
                vram=f"{g.vram_gb:.0f}")])
    lines = [t("System RAM: **{ram} GB**").format(ram=f"{ram:.0f}"), "",
             t("**Detected GPUs:**")]
    for g in gpus:
        tc = t("tensor cores") if g.tensor_cores else t("no tensor cores")
        link = f" · {g.pcie_label}" if g.pcie_label else ""
        bus = f" · bus {g.bus_id}" if g.bus_id else ""
        lines.append(f"- #{g.index} — {g.name} · {g.vram_gb:.0f} GB · "
                     f"{g.arch} ({tc}){link}{bus}")
    return "\n".join(lines)
