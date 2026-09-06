"""Chemins du projet, préférences utilisateur persistées et localisation de sd-cli."""
from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Any
from .fileio import atomic_write_text

ROOT = Path(__file__).resolve().parent.parent

# Dossiers (créés au besoin). Tout est local au projet -> portable.
LORA_DIR = ROOT / "loras"
BIN_DIR = ROOT / "bin"
OUTPUT_DIR = ROOT / "outputs"
TMP_DIR = ROOT / "tmp"
USERDATA_DIR = ROOT / "userdata"

CONFIG_DIR = ROOT / "config"
PREFS_FILE = USERDATA_DIR / "preferences.json"

DEFAULT_MODELS_DIR = ROOT / "models"


def _read_models_dir() -> Path:
    """Dossier des modèles : `models/` du projet, ou un dossier EXTERNE choisi
    dans les Réglages (ex. un NVMe rapide).

    Lu au CHARGEMENT du module (donc changement = redémarrage requis) : des
    modules capturent ce chemin à l'import. Lecture volontairement minimale —
    pas d'appel à load_prefs()/ensure_dirs() ici, qui dépendent de ce chemin.
    """
    try:
        if PREFS_FILE.is_file():
            data = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            raw = (data.get("models_dir") or "").strip()
            if raw:
                p = Path(raw).expanduser()
                if p.is_absolute():
                    return p
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return DEFAULT_MODELS_DIR


MODELS_DIR = _read_models_dir()
CUSTOM_DIR = MODELS_DIR / "custom"   # modèles téléchargés manuellement ailleurs

# Préférences par défaut (surchargées par l'onglet Réglages, persistées en JSON).
DEFAULT_PREFS: dict[str, Any] = {
    "theme": "light",           # thème de l'interface : "light" | "dark"
    # Dossier des modèles. Vide/None = `models/` dans le projet. Un chemin
    # ABSOLU le déplace ailleurs (ex. "D:\\IA\\models" sur un NVMe).
    # Pris en compte au REDÉMARRAGE de l'application.
    "models_dir": None,
    "gpu_index": None,          # None = auto (meilleure carte détectée)
    # GPU secondaire dédié au TEXTE (améliorateur de prompt). None = même GPU
    # que la génération. Ex. : mettre la 1080 Ti ici. La génération d'images et
    # l'upscale SDXL restent TOUJOURS sur le GPU de génération.
    "text_gpu_index": None,
    # EXPÉRIMENTAL : GPU dédié à l'encodeur de texte dans sd.cpp (--backend te=).
    # None = comportement normal (encodeur sur le GPU principal / déchargé RAM).
    "encoder_gpu_index": None,
    # Résidence explicite des POIDS sd.cpp (`--params-backend`). Le calcul est
    # réglé séparément par `--backend`. Cette séparation évite qu'un encodeur
    # annoncé « sur la 2e carte » soit en réalité relu depuis la RAM à travers
    # un port PCIe lent. Vide = laisser le moteur / le profil décider.
    "params_backend": "",
    # EXPÉRIMENTAL : répartit AUTOMATIQUEMENT le modèle (diffusion/encodeur/VAE)
    # sur TOUS les GPU visibles selon leur VRAM (sd.cpp --auto-fit). Permet
    # d'utiliser la VRAM d'une 2e carte (ex. 1080 Ti) pour la DIFFUSION elle-même,
    # pas seulement l'encodeur. Remplace le split d'encodeur ci-dessus.
    "auto_fit": False,
    # Mode de découpe entre GPU : "layer" (blocs entiers, défaut) ou "row"
    # (lignes de matmul réparties, CUDA seulement).
    "split_mode": "layer",
    "auto_optimize": True,      # déduire les flags du matériel
    "quant": None,              # None = recommandé selon VRAM
    "enc_quant": None,          # None = recommandé selon RAM
    # Surcharges manuelles (utilisées seulement si auto_optimize = False)
    "flags": {
        "diffusion_fa": True,
        "offload_to_cpu": True,
        "vae_tiling": True,
        "clip_on_cpu": False,
        "vae_on_cpu": False,
    },
    # Convolution DIRECTE au lieu d'im2col (sd.cpp --diffusion-conv-direct /
    # --vae-conv-direct). Volontairement HORS du dictionnaire « flags » : ce
    # n'est pas une déduction matérielle, c'est un compromis à essayer, et le
    # mettre dans flags le ferait écraser par le profil auto et par les presets
    # « 1 clic ». Détecté sur le binaire : ignoré s'il ne connaît pas l'option.
    "conv_direct_diffusion": False,
    "conv_direct_vae": False,
    "hf_endpoint": "https://huggingface.co",
    "civitai_token": "",        # jeton Civitai (optionnel, pour les LoRA protégés)
    # Accélération par cache (sd.cpp docs/caching.md). "" = désactivé.
    # Modes DiT (Flux/Krea) : easycache | dbcache | taylorseer | cache-dit | spectrum
    "cache_mode": "",
    "cache_option": "",         # ex. "threshold=0.2" (easycache) — vide = défauts
    # Presets ciblés : contrairement au cache global ci-dessus, ils ne touchent
    # que le modèle nommé — { "<id>": {"mode": ..., "option": ...} }. Plus
    # exposé dans l'interface depuis que tous les modèles du catalogue tournent
    # en 4 à 8 pas, régime où le cache ne gagne rien ; se règle à la main pour
    # un modèle qu'on ferait tourner avec beaucoup plus de pas.
    "cache_by_model": {},
    # Exécution SEGMENTÉE (sd.cpp --max-vram) : autorise le moteur à découper
    # son graphe de calcul pour tenir dans un budget, au lieu d'allouer d'un
    # bloc et d'échouer si ça ne rentre pas.
    #   ""     -> désactivé pour la génération ordinaire (comportement d'avant)
    #   "auto" -> VRAM libre détectée moins une marge
    #   "6" / "cuda0=6,cuda1=4" -> plafond ferme
    # La passe HD l'utilise de toute façon : c'est là que le tout-ou-rien casse.
    "max_vram": "",
    # Streaming des couches depuis leur backend de paramètres. Requiert les
    # poids de diffusion en RAM (`--offload-to-cpu` ou params diffusion=cpu),
    # mais PAS `--max-vram`. Coûteux en bande passante PCIe : opt-in.
    "stream_layers": False,
}


def served_paths() -> list[str]:
    """Dossiers que le serveur web a le droit de servir, en plus du cache Gradio.

    Sans cette liste, une image désignée par son chemin D'ORIGINE — et non par
    une copie de cache — reçoit un 403, donc une icône cassée côté navigateur.
    C'est le cas de tout ce que l'application produit elle-même.

    Volontairement limité aux deux dossiers qui contiennent des IMAGES. En
    partage réseau (`--listen`), cette liste est ce que la machine expose :
    y ajouter models/ ou la racine du projet publierait bien plus que ça.
    """
    return [str(OUTPUT_DIR), str(TMP_DIR)]


def ensure_dirs() -> None:
    for d in (MODELS_DIR, CUSTOM_DIR, LORA_DIR, BIN_DIR, OUTPUT_DIR, TMP_DIR,
              USERDATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_prefs() -> dict[str, Any]:
    ensure_dirs()
    prefs = json.loads(json.dumps(DEFAULT_PREFS))  # copie profonde
    if PREFS_FILE.is_file():
        try:
            saved = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            prefs.update({k: v for k, v in saved.items() if k != "flags"})
            if isinstance(saved.get("flags"), dict):
                prefs["flags"].update(saved["flags"])
        except (json.JSONDecodeError, OSError):
            pass
    return prefs


def save_prefs(prefs: dict[str, Any]) -> None:
    ensure_dirs()
    atomic_write_text(PREFS_FILE, json.dumps(prefs, indent=2))


# --- localisation du binaire stable-diffusion.cpp --------------------------
def sd_cli_names() -> list[str]:
    """Noms possibles du binaire selon la version de stable-diffusion.cpp."""
    if platform.system() == "Windows":
        return ["sd-cli.exe", "sd.exe"]
    return ["sd-cli", "sd"]


def find_sd_cli() -> Path | None:
    for name in sd_cli_names():
        for candidate in BIN_DIR.rglob(name):
            if candidate.is_file():
                return candidate
    for name in sd_cli_names():
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def model_repo_dir(repo: str) -> Path:
    """Emplacement local d'un dépôt HF : models/<owner>__<name>/."""
    return MODELS_DIR / repo.replace("/", "__")


def child_env(gpu_index: "int | None" = None) -> dict:
    """Environnement des sous-process Python (outils, installeurs).

    Force l'UTF-8 côté ENFANT. Sans ça, sous Windows la sortie standard d'un
    sous-process hérite de la page de codes de la console (cp1252) : le moindre
    emoji affiché par du code tiers lève un UnicodeEncodeError et tue le
    process — souvent à l'import, avant même d'avoir commencé à travailler.
    On ne peut pas corriger le code tiers, mais on peut lui donner un stdout
    capable d'encoder ce qu'il écrit. Nos propres lecteurs décodent déjà en
    UTF-8, donc les deux bouts sont cohérents.
    """
    import os
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"   # stdout/stderr de l'enfant en UTF-8
    env["PYTHONUTF8"] = "1"             # mode UTF-8 global (PEP 540)
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def configure_hf_env() -> None:
    """Configure l'environnement Hugging Face pour des téléchargements fiables.

    hf_transfer est désactivé : sous Windows il provoque des verrous de fichier
    (os error 32). Le téléchargement HTTP standard est plus lent mais robuste.
    """
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    prefs = load_prefs()
    os.environ.setdefault("HF_ENDPOINT",
                          prefs.get("hf_endpoint", "https://huggingface.co"))
