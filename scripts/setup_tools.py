#!/usr/bin/env python3
"""Installe les outils du Toolkit (Python embarqué, aucune commande à taper).

Outils :
  depth   -> Depth Anything V2 (Small) : carte de profondeur.
  bg      -> RMBG-1.4 : suppression d'arrière-plan (PNG transparent).
  sam     -> Segment Anything (facebook/sam-vit-base) : extraction d'objet au clic.
  enhance -> Qwen2.5-3B-Instruct : améliore un prompt brut (LLM).
  face    -> GFPGAN / RestoreFormer++ / CodeFormer + facexlib :
             restauration des visages.
  upscale -> SDXL base + VAE fp16-fix : upscale créatif tuilé (Ultimate SD Upscale).

Réutilise les helpers torch CUDA de _torch_setup (build adaptée au GPU,
sans verrouiller de DLL). Lançable depuis l'interface ou en ligne :
    python scripts/setup_tools.py depth
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from atelier import settings  # noqa: E402
from _torch_setup import ensure_torch_cuda, pin_numpy, sh  # noqa: E402

# Modèle léger (~100 Mo) : rapide, tourne même sur Pascal (GTX 10xx) et en CPU.
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
# RMBG-1.4 (~176 Mo) : code self-contained (pur torch), poids sur HF.
BG_REPO = "briaai/RMBG-1.4"
# Segment Anything (base, ~375 Mo) via transformers, depuis HF.
SAM_REPO = "facebook/sam-vit-base"
# CLIP pour l'étiquetage zéro-shot des zones segmentées. Le modèle « base
# patch32 » suffit largement : on lui demande de choisir parmi une vingtaine
# de catégories de scène, pas de faire de la reconnaissance fine.
CLIP_REPO = "openai/clip-vit-base-patch32"
# Améliorateur de prompt : petit LLM instruct (~6 Go fp16), tourne en sous-process.
ENHANCE_REPO = "Qwen/Qwen2.5-3B-Instruct"
# « Image → prompt » : modèle de VISION-langage, même famille que l'améliorateur
# ci-dessus (mêmes gabarits de chat, même licence Qwen Research). 3B, ~7,5 Go en
# bf16, chargé puis déchargé comme lui — donc aucune VRAM retenue pendant la
# génération. Le support natif (`Qwen2_5_VLForConditionalGeneration`) est arrivé
# dans transformers 4.49 : vérifié, la classe n'existe pas en 4.48.
DESCRIBE_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
# Épinglage PROPRE à cet outil, plus serré que le pin global : 4.45 suffirait
# aux autres add-ons mais ne connaît pas ce modèle, et l'échec serait un
# « KeyError: qwen2_5_vl » incompréhensible au premier clic.
DESCRIBE_TRANSFORMERS_PIN = "transformers>=4.49,<4.50"
# Restauration de visages. Les poids viennent des dépôts d'ORIGINE (pas d'un
# miroir personnel), et chacun est vérifié par SHA-256 — empreintes relevées
# une à une. Elles servent deux fois : refuser un fichier corrompu, et détecter
# un téléchargement tronqué DÉJÀ sur le disque (le cas le plus fréquent : une
# coupure réseau au milieu des 377 Mo de CodeFormer).
#
# Deux briques COMMUNES à tous les restaurateurs : la détection de visages et
# la segmentation qui sert au recollage. Elles ne dépendent pas du modèle
# choisi, et sans elles aucun ne fonctionne.
FACE_SHARED = (
    ("detection_Resnet50_Final.pth",
     "https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
     "detection_Resnet50_Final.pth", "détecteur de visages (~110 Mo)",
     "6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d"),
    ("parsing_parsenet.pth",
     "https://github.com/xinntao/facexlib/releases/download/v0.2.2/"
     "parsing_parsenet.pth", "segmentation du visage (~85 Mo)",
     "3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2"),
)
# Les trois restaurateurs, installés ENSEMBLE (~1 Go). Ils ne se valent pas
# selon l'image, et comparer sur son propre visage est la seule façon de
# trancher : une seconde procédure d'installation « au cas où » coûterait plus
# cher en confusion que ce gigaoctet en disque.
FACE_RESTORERS = (
    ("GFPGANv1.4.pth",
     "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
     "GFPGANv1.4.pth", "GFPGAN v1.4 — Apache-2.0 (~349 Mo)",
     "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"),
    ("RestoreFormer++.ckpt",
     "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/"
     "v1.0.0/RestoreFormer++.ckpt", "RestoreFormer++ — Apache-2.0 (~294 Mo)",
     "613fe52805f86bf8c2bffff08ae9f7a0b99f408be1bf221767af6183038be3a2"),
    ("codeformer.pth",
     "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/"
     "codeformer.pth", "CodeFormer — S-Lab 1.0, NON COMMERCIAL (~377 Mo)",
     "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7"),
)
FACE_URLS = FACE_SHARED + FACE_RESTORERS

# Upscale créatif tuilé : SDXL base (1 fichier) + VAE fp16-fix + ControlNet Tile
# (optionnel, verrouille la structure pour pousser la créativité sans dériver).
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FILE = "sd_xl_base_1.0.safetensors"
VAE_FIX_REPO = "madebyollin/sdxl-vae-fp16-fix"
CN_TILE_REPO = "xinsir/controlnet-tile-sdxl-1.0"


# --------------------------------------------------------------------------- #
#  Version de diffusers COMMUNE à tous les outils.
#
#  Tous les add-ons partagent le MÊME Python embarqué : deux outils qui exigent
#  des versions incompatibles s'écrasent mutuellement, et c'est le dernier
#  installé qui gagne. La borne HAUTE vient de torch : notre torch est 2.4.1
#  (choisi pour couvrir Pascal -> Ada, cf. _torch_setup), et son
#  torch._library.infer_schema ne sait PAS lire les annotations « X | None ».
#  Or diffusers >= 0.35 en utilise dans attention_dispatch.py, importé en
#  cascade au chargement du module transformers -> ValueError au tout premier
#  import. 0.33.1 précède attention_dispatch et couvre largement les API SDXL
#  (img2img + ControlNet) utilisées par l'upscale créatif.
#  ⚠️ Une seule valeur pour tout le monde : c'est ce qui garde l'environnement
#  cohérent quel que soit l'ordre d'installation des outils.
# --------------------------------------------------------------------------- #
DIFFUSERS_PIN = "diffusers==0.33.1"

# NumPy < 2 : impose par notre socle torch 2.4.1 / torchvision 0.19.
# Le passer DANS la commande pip (au lieu de le re-figer apres coup avec
# pin_numpy) change tout : le resolveur choisit alors lui-meme une version
# d'opencv-python compatible avec numpy 1.x, au lieu qu'on installe la derniere
# puis qu'on casse sa dependance en redescendant numpy. On ne devine plus la
# borne d'opencv — pip la trouve.
NUMPY_PIN = "numpy>=1.24,<2"

# transformers : borne BASSE pour les quatre add-ons qui l'utilisent
# (profondeur, detourage, SAM, ameliorateur), borne HAUTE dictee par torch.
#
# « <5 » ne suffisait pas : les 4.5x recents importent
# « from torch.distributed.tensor import DTensor », or ce module PUBLIC n'existe
# qu'a partir de torch 2.5 (en 2.4 c'est torch.distributed._tensor). Sur notre
# torch 2.4.1 ca donne un ImportError en cascade des que diffusers touche a
# transformers. On reste donc dans la generation contemporaine de torch 2.4 /
# diffusers 0.33.
# ⚠️ Cette borne est liee a _torch_setup : si torch passe un jour en >= 2.5
# (carte Blackwell, ou abandon de Pascal), elle peut etre relevee.
TRANSFORMERS_PIN = "transformers>=4.45,<4.50"

# Paquets dont on IMPOSE la version, quoi qu'en disent les requirements amont.
# Tous les add-ons partagent un seul Python : ce qui n'est pas borne ici finit
# par etre decide par le dernier « pip install » lance.
_PINS = {"diffusers": DIFFUSERS_PIN, "numpy": NUMPY_PIN,
         "transformers": TRANSFORMERS_PIN}


def install_depth():
    model_dir = settings.ROOT / "tools_repo" / "depth" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "pillow"])
    print(f"\nTéléchargement du modèle de profondeur ({DEPTH_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=DEPTH_REPO, local_dir=str(model_dir))
    pin_numpy()  # transformers peut réintroduire NumPy 2 -> on re-fige
    print("\n[OK] Depth Anything V2 installé (profondeur + normales).")


def install_bg():
    model_dir = settings.ROOT / "tools_repo" / "bg" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "scikit-image", "pillow"])
    print(f"\nTéléchargement du modèle de suppression d'arrière-plan ({BG_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=BG_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] RMBG-1.4 installé. Disponible dans l'onglet Toolkit.")


def install_clip():
    model_dir = settings.ROOT / "tools_repo" / "clip" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "pillow"])
    print(f"\nTéléchargement de CLIP ({CLIP_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=CLIP_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] CLIP installé — la décomposition en calques sait maintenant "
          "nommer et regrouper les zones.")


def install_sam():
    model_dir = settings.ROOT / "tools_repo" / "sam" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "pillow"])
    print(f"\nTéléchargement de Segment Anything ({SAM_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=SAM_REPO, local_dir=str(model_dir))
    pin_numpy()
    print("\n[OK] Segment Anything installé. Disponible dans l'onglet Toolkit.")


def install_enhance():
    model_dir = settings.ROOT / "tools_repo" / "enhance" / "model"
    ensure_torch_cuda()
    print("Installation de transformers + accelerate…")
    sh([sys.executable, "-m", "pip", "install",
        TRANSFORMERS_PIN, NUMPY_PIN, "accelerate", "safetensors"])
    print(f"\nTéléchargement de l'améliorateur de prompt ({ENHANCE_REPO}, ~6 Go)…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=ENHANCE_REPO, local_dir=str(model_dir),
                      allow_patterns=["*.json", "*.safetensors", "*.txt",
                                      "tokenizer*", "vocab*", "merges*"])
    pin_numpy()
    print("\n[OK] Améliorateur de prompt installé. Bouton « ✨ Améliorer ».")


def install_describe():
    model_dir = settings.ROOT / "tools_repo" / "describe" / "model"
    ensure_torch_cuda()
    print("Installation de transformers + accelerate…")
    sh([sys.executable, "-m", "pip", "install",
        DESCRIBE_TRANSFORMERS_PIN, NUMPY_PIN, "accelerate", "safetensors",
        "pillow"])
    print(f"\nTéléchargement du modèle image → prompt ({DESCRIBE_REPO}, "
          f"~7,5 Go)…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=DESCRIBE_REPO, local_dir=str(model_dir),
                      allow_patterns=["*.json", "*.safetensors", "*.txt",
                                      "tokenizer*", "vocab*", "merges*",
                                      "preprocessor*", "chat_template*"])
    pin_numpy()
    print("\n[OK] Image → prompt installé. Onglet Outils → « 📝 Image → prompt ».")


def _hf_fetch(fn, desc: str, manual_url: str, dest) -> None:
    """Téléchargement HF avec 3 tentatives + diagnostic réseau actionnable.

    Sur un poste d'entreprise, l'erreur huggingface_hub « cannot find the
    requested files / check your connection » cache presque toujours un proxy
    obligatoire, une inspection SSL ou un huggingface.co filtré — on l'explique
    au lieu de laisser le traceback brut."""
    import time
    for attempt in range(3):
        try:
            fn()
            return
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  [!] échec ({type(exc).__name__}) — nouvel essai "
                      f"dans {wait} s…")
                time.sleep(wait)
            else:
                print(f"\n[X] Téléchargement impossible : {desc}")
                print("    Causes fréquentes sur un réseau d'entreprise :")
                print("    • proxy obligatoire → définissez HTTPS_PROXY="
                      "http://proxy:port avant de lancer run.bat ;")
                print("    • inspection SSL → REQUESTS_CA_BUNDLE="
                      "chemin\\vers\\ca-entreprise.pem ;")
                print("    • huggingface.co filtré → téléchargez à la main :")
                print(f"      {manual_url}")
                print(f"      et placez le fichier dans : {dest}")
                raise




def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_release_file(url: str, dest: Path, desc: str, sha256: str) -> None:
    """Télécharge un poids depuis une release GitHub et vérifie son empreinte.

    Écrit d'abord dans un « .part » : un téléchargement coupé (réseau, Ctrl-C)
    ne laisse jamais derrière lui un fichier tronqué qui passerait pour valide
    au lancement suivant. Et un fichier déjà présent est quand même vérifié —
    c'est justement celui-là qui peut être à moitié écrit.
    """
    import time
    import urllib.request
    if dest.is_file():
        if _sha256(dest) == sha256:
            print(f"  [OK] {desc} déjà présent et vérifié.")
            return
        print(f"  [!] {dest.name} incomplet ou corrompu — retéléchargement.")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Turbo-Slop/1"})
    for attempt in range(3):
        try:
            print(f"\nTéléchargement : {desc}…", flush=True)
            with urllib.request.urlopen(req, timeout=120) as response, \
                    open(part, "wb") as out:
                shutil.copyfileobj(response, out, 1024 * 256)
            got = _sha256(part)
            if got != sha256:
                raise RuntimeError(f"empreinte inattendue ({got[:12]}…)")
            part.replace(dest)
            print(f"  [OK] {dest.name} vérifié.")
            return
        except Exception as exc:  # noqa: BLE001
            part.unlink(missing_ok=True)
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  [!] échec ({type(exc).__name__}) — nouvel essai dans "
                      f"{wait} s…")
                time.sleep(wait)
            else:
                print(f"\n[X] Téléchargement impossible : {desc}")
                print("    Sur un réseau d'entreprise, github.com est souvent")
                print("    filtré. Téléchargez le fichier à la main :")
                print(f"      {url}")
                print(f"    et placez-le dans : {dest.parent}")
                raise


def install_face():
    """Restauration de visages : CodeFormer (spandrel) + facexlib.

    On installe facexlib SANS ses dépendances : ses `install_requires` tirent
    numba et filterpy, qui ne servent qu'à son suiveur de visages en vidéo —
    dont on n'utilise rien. Et numba impose sa propre borne sur NumPy, qui
    entrerait en conflit avec celle de notre socle torch. Les modules qu'on
    importe réellement (détection, segmentation, recollage) n'ont besoin que de
    cv2, numpy, torch et torchvision.
    """
    model_dir = settings.ROOT / "tools_repo" / "face" / "model"
    ensure_torch_cuda()
    print("Installation de spandrel (architectures) + OpenCV…")
    sh([sys.executable, "-m", "pip", "install", NUMPY_PIN, "opencv-python",
        "spandrel>=0.4,<0.5", "spandrel-extra-arches>=0.2,<0.3", "pillow"])
    print("Installation de facexlib (détection et recollage), sans ses extras…")
    sh([sys.executable, "-m", "pip", "install", "--no-deps", "facexlib>=0.3"])
    for name, url, desc, sha in FACE_URLS:
        _fetch_release_file(url, model_dir / name, desc, sha)
    pin_numpy()
    print("\n[OK] Restauration de visages installée "
          "(onglet Toolkit → « 🙂 Visages »).")
    print("     Trois modèles disponibles : GFPGAN et RestoreFormer++ "
          "(Apache-2.0, usage commercial libre),")
    print("     CodeFormer (S-Lab 1.0, NON COMMERCIAL).")


def install_upscale():
    base = settings.ROOT / "tools_repo" / "upscale"
    ensure_torch_cuda()
    print(f"Installation de {DIFFUSERS_PIN} + accelerate…")
    sh([sys.executable, "-m", "pip", "install", *_PINS.values(),
        "accelerate", "safetensors", "omegaconf", "pillow"])
    from huggingface_hub import hf_hub_download, snapshot_download
    # Dossier où déposer des checkpoints SDXL perso (sélectionnables dans l'UI).
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)

    # MODE HORS-LIGNE : chaque composant DÉJÀ présent est conservé (aucun
    # téléchargement). Sur un réseau qui bloque huggingface.co (entreprise),
    # téléchargez les fichiers depuis un poste qui atteint HF et déposez-les
    # aux emplacements indiqués — puis relancez : l'install les détecte et saute.
    base_ck = base / "sd_xl_base_1.0.safetensors"
    vae_ok = (base / "vae").is_dir() and any((base / "vae").glob("*.safetensors"))
    cn_ok = (base / "controlnet").is_dir() and \
        any((base / "controlnet").glob("*.safetensors"))

    if base_ck.is_file():
        print(f"  [OK] checkpoint SDXL déjà présent ({SDXL_FILE}) — pas de "
              "téléchargement.")
    else:
        print(f"\nTéléchargement du checkpoint SDXL ({SDXL_REPO}/{SDXL_FILE}, "
              "~6,6 Go)…")
        _hf_fetch(lambda: hf_hub_download(repo_id=SDXL_REPO, filename=SDXL_FILE,
                                          local_dir=str(base)),
                  f"checkpoint SDXL ({SDXL_FILE})",
                  f"https://huggingface.co/{SDXL_REPO}/resolve/main/{SDXL_FILE}",
                  base)

    if vae_ok:
        print("  [OK] VAE fp16-fix déjà présente — pas de téléchargement.")
    else:
        print(f"\nTéléchargement de la VAE fp16-fix ({VAE_FIX_REPO})…")
        _hf_fetch(lambda: snapshot_download(repo_id=VAE_FIX_REPO,
                                            local_dir=str(base / "vae"),
                                            allow_patterns=["*.json", "*.safetensors"]),
                  "VAE fp16-fix",
                  f"https://huggingface.co/{VAE_FIX_REPO}/tree/main",
                  base / "vae")

    # ControlNet Tile : OPTIONNEL (l'upscale marche sans). Un échec ici n'arrête
    # pas l'install.
    if cn_ok:
        print("  [OK] ControlNet Tile déjà présent — pas de téléchargement.")
    else:
        print(f"\nTéléchargement du ControlNet Tile ({CN_TILE_REPO}, ~2,5 Go, "
              "optionnel)…")
        try:
            _hf_fetch(lambda: snapshot_download(repo_id=CN_TILE_REPO,
                                                local_dir=str(base / "controlnet"),
                                                allow_patterns=["*.json", "*.safetensors"]),
                      "ControlNet Tile",
                      f"https://huggingface.co/{CN_TILE_REPO}/tree/main",
                      base / "controlnet")
        except Exception:  # noqa: BLE001
            print("  [!] ControlNet Tile non installé (optionnel) — l'upscale "
                  "créatif fonctionne sans, décochez « ControlNet » dans l'UI.")

    pin_numpy()
    print("\n[OK] Upscale créatif SDXL installé "
          "(onglet Toolkit → Upscale créatif).")


def install_spandrel():
    """Only the optional SR engine and one small baseline model; no diffusion."""
    ensure_torch_cuda()
    sh([sys.executable, "-m", "pip", "install", NUMPY_PIN,
        "spandrel>=0.4,<0.5", "pillow"])
    base = settings.ROOT / "tools_repo" / "spandrel"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / "RealESRGAN_x4plus.pth"
    # Official author's release, downloaded atomically and validated before use.
    if not dest.is_file():
        import urllib.request
        import uuid
        part = base / f"{uuid.uuid4().hex}.part"
        try:
            url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                   "v0.1.0/RealESRGAN_x4plus.pth")
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out)
            import torch
            torch.load(part, map_location="cpu", weights_only=True)
            part.replace(dest)
        finally:
            part.unlink(missing_ok=True)
    from spandrel import ModelLoader
    model = ModelLoader().load_from_file(dest)
    if model.scale != 4:
        raise RuntimeError("Unexpected model scale")
    pin_numpy()
    print("Spandrel x4 installed. Import DAT/HAT x4 weights from the Toolkit "
          "to compare them with RealESRGAN.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["depth", "bg", "sam", "clip", "enhance",
                                     "describe", "face", "upscale", "spandrel"])
    args = ap.parse_args()
    settings.configure_hf_env()
    if args.tool == "depth":
        install_depth()
    elif args.tool == "bg":
        install_bg()
    elif args.tool == "sam":
        install_sam()
    elif args.tool == "clip":
        install_clip()
    elif args.tool == "enhance":
        install_enhance()
    elif args.tool == "describe":
        install_describe()
    elif args.tool == "face":
        install_face()
    elif args.tool == "upscale":
        install_upscale()
    elif args.tool == "spandrel":
        install_spandrel()


if __name__ == "__main__":
    main()
