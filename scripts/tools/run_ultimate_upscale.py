#!/usr/bin/env python3
"""Upscale créatif « Ultimate SD Upscale » — SDXL img2img tuilé (diffusers).

Réécriture autonome (sans dépendance A1111) de l'algorithme : pré-agrandissement
puis RAFFINAGE tuile par tuile en img2img à FAIBLE débruitage, avec recouvrement
+ fondu cosinus pour des coutures invisibles. Le modèle SDXL reste RÉSIDENT sur
le GPU → chaque tuile est rapide (pas de rechargement). Lancé en sous-process
pour ne pas verrouiller les DLL torch dans le process Gradio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# Choix du back-end de calcul (CUDA / Metal-MPS / CPU), partagé par les runners.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import label, pick_device, pick_dtype  # noqa: E402


def _round8(v: int) -> int:
    return max(8, int(round(v / 8)) * 8)


def _feather(h: int, w: int, fade: int):
    """Masque de fondu cosinus sur les bords (pour recoller les tuiles)."""
    import numpy as np
    fy = np.ones(h, np.float32)
    fx = np.ones(w, np.float32)
    f = max(1, min(fade, h // 2, w // 2))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f, dtype=np.float32))
    fy[:f] = ramp; fy[-f:] = ramp[::-1]
    fx[:f] = ramp; fx[-f:] = ramp[::-1]
    return (fy[:, None] * fx[None, :])[:, :, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--vae", default="",
                    help="VAE externe (dossier). Vide = VAE intégrée au modèle.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=0,
                    help="cible explicite (sinon scale × taille source)")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--denoise", type=float, default=0.35,
                    help="force du raffinage (0.2 fidèle, 0.5 inventif)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--prompt", default="")
    # Le négatif par défaut est orienté PHOTO. Sur du dessin il faut le
    # remplacer (« photorealistic, film grain, texture on flat colors… »),
    # sinon SDXL pose du grain et de la matière sur les aplats.
    ap.add_argument("--negative", default="")
    ap.add_argument("--preview-path", default="")
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--max-size", type=int, default=8192)
    ap.add_argument("--controlnet", default="",
                    help="dossier ControlNet Tile SDXL (verrouille la structure)")
    ap.add_argument("--cn-scale", type=float, default=0.6,
                    help="force du ControlNet (↑ = plus fidèle à la structure)")
    args = ap.parse_args()

    import warnings
    # Bruits connus et inoffensifs : flash-attention non compilé (PyTorch utilise
    # un autre noyau) et troncature CLIP à 77 tokens (limite SDXL ; pour un
    # upscale, un prompt court suffit largement).
    warnings.filterwarnings("ignore", message=".*[Ff]lash attention.*")
    warnings.filterwarnings("ignore", message=".*CLIP can only handle.*")

    import numpy as np
    import torch
    from PIL import Image
    try:
        from transformers.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass
    try:
        from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
    except ImportError:
        sys.exit("diffusers manquant. Réinstallez l'upscale SDXL (onglet Toolkit).")

    device = pick_device(torch)
    dtype = pick_dtype(torch, device)
    if device == "cpu":
        print("⚠️  Aucun GPU disponible : l'upscale SDXL tournerait sur CPU "
              "(très lent). Vérifiez les pilotes NVIDIA, ou côté Mac que MPS "
              "est bien actif.", flush=True)
    print(f"[usdu] calcul sur {label(device)}.", flush=True)

    # VAE : externe (fp16-fix) si fournie, sinon celle intégrée au checkpoint.
    load_kw = dict(torch_dtype=dtype, add_watermarker=False)
    if args.vae:
        load_kw["vae"] = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
        print("[usdu] VAE externe (fp16-fix).", flush=True)
    else:
        print("[usdu] VAE intégrée au modèle.", flush=True)
    # ControlNet Tile (optionnel) : conditionne chaque tuile sur la source -> on
    # peut pousser la créativité sans dériver de la structure d'origine.
    use_cn = bool(args.controlnet)
    if use_cn:
        try:
            from diffusers import (ControlNetModel,
                                   StableDiffusionXLControlNetImg2ImgPipeline)
        except ImportError:
            sys.exit("diffusers trop ancien pour ControlNet. Réinstallez l'upscale.")
        print(f"[usdu] chargement SDXL + ControlNet Tile sur {label(device)}…",
              flush=True)
        load_kw["controlnet"] = ControlNetModel.from_pretrained(
            args.controlnet, torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            args.base_model, **load_kw)
    else:
        print(f"[usdu] chargement SDXL sur {label(device)}…", flush=True)
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            args.base_model, **load_kw)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda" and args.low_vram:
        print("[usdu] VRAM serrée → offload CPU du modèle (plus lent mais tient).",
              flush=True)
        pipe.enable_model_cpu_offload()
    elif device != "cpu":
        # MPS compris : sans ce .to(), le pipeline resterait sur CPU côté Mac
        # et l'upscale prendrait des heures sans que rien ne l'indique.
        pipe.to(device)
    try:
        pipe.enable_attention_slicing()
        # Le VAE tiling découpe le décodage spatialement et laisse une GRILLE
        # fine sur les zones lisses (peau, fond). On ne l'active qu'en VRAM
        # serrée, où il est nécessaire pour éviter l'OOM.
        if args.low_vram:
            pipe.enable_vae_tiling()
    except Exception:  # noqa: BLE001
        pass

    src = Image.open(args.input).convert("RGB")
    if args.width and args.height:        # cible explicite (ex. après ESRGAN)
        tw, th = _round8(args.width), _round8(args.height)
    else:
        tw = _round8(int(src.width * args.scale))
        th = _round8(int(src.height * args.scale))
    if max(tw, th) > args.max_size:
        r = args.max_size / max(tw, th)
        tw, th = _round8(int(tw * r)), _round8(int(th * r))
        print(f"[usdu] cible plafonnée à {tw}x{th} (max {args.max_size}px).",
              flush=True)
    print(f"[usdu] pré-agrandissement {src.width}x{src.height} -> {tw}x{th} "
          "(Lanczos)…", flush=True)
    # Base Lanczos PROPRE (pas de pré-accentuation : elle introduit des halos et
    # du grain que la diffusion fige ensuite). SDXL ajoute le détail net.
    base = src.resize((tw, th), Image.LANCZOS)

    prompt = args.prompt or ("highly detailed, sharp focus, intricate fine "
                             "textures, photorealistic, high quality")
    negative = args.negative or ("blurry, jpeg artifacts, lowres, "
                                 "oversharpened, deformed")

    tile = _round8(max(512, args.tile))
    overlap = max(32, min(args.overlap, tile // 2))
    step = max(64, tile - overlap)
    xs = list(range(0, max(1, tw - overlap), step)) or [0]
    ys = list(range(0, max(1, th - overlap), step)) or [0]
    total = len(xs) * len(ys)
    print(f"[usdu] raffinage SDXL : {total} tuiles de {tile}px "
          f"(débruitage {args.denoise}, {args.steps} pas)…", flush=True)

    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw, 1), np.float32)
    gen = torch.Generator(device=device).manual_seed(0)
    n = 0
    for y in ys:
        for x in xs:
            x2, y2 = min(x + tile, tw), min(y + tile, th)
            x1, y1 = max(0, x2 - tile), max(0, y2 - tile)
            cw, ch = x2 - x1, y2 - y1              # taille réelle (dans l'image)
            crop = base.crop((x1, y1, x2, y2))
            # SDXL exige des dimensions multiples de 8 → on agrandit pour le
            # modèle puis on remappe exactement sur la tuile (pas de bord noir).
            inp = crop.resize((_round8(cw), _round8(ch)), Image.LANCZOS)
            n += 1
            print(f"[usdu]   tuile {n}/{total} ({x1},{y1})…", flush=True)
            kw = dict(prompt=prompt, negative_prompt=negative, image=inp,
                      strength=float(args.denoise),
                      num_inference_steps=int(args.steps),
                      guidance_scale=float(args.cfg), generator=gen)
            if use_cn:
                # Tile : l'image de contrôle est la tuile (agrandie) elle-même.
                kw["control_image"] = inp
                kw["controlnet_conditioning_scale"] = float(args.cn_scale)
            out = pipe(**kw).images[0]
            # Filtre EXPLICITE : le défaut de PIL est bicubique, qui laisse un
            # léger crénelage sur les diagonales à chaque remise à la taille de
            # tuile — répété sur des dizaines de tuiles, ça se voit.
            arr = np.asarray(
                out.convert("RGB").resize((cw, ch), Image.LANCZOS), np.float32)
            mask = _feather(ch, cw, overlap)
            acc[y1:y2, x1:x2] += arr * mask
            wsum[y1:y2, x1:x2] += mask
            if args.preview_path:
                cur = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype("uint8")
                pv = Image.fromarray(cur)
                if max(pv.size) > 1280:
                    r = 1280 / max(pv.size)
                    pv = pv.resize((int(pv.width * r), int(pv.height * r)))
                try:
                    pv.save(args.preview_path)
                except OSError:
                    pass

    final = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype("uint8")
    # Pas d'accentuation finale : SDXL fournit déjà un rendu net ; une passe
    # UnsharpMask ne ferait que ré-amplifier grain et coutures.
    result = Image.fromarray(final)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_usdu.png")
    result.save(dest)
    print(f"[usdu] image finale {result.width}x{result.height} : {dest}",
          flush=True)


if __name__ == "__main__":
    main()
