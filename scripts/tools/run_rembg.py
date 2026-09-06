#!/usr/bin/env python3
"""Runner de suppression d'arrière-plan (RMBG-1.4 via transformers).

Charge le modèle local (--model-dir, trust_remote_code) et produit un PNG RGBA
avec arrière-plan transparent. Lancé en sous-process pour ne pas verrouiller les
DLL de torch dans Gradio.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    try:
        from transformers import AutoModelForImageSegmentation
    except ImportError:
        sys.exit("transformers manquant. Réinstallez l'outil depuis le Toolkit.")

    device = pick_device(torch)
    print(f"[rembg] chargement du modèle sur {label(device)}…", flush=True)
    model = AutoModelForImageSegmentation.from_pretrained(
        args.model_dir, trust_remote_code=True)
    model.to(device).eval()

    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size
    print(f"[rembg] segmentation ({orig_w}x{orig_h})…", flush=True)

    # Pré-traitement RMBG-1.4 : 1024x1024, normalisation [0.5]/[1.0].
    im = img.resize((1024, 1024), Image.BILINEAR)
    arr = np.asarray(im, dtype="float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = (t - 0.5) / 1.0
    t = t.to(device)

    with torch.no_grad():
        res = model(t)
    # RMBG-1.4 renvoie une liste de sorties ; la plus fine est res[0][0].
    pred = res[0][0] if isinstance(res, (list, tuple)) else res
    pred = pred.squeeze().detach().cpu().float()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mask = Image.fromarray((pred.numpy() * 255).round().astype("uint8"), "L")
    mask = mask.resize((orig_w, orig_h), Image.BILINEAR)

    cutout = img.convert("RGBA")
    cutout.putalpha(mask)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_nobg.png")
    cutout.save(dest)
    print(f"[rembg] image écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
