#!/usr/bin/env python3
"""Runner d'estimation de profondeur (Depth Anything V2 via transformers).

Produit une carte de profondeur en niveaux de gris (clair = proche, sombre =
loin). Lancé en sous-process pour ne pas verrouiller les DLL de torch dans
Gradio.
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

    import torch
    from PIL import Image
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit("transformers manquant. Réinstallez l'outil depuis le Toolkit.")

    dev = pick_device(torch)
    # transformers.pipeline veut un index CUDA, -1 pour le CPU, ou une chaîne
    # de device pour le reste : « mps » ne se code pas en entier.
    device = 0 if dev == "cuda" else (-1 if dev == "cpu" else dev)
    print(f"[depth] chargement du modèle sur {label(dev)}…", flush=True)
    pipe = pipeline("depth-estimation", model=args.model_dir, device=device)

    img = Image.open(args.input).convert("RGB")
    print(f"[depth] estimation ({img.width}x{img.height})…", flush=True)
    res = pipe(img)

    depth = res.get("depth") if isinstance(res, dict) else None
    if depth is None:
        pd = res["predicted_depth"].squeeze().detach().cpu().numpy().astype("float32")
        pd = (pd - pd.min()) / (pd.max() - pd.min() + 1e-8)
        depth = Image.fromarray((pd * 255).round().astype("uint8"))
    depth = depth.convert("L")
    if depth.size != img.size:
        depth = depth.resize(img.size, Image.BICUBIC)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_depth.png")
    depth.save(dest)
    print(f"[depth] carte écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
