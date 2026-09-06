"""Choix du périphérique de calcul, commun à tous les runners PyTorch.

Trois back-ends possibles selon la machine : CUDA (PC NVIDIA), MPS (Metal, Mac
Apple Silicon), et CPU en dernier recours. Le détail qui compte : sur MPS, une
partie des opérateurs n'est pas implémentée. Sans repli, le runner meurt avec un
« NotImplementedError » au milieu du calcul ; avec PYTORCH_ENABLE_MPS_FALLBACK,
l'opérateur manquant passe sur CPU et le reste continue sur le GPU.
"""
from __future__ import annotations

import os


def pick_device(torch) -> str:
    """« cuda » | « mps » | « cpu », selon ce qui est réellement disponible."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # Doit être posé AVANT la première opération MPS pour être pris en
        # compte ; on le fait ici, au moment où l'on choisit le device.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def pick_dtype(torch, device: str):
    """float16 sur GPU, float32 sur CPU.

    MPS gère le float16 et y gagne autant qu'en CUDA. Le float32 est réservé au
    CPU, où le demi-précision est souvent plus lent que la simple."""
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def label(device: str) -> str:
    """Nom lisible pour le journal (l'utilisateur doit voir où ça tourne)."""
    return {"cuda": "GPU CUDA", "mps": "GPU Apple (Metal)",
            "cpu": "CPU (lent)"}.get(device, device)
