"""Sélection tolérante de la quantification d'un fichier GGUF.

Les dépôts n'ont pas toujours la quantification exacte demandée (ex. on veut
Q4_K_M mais seuls Q4_K et Q8_0 existent). On choisit alors la plus proche.
"""
from __future__ import annotations

# Échelle ordonnée approximative qualité/taille (du plus léger au plus lourd).
ORDER = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K", "Q3_K_L",
    "Q4_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K",
    "Q5_K_S", "Q5_0", "Q5_1", "Q5_K_M", "Q5_K",
    "Q6_K", "Q8_0", "Q8_1", "F16", "BF16", "F32",
]
# Recherche du plus long token d'abord (Q4_K_M avant Q4_K).
_BY_LEN = sorted(ORDER, key=len, reverse=True)


def find_quant(name: str) -> str | None:
    up = name.upper()
    for q in _BY_LEN:
        if q in up:
            return q
    return None


def _idx(q: str | None) -> int | None:
    if q is None:
        return None
    try:
        return ORDER.index(q)
    except ValueError:
        return None


def best(names: list[str], requested: str) -> str | None:
    """Parmi `names`, renvoie le quant le mieux adapté à `requested`.

    On ARRONDIT VERS LE BAS : à défaut du quant exact, on prend le plus gros
    quant **inférieur ou égal** au demandé (pour respecter le budget VRAM). On
    ne monte au-dessus que si AUCUN quant ≤ demandé n'existe. Monter en quant
    (ex. Q5 demandé -> Q6 téléchargé) gonfle la VRAM et provoque OOM/lenteur :
    c'est précisément ce qu'on évite ici.

    Tri par paliers : (0) ≤ demandé, le plus proche par en-dessous ; (1) > demandé,
    le plus proche par au-dessus ; (2) quant non reconnu. Départage : nom court.
    """
    if not names:
        return None
    req = _idx(requested)

    def key(name: str):
        qi = _idx(find_quant(name))
        if qi is None:
            return (2, 0, len(name))
        if req is None:
            return (0, -qi, len(name))
        if qi <= req:
            return (0, req - qi, len(name))   # ≤ demandé : le plus haut d'abord
        return (1, qi - req, len(name))        # > demandé : le plus bas d'abord

    return sorted(names, key=key)[0]
