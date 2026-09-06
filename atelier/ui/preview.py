"""Aperçu de secours : voir l'image même quand le navigateur refuse de la charger.

Le symptôme : on dépose une image dans un outil, **elle est bien reçue et bien
utilisée** — la décomposition part, le résultat est juste — mais la vignette
reste une icône cassée. Gênant partout, franchement pénible en mode manuel de
la décomposition en calques, où il faut voir l'image pour savoir où cliquer.

Ce module ne prétend pas corriger la cause : elle n'est pas reproductible ici
(six noms de fichiers hostiles et cinq formats s'affichent tous), et le
diagnostic de « Gestion & aide » est là pour l'identifier. Il contourne le
problème par un chemin qui NE PEUT PAS échouer.

Une vignette normale est une adresse : le navigateur redemande le fichier au
serveur (`/gradio_api/file=…`). Tout ce qui peut casser est là — le chemin, le
type MIME déduit de la base de registre Windows, le nom du fichier, une
requête refusée. Un **data: URI** ne demande rien à personne : les pixels sont
dans la page. Pas de requête, pas de chemin, pas de MIME, pas de nom de
fichier.

Ce qu'il ne fait PAS, et il faut le dire : cette vignette n'est pas cliquable.
Elle sert à voir, pas à désigner. En mode manuel, elle permet de repérer ce
qu'on vise avant de cliquer dans le composant, elle ne remplace pas le clic.

Volontairement replié par défaut : sur une machine où l'aperçu fonctionne,
c'est un doublon inutile.
"""
from __future__ import annotations

import base64
import io

# Côté long de la vignette. Assez pour se repérer dans une image, assez petit
# pour que le base64 (≈ 4/3 de la taille du PNG) ne pèse pas sur la page — un
# data: URI voyage dans le HTML, il n'est pas mis en cache.
MAX_SIDE = 720


def data_uri(img, max_side: int = MAX_SIDE) -> str:
    """`data:image/png;base64,…` réduit, ou `""` si l'image est inutilisable."""
    if img is None:
        return ""
    try:
        from PIL import Image
        im = img if hasattr(img, "size") else Image.open(img)
        im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else \
            im.convert("RGB")
        w, h = im.size
        longest = max(w, h) or 1
        if longest > max_side:
            scale = max_side / longest
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    except Exception:  # noqa: BLE001 - Pillow lève un peu de tout
        return ""
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def html(img, max_side: int = MAX_SIDE) -> str:
    """Vignette prête à poser dans un `gr.HTML`."""
    uri = data_uri(img, max_side)
    if not uri:
        return ("<p style='opacity:.7'>No image loaded — drop one above.</p>")
    return (f"<img src='{uri}' alt='preview' "
            "style='max-width:100%;height:auto;border-radius:8px'>")
