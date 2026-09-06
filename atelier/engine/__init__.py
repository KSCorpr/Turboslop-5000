"""Paquet moteur.

Ce fichier ne contient qu'UNE fonction, et elle est ici parce que c'est le seul
endroit du paquet qui ne dépende de rien : `sdcpp`, `tools` et `trellis`
l'appellent tous les trois avant de prendre le GPU.

Elle attrape aussi l'ImportError du moteur résident. Ça n'est pas de la
paranoïa : l'application se met à jour en dézippant une archive par-dessus le
dossier existant, et une extraction partielle (fichier verrouillé par
l'application en cours, antivirus, copie interrompue) laisse un dossier où un
module récent manque. Le moteur résident est une OPTION — son absence doit
retirer l'option, pas empêcher l'application de démarrer.
"""
from __future__ import annotations

from typing import Callable


def resident_engine():
    """Le module du moteur résident, ou None s'il n'est pas installé."""
    try:
        from . import sdserver
    except ImportError:
        return None
    return sdserver


def release_resident_engine(reason: str = "",
                            log: Callable[[str], None] | None = None) -> None:
    """Rend la VRAM tenue par le moteur résident, s'il y en a un qui tourne."""
    server = resident_engine()
    if server is not None and server.is_running():
        server.stop(reason, log)


def engine_build_source() -> str:
    """D'où vient le moteur installé : « custom-ci », « official », ou "".

    L'archive de la CI du projet et celle de leejet ne contiennent pas la même
    chose : savoir laquelle est installée est la différence entre « il manque
    un fichier, débrouillez-vous » et « votre build ne l'a jamais contenu,
    voici le bouton ».
    """
    import json
    from .. import settings
    for path in settings.BIN_DIR.rglob("engine-build.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("source"):
            return str(data["source"])
    return ""
