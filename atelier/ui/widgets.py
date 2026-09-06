"""Réglages d'affichage communs aux composants image.

Gradio 6 a remplacé les `show_*_button` par une LISTE de boutons, et en a
profité pour changer le défaut : une image de sortie affiche désormais
`["download", "share", "fullscreen"]`. Le bouton « share » partage vers les
Discussions Hugging Face Spaces — sans objet dans une application qui tourne
sur votre machine, sans compte et sans réseau. Sous Gradio 5 il n'apparaissait
pas en local (`show_share_button=None` = « seulement sur Spaces »), donc le
laisser faire serait un bouton de plus, arrivé tout seul, qui ne fait rien
d'utile.

D'où ces listes nommées : elles disent en un endroit ce qu'on veut voir, et
elles sont passées explicitement. Un composant sans `buttons=` reprendrait le
défaut de Gradio, « share » compris — c'est le piège que ce module existe pour
fermer.
"""
from __future__ import annotations

# Sorties : on télécharge et on agrandit. Rien d'autre.
IMAGE_BUTTONS: list[str] = ["download", "fullscreen"]

# Entrées et aperçus intermédiaires : agrandir suffit, il n'y a rien à garder.
IMAGE_VIEW_ONLY: list[str] = ["fullscreen"]

# Galeries : idem, plus le téléchargement groupé quand plusieurs images sortent.
GALLERY_BUTTONS: list[str] = ["download", "fullscreen"]

# Champs texte : le bouton « copier » quand le contenu est fait pour être repris.
TEXT_COPY: list[str] = ["copy"]


# --------------------------------------------------------------------------- #
#  Bouton « Stop »
# --------------------------------------------------------------------------- #
# Tous les boutons d'arrêt appelaient `cancel()`, qui RENVOIE un message
# (« ⏹️ Génération annulée. »), avec `outputs=None`. Le message était donc
# calculé puis jeté : on appuyait sur Stop et rien ne le confirmait à l'écran.
# Gradio 6 le dit maintenant tout haut — « A function returned too many output
# values (needed: 0, returned: 1) » — mais le défaut est plus ancien que
# l'avertissement, et la bonne réponse n'est pas de le taire : c'est d'afficher
# le message.

def stop_into_status(button, cancel_fn, status, cancels) -> None:
    """Arrêt dont la confirmation va dans une zone d'état (elle est remplacée)."""
    button.click(lambda: cancel_fn(), outputs=[status], cancels=cancels)


def stop_into_log(button, cancel_fn, log, cancels) -> None:
    """Arrêt dont la confirmation s'AJOUTE au journal.

    Écrire dans un journal, c'est le remplacer : on relit donc son contenu en
    entrée pour poser la ligne à la suite, au lieu d'effacer la trace de ce
    qu'on vient d'interrompre — c'est précisément ce qu'on veut consulter après
    avoir appuyé sur Stop.
    """
    def _append(current):
        msg = cancel_fn()
        return f"{current}\n{msg}" if current else msg

    button.click(_append, inputs=[log], outputs=[log], cancels=cancels)
