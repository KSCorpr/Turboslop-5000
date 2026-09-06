"""Vocabulaire de scène pour l'étiquetage zéro-shot des calques (CLIP).

Ce fichier est volontairement une DONNÉE, pas du code : c'est lui qu'on
retouche quand les étiquettes tombent à côté, sans rien changer au pipeline.

Trois choses le rendent utilisable là où une simple liste de mots échouerait :

1. **Chaque entrée porte plusieurs formulations.** CLIP note une image contre un
   texte ; « voiture » et « une automobile vue de face » n'obtiennent pas le
   même score sur la même image. On garde le meilleur des variantes, ce qui
   rattrape les cadrages inhabituels.

2. **Il y a des catégories POUBELLE** (`_JUNK`). Sans elles, CLIP est forcé de
   choisir dans une liste d'objets et étiquette « voiture » un bout de bitume
   flou — un classifieur zéro-shot ne sait pas dire « rien ». Leur seule raison
   d'être est d'absorber ces zones pour qu'on puisse les écarter.

3. **Chaque entrée déclare une profondeur TYPIQUE** (`depth`, 0 = fond,
   1 = premier plan). Le ciel est derrière, le sol aussi, un personnage devant.
   Ça sert de départage quand le modèle de profondeur n'est pas installé, et
   ça évite l'aberration classique du ciel posé au premier plan.
"""
from __future__ import annotations

# (étiquette affichée, formulations pour CLIP, profondeur typique 0..1)
SCENE_VOCAB: list[tuple[str, list[str], float]] = [
    ("ciel",        ["the sky", "clouds in the sky", "a clear blue sky",
                     "an overcast grey sky"], 0.02),
    ("mer",         ["the sea", "the ocean", "a large body of water"], 0.15),
    ("montagne",    ["a mountain", "a rocky cliff", "a mountain range"], 0.2),
    ("building",    ["a building", "a house", "city buildings",
                     "an apartment block"], 0.3),
    ("ville",       ["a cityscape", "a town seen from far away"], 0.25),
    ("arbre",       ["a tree", "foliage", "trees and bushes"], 0.35),
    ("vegetation",  ["grass", "a field of vegetation", "a hedge"], 0.35),
    ("route",       ["a road", "asphalt road surface", "a race track"], 0.45),
    ("sol",         ["the ground", "a floor", "sand", "snow on the ground"],
                    0.45),
    ("eau",         ["a river", "a puddle", "water surface"], 0.4),
    ("mur",         ["a wall", "a fence", "a guardrail", "a barrier"], 0.5),
    # Ajoutées après une photo de course : la gerbe d'eau derrière la voiture
    # et le public derrière les grillages n'avaient aucune entrée où tomber, et
    # se faisaient donc étiqueter comme le premier objet venu — puis fusionner
    # avec lui.
    ("smoke",       ["smoke", "steam", "a cloud of spray", "mist", "fog"],
                    0.6),
    ("foule",       ["a crowd of people", "spectators behind a fence",
                     "a group of people standing"], 0.55),
    ("vehicle",    ["a car", "a racing car", "a truck", "a motorcycle",
                     "a vehicle"], 0.75),
    ("personne",    ["a person", "a man", "a woman", "a human face",
                     "a driver in a helmet"], 0.8),
    ("animal",      ["an animal", "a dog", "a cat", "a bird"], 0.75),
    ("objet",       ["an object", "a piece of equipment", "a sign",
                     "a lamp post"], 0.7),
    ("plante",      ["a potted plant", "a flower"], 0.7),
    ("meuble",      ["a piece of furniture", "a chair", "a table"], 0.65),
    ("nourriture",  ["food", "a meal on a plate"], 0.8),
    ("texte",       ["written text", "a logo", "lettering"], 0.85),
]

# Catégories dont la seule fonction est d'ABSORBER ce qui n'est rien
# d'identifiable, pour qu'on puisse l'écarter ensuite.
_JUNK: list[tuple[str, list[str], float]] = [
    ("_flou",       ["a blurry out of focus region", "motion blur"], 0.5),
    ("_texture",    ["a flat texture patch", "an abstract colour gradient",
                     "a uniform coloured surface"], 0.5),
    ("_fragment",   ["a meaningless fragment of an image",
                     "a random crop of a photo"], 0.5),
]

JUNK_LABELS = frozenset(name for name, _p, _d in _JUNK)


def entries() -> list[tuple[str, list[str], float]]:
    return SCENE_VOCAB + _JUNK


def prompts() -> tuple[list[str], list[int]]:
    """(textes à encoder, index de l'entrée d'origine pour chacun).

    Le patron « a photo of … » est celui sur lequel CLIP a été entraîné ; sans
    lui, les scores zéro-shot se dégradent nettement.
    """
    texts: list[str] = []
    owner: list[int] = []
    for i, (_name, variants, _d) in enumerate(entries()):
        for v in variants:
            texts.append(f"a photo of {v}")
            owner.append(i)
    return texts, owner


def typical_depth(label: str) -> float:
    for name, _v, d in entries():
        if name == label:
            return d
    return 0.5
