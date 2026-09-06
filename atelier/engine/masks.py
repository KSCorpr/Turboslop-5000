"""Nettoyage et description de masques binaires — numpy pur.

Écrit à la main plutôt qu'avec scipy/OpenCV : l'add-on de segmentation pèse
déjà lourd (PyTorch + transformers), et ces opérations tiennent en quelques
dizaines de lignes. Ajouter une dépendance de 40 Mo pour un remplissage de trous
serait un mauvais échange.

Les composantes connexes passent par un union-find sur les PLAGES de chaque
ligne, pas sur les pixels : sur une image d'un million de pixels, une union-find
par pixel en Python prendrait des secondes, alors que le nombre de plages se
compte en milliers. La même primitive sert au remplissage des trous — un trou
n'étant qu'une composante du complément qui ne touche aucun bord.
"""
from __future__ import annotations

import numpy as np


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Plages contiguës de True dans une ligne : [(début, fin exclue), …]."""
    if not row.any():
        return []
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Étiquette les composantes connexes (voisinage 4). (étiquettes, nombre).

    0 = fond. Les étiquettes retournées sont compactées à partir de 1.
    """
    h, w = mask.shape
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    rows: list[list[tuple[int, int, int]]] = []      # (début, fin, étiquette)
    prev: list[tuple[int, int, int]] = []
    for y in range(h):
        cur: list[tuple[int, int, int]] = []
        for s, e in _runs(mask[y]):
            lab = 0
            for ps, pe, plab in prev:
                if ps < e and s < pe:                # plages qui se chevauchent
                    lab = plab if lab == 0 else lab
                    union(lab, plab)
            if lab == 0:
                parent.append(len(parent))
                lab = len(parent) - 1
            cur.append((s, e, lab))
        rows.append(cur)
        prev = cur

    # Compactage des étiquettes après résolution des équivalences.
    remap: dict[int, int] = {}
    out = np.zeros((h, w), np.int32)
    for y, cur in enumerate(rows):
        for s, e, lab in cur:
            root = find(lab)
            if root not in remap:
                remap[root] = len(remap) + 1
            out[y, s:e] = remap[root]
    return out, len(remap)


def component_sizes(labels: np.ndarray, count: int) -> np.ndarray:
    """Nombre de pixels par étiquette (index 0 = fond)."""
    return np.bincount(labels.ravel(), minlength=count + 1)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Bouche les trous INTÉRIEURS (ce qui donne l'aspect « gruyère »).

    Un trou est une composante du complément qui ne touche aucun bord : le fond
    extérieur, lui, en touche forcément un.
    """
    inv = ~mask
    labels, n = label_components(inv)
    if n == 0:
        return mask.copy()
    outside = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])).tolist())
    outside.discard(0)
    holes = inv & ~np.isin(labels, list(outside))
    return mask | holes


def largest_components(mask: np.ndarray, min_area: int,
                       max_parts: int = 0) -> list[np.ndarray]:
    """Découpe un masque en ses morceaux connexes, du plus grand au plus petit.

    C'est le correctif le plus visible : un « calque » fait de trente taches
    éparpillées aux quatre coins de l'image n'est pas un calque, c'est du bruit
    que rien ne permet d'utiliser. Chaque morceau devient une zone à part, et
    ceux qui n'atteignent pas `min_area` disparaissent.
    """
    labels, n = label_components(mask)
    if n == 0:
        return []
    sizes = component_sizes(labels, n)
    order = np.argsort(sizes[1:])[::-1] + 1
    out = []
    for lab in order:
        if sizes[lab] < min_area:
            break
        out.append(labels == lab)
        if max_parts and len(out) >= max_parts:
            break
    return out


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Flou moyen séparable par sommes cumulées (rapide, sans dépendance)."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    pad = np.pad(a, ((radius, radius), (radius, radius)), mode="edge")
    cs = np.cumsum(pad, axis=0, dtype=np.float32)
    cs = np.vstack([np.zeros((1, cs.shape[1]), np.float32), cs])
    a = (cs[k:, :] - cs[:-k, :]) / k
    cs = np.cumsum(a, axis=1, dtype=np.float32)
    cs = np.hstack([np.zeros((cs.shape[0], 1), np.float32), cs])
    return (cs[:, k:] - cs[:, :-k]) / k


# --------------------------------------------------------------------------- #
#  Comparer des masques SANS les comparer pixel à pixel
#
#  Les deux opérations qui décident du découpage — « ces deux zones sont-elles
#  la même ? » et « se touchent-elles ? » — sont QUADRATIQUES en nombre de
#  zones. Menées à pleine résolution, elles rendent l'outil inutilisable dès
#  qu'on sort du petit format : mesuré sur une image 4096×4096, un seul calcul
#  d'IoU coûte 77 ms, soit 14 MINUTES pour 150 zones, auxquelles s'ajoutent
#  70 s d'adjacence. Le tout pour des réponses qui ne dépendent pas du dernier
#  pixel.
#
#  On calcule donc UNE FOIS par masque une grille de couverture — la fraction
#  de pixels allumés dans chaque bloc — et toutes les comparaisons se font
#  dessus. 256×256 blocs contre 16,8 millions de pixels : trois ordres de
#  grandeur, et une grille pèse 262 Ko au lieu de 16,8 Mo.
# --------------------------------------------------------------------------- #
COMPARE_GRID = 256


def coverage_grid(mask: np.ndarray, side: int = COMPARE_GRID) -> np.ndarray:
    """Fraction de pixels allumés par bloc (float32, 0..1).

    Volontairement une MOYENNE et pas un « ou » : la moyenne conserve les
    surfaces, ce qu'exige l'IoU. Un « ou » ferait grossir les petites zones et
    ferait passer pour identiques deux taches voisines mais distinctes.
    """
    h, w = mask.shape
    step = max(1, int(np.ceil(max(h, w) / side)))
    if step == 1:
        return mask.astype(np.float32)
    ph, pw = (-h) % step, (-w) % step
    if ph or pw:
        mask = np.pad(mask, ((0, ph), (0, pw)))
    return mask.reshape(mask.shape[0] // step, step,
                        mask.shape[1] // step, step
                        ).mean(axis=(1, 3), dtype=np.float32)


def grid_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU approché à partir de deux grilles de couverture.

    `min` et `max` bloc à bloc jouent le rôle de l'intersection et de l'union :
    exact quand les blocs sont pleins ou vides, très proche sinon. Utilisé pour
    reconnaître un QUASI-DOUBLON (seuil 0,75), pas pour mesurer une surface.
    """
    inter = float(np.minimum(a, b).sum())
    if inter <= 0.0:
        return 0.0
    return inter / float(np.maximum(a, b).sum())


def grid_touches(a: np.ndarray, b: np.ndarray, gap: int = 2) -> bool:
    """Deux grilles de couverture se touchent-elles à `gap` blocs près ?"""
    sa, sb = a > 0, b > 0
    if not sa.any() or not sb.any():
        return False
    return bool((dilate(sa, max(1, gap)) & sb).any())


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Épaissit un masque de `radius` cases (voisinage carré, séparable).

    Décalages avec REMPLISSAGE À ZÉRO, surtout pas `np.roll` : celui-ci fait
    réapparaître à droite ce qui sort à gauche. Une zone collée au bord gauche
    devenait donc voisine d'une zone collée au bord droit — l'inverse exact de
    ce que cette fonction sert à établir.
    """
    if radius < 1:
        return mask
    out = mask
    for axis in (0, 1):
        acc = out
        n = out.shape[axis]
        for shift in range(1, radius + 1):
            if shift >= n:
                break
            fwd = np.zeros_like(out)
            bwd = np.zeros_like(out)
            if axis == 0:
                fwd[shift:, :] = out[:-shift, :]
                bwd[:-shift, :] = out[shift:, :]
            else:
                fwd[:, shift:] = out[:, :-shift]
                bwd[:, :-shift] = out[:, shift:]
            acc = acc | fwd | bwd
        out = acc
    return out


def touches(a: np.ndarray, b: np.ndarray, gap: int = 2) -> bool:
    """Les deux zones se TOUCHENT-elles réellement (à `gap` cases près) ?

    Cette fonction existe à cause d'un bug précis, et son intérêt est de
    remplacer ce qu'on faisait avant : tester le chevauchement des BOÎTES
    ENGLOBANTES. Sur une image large, la boîte d'un objet allongé couvre
    presque tout le cadre ; deux zones aux extrémités opposées se retrouvaient
    donc « voisines » alors qu'elles n'ont pas un pixel en commun. Résultat sur
    une photo de course : la voiture, la fumée à l'autre bout et le grillage du
    fond fusionnaient en un seul calque de 25 % de l'image.

    On compare donc les PIXELS, sur une grille réduite pour que ça reste peu
    coûteux.

    Commodité pour un appel isolé : dans une boucle, calculez les grilles une
    fois avec `coverage_grid()` et utilisez `grid_touches()` — sinon la
    réduction est refaite à chaque paire, ce qui est exactement le coût qu'on
    cherche à éviter.
    """
    return grid_touches(coverage_grid(a), coverage_grid(b), gap)


def feather_alpha(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Alpha 0-255 avec un bord adouci.

    Les masques de SAM sont binaires : collés tels quels, les découpes ont ce
    bord en marches d'escalier qui trahit le détourage automatique. Un ou deux
    pixels de transition suffisent à le faire disparaître.
    """
    a = _box_blur(mask.astype(np.float32), radius)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Description : donner un NOM utilisable à une zone.
#
#  « Zone 9 — 0,48 % » n'apprend rien à personne. Trois informations qu'on a
#  déjà sous la main suffisent à rendre la liste de calques lisible sans
#  ouvrir chaque vignette : où c'est, quelle taille, et de quelle couleur.
# --------------------------------------------------------------------------- #
_COLOR_NAMES = [
    ((15, 15, 18), "black"), ((70, 70, 75), "dark grey"),
    ((140, 140, 145), "grey"), ((205, 205, 210), "light grey"),
    ((248, 248, 248), "white"),
    ((190, 40, 40), "red"), ((235, 130, 40), "orange"),
    ((240, 220, 90), "yellow"), ((70, 160, 70), "green"),
    ((35, 90, 55), "dark green"),
    ((60, 120, 210), "blue"), ((150, 190, 230), "light blue"),
    ((25, 50, 110), "dark blue"),
    ((120, 70, 170), "purple"), ((215, 120, 170), "pink"),
    ((130, 90, 60), "brown"), ((235, 210, 175), "beige"),
]


def dominant_color_name(rgb: np.ndarray, mask: np.ndarray) -> str:
    """Nom de la couleur MÉDIANE de la zone (la médiane résiste aux reflets).

    Comparaison pondérée façon luminance : l'œil sépare d'abord le clair du
    sombre. Une distance euclidienne brute en RGB fait passer un gris anthracite
    pour du vert foncé — les trois canaux y pèsent pareil alors qu'ils ne
    comptent pas pareil.
    """
    if not mask.any():
        return ""
    med = np.median(rgb[mask].reshape(-1, 3), axis=0).astype(np.float32)
    best, score = "", None
    for ref, label in _COLOR_NAMES:
        c = np.array(ref, np.float32)
        # Écart de luminosité, puis écart de teinte une fois la luminosité ôtée.
        dl = abs(float(med.mean()) - float(c.mean()))
        dh = float(np.abs((med - med.mean()) - (c - c.mean())).sum())
        s = dl * 1.6 + dh
        if score is None or s < score:
            best, score = label, s
    return best


def position_name(mask: np.ndarray) -> str:
    """“top left”, “centre”, “bottom right”… from the centre of mass."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return ""
    h, w = mask.shape
    cy, cx = ys.mean() / h, xs.mean() / w
    vert = "top" if cy < 0.34 else ("bottom" if cy > 0.66 else "middle")
    horiz = "left" if cx < 0.34 else ("right" if cx > 0.66 else "centre")
    if vert == "middle" and horiz == "centre":
        return "centre"
    return f"{vert} {horiz}".replace("middle ", "").replace(" centre", "")


def depth_band_name(rank: int, total: int) -> str:
    """Bande de profondeur : c'est ce qui compte le plus pour un calque."""
    if total <= 1:
        return "single plane"
    r = rank / max(1, total - 1)
    if r < 0.34:
        return "background"
    if r > 0.66:
        return "foreground"
    return "middle ground"


def describe(mask: np.ndarray, rgb: np.ndarray, rank: int, total: int) -> str:
    """Nom de calque lisible : plan, position, couleur, taille."""
    pct = 100.0 * float(mask.sum()) / mask.size
    bits = [depth_band_name(rank, total)]
    pos = position_name(mask)
    if pos:
        bits.append(pos)
    col = dominant_color_name(rgb, mask)
    if col:
        bits.append(col)
    return " · ".join(bits) + f" — {pct:.1f}%"
