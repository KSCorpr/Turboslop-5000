"""Outpaint directionnel (façon Midjourney) : étendre une image à gauche,
à droite, en haut, en bas — ou tout autour.

CE QUI DÉTERMINE LA QUALITÉ, avant tout réglage : le modèle doit VOIR l'image.
Un modèle d'ÉDITION (Flux.2 Klein) reçoit la toile en référence
(-r) : son conditionnement image lui donne le contenu de la scène, et une
consigne d'extension (`instruction()`) lui dit quoi en faire. Un modèle de
text-to-image ordinaire n'a, lui, qu'un latent bruité en img2img : il ne sait
pas ce qu'il prolonge, donc il réinvente — bords incohérents garantis, quels
que soient la force, le fondu ou le remplissage. Le repli img2img existe pour
ne pas bloquer, pas parce qu'il donne un bon résultat.

Chaîne complète :

1. `plan()`         — géométrie : combien de pixels de chaque côté, toile finale.
2. `build_canvas()` — la nouvelle zone est pré-remplie à partir des bords.
3. `build_mask()`   — masque d'inpainting (blanc = à générer, noir = à garder).
   Utilisé UNIQUEMENT si le binaire sd-cli installé connaît l'option de masque
   (détectée à l'exécution, cf. sdcpp.mask_flag). C'est le mode de qualité : la
   zone d'origine n'est pas rebruitée du tout.
4. `match_tone()`   — recale le contraste/la couleur du résultat sur l'original.
   INDISPENSABLE : en img2img le modèle re-rend toute la toile avec un contraste
   plus marqué ; recoller l'original tel quel laissait alors un rectangle
   visiblement plus terne au centre. On corrige le NEUF pour qu'il rejoigne
   l'ancien (jamais l'inverse : l'original doit rester intact).
5. `composite_back()` — l'original est recollé, avec un fondu de raccord.

Les étapes 4 et 5 garantissent que la zone d'origine est préservée au pixel près
et que la jonction ne se voit pas, même quand le moteur n'a pas de masque — d'où
la compatibilité avec n'importe quel modèle du catalogue.
"""
from __future__ import annotations

DIRECTIONS = ["left", "right", "top", "bottom"]

# Modes de pré-remplissage de la nouvelle zone.
FILLS = [
    ("Neutral grey (editing models)", "neutral"),
    ("Blurred stretch", "edge"),
    ("Miroir (motifs, textures)", "mirror"),
]


def instruction(p: dict, user_prompt: str = "") -> str:
    """Consigne d'extension envoyée au modèle d'édition.

    C'est ELLE qui remplace le « sans prompt » : l'utilisateur n'écrit rien, mais
    le modèle reçoit une instruction explicite. Un modèle d'édition ne devine pas
    qu'on veut prolonger une image — sans consigne, il se contente de reproduire
    ou de réinventer, d'où des bords incohérents.
    """
    fr = {"left": "left", "right": "right", "top": "top", "bottom": "bottom"}
    sides = [fr[d] for d in DIRECTIONS if p.get(d)]
    if len(sides) > 1:
        where = ", ".join(sides[:-1]) + " and " + sides[-1]
    else:
        where = sides[0] if sides else "border"
    base = (
        "Outpainting task. The image has empty margins added on the "
        f"{where}. Fill ONLY those empty margins by continuing the existing "
        "scene outward, as if the photograph or artwork had always been wider. "
        "Keep the exact same perspective, horizon line, lighting direction, "
        "color palette, materials, grain and art style. Objects cut off at the "
        "old border must continue naturally into the new space. Do not add new "
        "subjects, do not repeat or mirror existing ones, and do not alter the "
        "original centre of the image in any way.")
    extra = (user_prompt or "").strip()
    return f"{base} In the new area: {extra}" if extra else base


def plan(size: tuple[int, int], directions: list[str], amount: float,
         multiple: int = 16, max_side: int = 2048) -> dict:
    """Calcule la nouvelle toile et la position de l'original.

    `amount` = proportion de la dimension d'origine ajoutée par côté (0.25 = +25 %).
    La toile est alignée sur `multiple` (contrainte des modèles) et plafonnée.
    """
    w, h = size
    dirs = [d for d in (directions or []) if d in DIRECTIONS]
    pad_x = int(round(w * amount)) if amount > 0 else 0
    pad_y = int(round(h * amount)) if amount > 0 else 0
    left = pad_x if "left" in dirs else 0
    right = pad_x if "right" in dirs else 0
    top = pad_y if "top" in dirs else 0
    bottom = pad_y if "bottom" in dirs else 0

    new_w, new_h = w + left + right, h + top + bottom
    # Plafond : on réduit les marges proportionnellement si on dépasse.
    def _clamp(new, orig, a, b):
        if new <= max_side:
            return a, b, new
        excess = new - max_side
        total = a + b
        if total <= 0:
            return a, b, new
        a2 = max(0, a - int(round(excess * a / total)))
        b2 = max(0, b - (excess - (a - a2)))
        return a2, b2, orig + a2 + b2

    left, right, new_w = _clamp(new_w, w, left, right)
    top, bottom, new_h = _clamp(new_h, h, top, bottom)

    # Alignement sur `multiple` : on rallonge la marge existante (jamais 0, pour
    # ne pas déplacer l'original si un côté n'est pas étendu).
    def _snap(new, a, b):
        rem = new % multiple
        if rem:
            add = multiple - rem
            if b > 0:
                b += add
            elif a > 0:
                a += add
            else:
                return a, b, new      # aucune extension sur cet axe
            new += add
        return a, b, new

    left, right, new_w = _snap(new_w, left, right)
    top, bottom, new_h = _snap(new_h, top, bottom)
    return {"left": left, "right": right, "top": top, "bottom": bottom,
            "width": new_w, "height": new_h, "orig": (w, h)}


def build_canvas(img, p: dict, fill: str = "edge"):
    """Toile agrandie, nouvelles zones pré-remplies à partir des bords.

    `fill="neutral"` — gris uni. À utiliser avec un MODÈLE D'ÉDITION : il voit
    l'image via son encodeur vision, et une zone franchement vide se lit comme
    « à remplir ». Un faux décor l'induirait en erreur.
    `fill="edge"` — les pixels du bord sont ÉTIRÉS vers l'extérieur puis floutés :
    on ne transmet que la couleur et la luminosité, aucune forme. Pour les
    modèles SANS édition, qui n'ont que ça comme point de départ.
    `fill="mirror"` — reflet des bords : la continuité est parfaite pour un motif
    ou une texture, MAIS un sujet proche du bord est dupliqué en miroir, et le
    modèle transforme volontiers ce reflet en un second objet bien réel. À
    réserver aux fonds réguliers.
    """
    from PIL import Image, ImageFilter, ImageOps
    src = img.convert("RGB")
    w, h = src.size
    left, top = p["left"], p["top"]
    right, bottom = p["right"], p["bottom"]
    if not any((left, right, top, bottom)):
        return src.copy()

    if fill == "neutral":
        canvas = Image.new("RGB", (p["width"], p["height"]), (128, 128, 128))
        canvas.paste(src, (left, top))
        return canvas

    canvas = Image.new("RGB", (p["width"], p["height"]))
    canvas.paste(src, (left, top))
    mirror = (fill == "mirror")

    # --- bandes horizontales ---
    def _side(box_w: int, from_left: bool):
        if mirror:
            band = src.crop((0, 0, min(box_w, w), h)) if from_left else \
                src.crop((max(0, w - box_w), 0, w, h))
            band = ImageOps.mirror(band)
            if band.width < box_w:          # marge plus large que l'image
                band = band.resize((box_w, h))
            return band
        # Étirement : une colonne de 1 px, dilatée. Constante en x -> aucune forme.
        edge = src.crop((0, 0, 1, h)) if from_left else src.crop((w - 1, 0, w, h))
        return edge.resize((box_w, h), Image.NEAREST)

    if left:
        canvas.paste(_side(left, True), (0, top))
    if right:
        canvas.paste(_side(right, False), (left + w, top))

    # --- bandes verticales : prélevées sur la toile DÉJÀ complétée en largeur,
    #     pour que les coins soient cohérents.
    mid = canvas.crop((0, top, p["width"], top + h))

    def _vside(box_h: int, from_top: bool):
        if mirror:
            band = mid.crop((0, 0, p["width"], min(box_h, h))) if from_top else \
                mid.crop((0, max(0, h - box_h), p["width"], h))
            band = ImageOps.flip(band)
            if band.height < box_h:
                band = band.resize((p["width"], box_h))
            return band
        edge = mid.crop((0, 0, p["width"], 1)) if from_top else \
            mid.crop((0, h - 1, p["width"], h))
        return edge.resize((p["width"], box_h), Image.NEAREST)

    if top:
        canvas.paste(_vside(top, True), (0, 0))
    if bottom:
        canvas.paste(_vside(bottom, False), (0, top + h))

    # Adoucit le remplissage (stries d'étirement, coutures du miroir) sans jamais
    # toucher l'original : on floute la toile entière, puis on recolle l'original
    # net par-dessus.
    canvas = canvas.filter(ImageFilter.GaussianBlur(12 if mirror else 24))
    canvas.paste(src, (left, top))
    return canvas


def build_mask(p: dict, feather: int = 24):
    """Masque d'inpainting : BLANC = à générer, NOIR = à conserver.

    (sd.cpp fabrique un masque tout blanc quand on n'en fournit pas — c'est-à-dire
    « repeins tout », l'img2img normal.) La bordure est adoucie pour que le modèle
    raccorde progressivement au lieu de buter sur une arête franche.
    """
    from PIL import Image, ImageFilter
    w, h = p["orig"]
    left, top = p["left"], p["top"]
    mask = Image.new("L", (p["width"], p["height"]), 255)     # tout à générer
    f = max(0, int(feather))
    # Zone conservée = l'original. Le dégradé doit mordre sur le NEUF, pas sur
    # l'original : on garde donc la boîte exacte et on floute vers l'extérieur.
    keep = (left, top, left + w, top + h)
    if keep[2] > keep[0] and keep[3] > keep[1]:
        mask.paste(0, keep)
        if f:
            mask = mask.filter(ImageFilter.GaussianBlur(f / 2.0))
            # Le flou a grignoté l'intérieur : on re-noircit le cœur pour que
            # l'original reste vraiment intouché.
            inner = (min(keep[2], left + f), min(keep[3], top + f),
                     max(keep[0], left + w - f), max(keep[1], top + h - f))
            if inner[2] > inner[0] and inner[3] > inner[1]:
                mask.paste(0, inner)
    return mask


def match_tone(generated, original, p: dict, amount: float = 1.0):
    """Recale la tonalité du résultat sur celle de l'original.

    En img2img le modèle re-rend TOUTE la toile, souvent plus contrastée et plus
    saturée. Si on recolle l'original tel quel, il apparaît alors comme un
    rectangle terne au milieu d'un décor punchy — la couture saute aux yeux même
    avec un fondu. On mesure donc, canal par canal, moyenne et écart-type sur la
    zone commune (là où les deux images montrent la MÊME chose), et on applique
    la correction affine correspondante à toute l'image générée.
    """
    from PIL import Image, ImageStat
    gen = generated.convert("RGB")
    src = original.convert("RGB")
    w, h = p["orig"]
    left, top = p["left"], p["top"]
    if gen.size != (p["width"], p["height"]):
        gen = gen.resize((p["width"], p["height"]))
    if amount <= 0:
        return gen
    box = (left, top, left + w, top + h)
    if box[2] <= box[0] or box[3] <= box[1]:
        return gen

    ref = ImageStat.Stat(src)
    cur = ImageStat.Stat(gen.crop(box))
    out_bands = []
    for i, band in enumerate(gen.split()):
        sd_c = cur.stddev[i] or 0.0
        sd_r = ref.stddev[i] or 0.0
        # Écart-type quasi nul (aplat uni) : on ne corrige que la moyenne.
        scale = (sd_r / sd_c) if sd_c > 1e-3 else 1.0
        scale = max(0.5, min(2.0, scale))               # garde-fou
        offset = ref.mean[i] - scale * cur.mean[i]
        # Dosage : 1.0 = correction complète, 0 = aucune.
        scale = 1.0 + (scale - 1.0) * amount
        offset *= amount
        out_bands.append(band.point(
            lambda v, s=scale, o=offset: max(0, min(255, int(round(v * s + o))))))
    return Image.merge("RGB", out_bands)


def composite_back(generated, original, p: dict, feather: int = 24):
    """Recolle l'original sur le résultat, avec un fondu sur `feather` pixels.

    Garantit que la zone d'origine est PRÉSERVÉE, quel que soit le modèle et
    sans masque côté moteur. Le fondu évite une couture visible.
    """
    from PIL import Image, ImageFilter
    out = generated.convert("RGB").copy()
    src = original.convert("RGB")
    w, h = p["orig"]
    left, top = p["left"], p["top"]
    if out.size != (p["width"], p["height"]):
        out = out.resize((p["width"], p["height"]))

    # Masque : blanc = on garde l'original. On rétrécit puis on floute pour
    # obtenir un dégradé UNIQUEMENT vers l'intérieur de la zone d'origine.
    mask = Image.new("L", (p["width"], p["height"]), 0)
    f = max(0, int(feather))
    inner = (left + f, top + f, left + w - f, top + h - f)
    if inner[2] > inner[0] and inner[3] > inner[1]:
        mask.paste(255, inner)
        if f:
            mask = mask.filter(ImageFilter.GaussianBlur(f / 2.0))
    else:                                  # image trop petite pour le fondu
        mask.paste(255, (left, top, left + w, top + h))
    out.paste(src, (left, top), mask.crop((left, top, left + w, top + h)))
    return out


def describe(p: dict) -> str:
    parts = [f"{k} +{p[k]}px" for k in DIRECTIONS if p.get(k)]
    return (f"{p['orig'][0]}×{p['orig'][1]} → {p['width']}×{p['height']}"
            + (" (" + ", ".join(parts) + ")" if parts else " (no extension)"))
