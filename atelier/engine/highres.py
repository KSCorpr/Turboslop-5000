"""« Haute résolution » — repasser une image dans Flux.2 à sa résolution native.

La méthode vient d'un fil r/StableDiffusion et tient en trois gestes, dont
aucun n'est évident :

1. **Demander « high resolution », pas « upscale ».** Dans les légendes
   d'entraînement, « upscaled » désigne des images RÉELLEMENT upscalées, donc
   porteuses des artefacts qu'on cherche justement à éviter. « High resolution »
   désigne des photos nettes d'origine. On ne demande pas la même distribution.

2. **Pré-agrandir l'image en BILINÉAIRE avant de la donner au modèle.** Flux.2
   travaille autour de 1024² : lui donner 600×400 le sort de son régime. Le
   bilinéaire est choisi pour sa fadeur — Lanczos ajoute du ringing que le
   modèle relira comme du détail et amplifiera.

3. **La même image en RÉFÉRENCE (-r) et en DÉPART (-i).** Les deux ne font pas
   le même travail : chez Flux.2 la référence passe par la VAE et porte le
   contenu, le latent de départ porte la structure. C'est le cœur de la
   méthode, et c'est précisément ce que l'interface ne savait pas faire — elle
   envoyait l'image en référence OU en img2img, jamais les deux.

CE QUE ÇA N'EST PAS : une restauration. À 0,8 de débruitage le modèle
REDESSINE l'essentiel de l'image ; ce qui est préservé, c'est la plausibilité,
pas la fidélité. Pour rester fidèle à l'original (une vraie photo, un visage
qu'on doit reconnaître), SeedVR2 reste l'outil. Ici on refabrique une image
convaincante à partir de la vôtre.

La dérive de couleur qui va avec est traitée à la fin, et pas au prompt :
« preserve exact color saturation » est un vœu, `color_match` est un calcul.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter

from .. import registry, settings
from . import generate as gen_engine
from . import sdcpp

# Régime natif de Flux.2. Le pré-agrandissement vise cette surface au minimum :
# en dessous, le modèle travaille hors de ce qu'il connaît.
FLUX_PIXELS = 1024 * 1024
# Plafond de surface pour la passe. Au-delà, Flux.2 perd la cohérence globale
# (répétitions, visages qui se dédoublent) et la VRAM explose : le tampon de
# calcul croît avec le nombre de pixels, EN PLUS des poids déjà sur la carte.
# Ce n'est pas une limite de l'implémentation, c'est celle du modèle.
MAX_PIXELS = 2560 * 1440
# Le prompt EST la méthode : ce mot-là, pas « upscale ».
DEFAULT_PROMPT = "high resolution image 1"
# Klein est distillé pour 4 pas. À 0,8 de débruitage il n'en resterait que 3
# d'effectifs, ce qui est trop court pour reconstruire du détail : on en donne
# assez pour que la passe ait de la matière à travailler.
STEPS = 8
# Réduction appliquée quand la carte refuse, et nombre de tentatives.
RETRY_FACTOR = 0.8
MAX_RETRIES = 3


def edit_models(prefs: dict | None = None) -> list[tuple[str, str]]:
    """Modèles utilisables ici : ceux qui savent prendre une RÉFÉRENCE.

    La méthode repose sur `-r`. Un modèle sans édition native ne ferait qu'un
    img2img ordinaire — autant ne pas le proposer et laisser croire.
    """
    prefs = prefs if prefs is not None else settings.load_prefs()
    out = []
    for model in registry.load_base_models(prefs):
        kind = model.defaults.get("edit")
        if kind in (True, "full") and registry.model_is_ready(model):
            out.append((model.name, model.id))
    return out


def plan_size(size: tuple[int, int], factor: float) -> tuple[int, int]:
    """Taille de sortie : le facteur demandé, borné par ce qui a du sens.

    Deux bornes, pour deux raisons différentes. En BAS, la surface native de
    Flux.2 : agrandir moins ne servirait à rien puisque le modèle repasserait
    quand même par là. En HAUT, la cohérence du modèle et la VRAM.
    """
    width, height = size
    ratio = width / height if height else 1.0
    pixels = max(1, width * height) * max(1.0, float(factor)) ** 2
    pixels = max(float(FLUX_PIXELS), min(float(MAX_PIXELS), pixels))
    # sd.cpp veut des multiples de 16 : on les impose ici plutôt que de laisser
    # le moteur arrondir dans notre dos et décaler le cadrage.
    new_h = (pixels / ratio) ** 0.5
    new_w = new_h * ratio
    return (max(256, int(round(new_w / 16)) * 16),
            max(256, int(round(new_h / 16)) * 16))


def enlarge(image: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Pré-agrandissement BILINÉAIRE — volontairement fade (cf. en-tête)."""
    return image.convert("RGB").resize(target, Image.BILINEAR)


def color_match(result: Image.Image, source: Image.Image,
                radius: float = 12.0) -> Image.Image:
    """Basses fréquences de l'ORIGINAL, hautes fréquences du RÉSULTAT.

    Le modèle rend souvent une image plus fade que la vôtre : à fort
    débruitage il impose son propre a priori colorimétrique. Corriger ça au
    prompt ne marche qu'à moitié. Ici on remet le rendu couleur de l'original
    (ses basses fréquences : teintes, exposition, dominantes) SOUS le détail
    que le modèle vient d'ajouter (ses hautes fréquences). Le détail est
    conservé, la couleur redevient la vôtre.
    """
    import numpy as np
    src = source.convert("RGB").resize(result.size, Image.LANCZOS)
    res = result.convert("RGB")
    high = (np.asarray(res, dtype=np.float32)
            - np.asarray(res.filter(ImageFilter.GaussianBlur(radius)),
                         dtype=np.float32))
    low = np.asarray(src.filter(ImageFilter.GaussianBlur(radius)),
                     dtype=np.float32)
    return Image.fromarray(np.clip(high + low, 0, 255).astype("uint8"))


def high_resolution(image, model_id: str, factor: float = 2.0,
                    strength: float = 0.8, prompt: str = "",
                    seed: int = -1, match_colors: bool = True,
                    log: Callable[[str], None] | None = None) -> Path:
    """Une passe « haute résolution » sur `image`. Renvoie le fichier produit."""
    if image is None:
        raise sdcpp.EngineError("Provide an image.")
    source = (Image.open(image) if isinstance(image, (str, Path))
              else image).convert("RGB")

    text = (prompt or "").strip() or DEFAULT_PROMPT
    settings.ensure_dirs()
    tmp = settings.TMP_DIR
    tmp.mkdir(parents=True, exist_ok=True)

    # La taille visée est calculée UNE fois, puis réduite directement en cas
    # de refus mémoire. Réduire le « facteur » ne marcherait pas : au-delà du
    # plafond, deux facteurs différents donnent la même taille et la reprise
    # tournerait en rond sur exactement la même demande.
    width, height = plan_size(source.size, float(factor))
    attempt = 0
    while True:
        prepared = enlarge(source, (width, height))
        # UN SEUL fichier, donné deux fois : c'est bien la même image qui sert
        # de référence et de point de départ.
        path = tmp / "highres_input.png"
        prepared.save(path)
        if log:
            log(f"High resolution: {source.width}×{source.height} → "
                f"{width}×{height} (×{width / max(1, source.width):.2f}), "
                f"denoise {strength:.2f}.")
        try:
            paths = gen_engine.generate(
                model_id=model_id, prompt=text, negative="",
                steps=STEPS, cfg_scale=1.0, width=width, height=height,
                seed=seed, batch_count=1,
                init_image=path, strength=float(strength),
                ref_image=path, save_prompt=False, log=log)
            break
        except sdcpp.VramError:
            attempt += 1
            new_w = max(256, int(round(width * RETRY_FACTOR / 16)) * 16)
            new_h = max(256, int(round(height * RETRY_FACTOR / 16)) * 16)
            # On s'arrête net si la reprise ne réduit rien, ou si elle
            # descendrait sous l'image de départ : agrandir moins que rien
            # n'est pas un repli, c'est une perte de temps déguisée.
            if (attempt > MAX_RETRIES or (new_w, new_h) == (width, height)
                    or new_w * new_h <= source.width * source.height):
                raise
            width, height = new_w, new_h
            if log:
                log(f"↻ Not enough memory — retrying at "
                    f"{width}×{height}.")
    if not paths:
        raise sdcpp.EngineError("No image produced (see the log).")

    out = paths[0]
    if match_colors:
        # Le fichier est FERMÉ avant d'être réécrit : sous Windows, un fichier
        # encore ouvert ne peut pas être remplacé.
        with Image.open(out) as rendered:
            fixed = color_match(rendered, source)
        fixed.save(out)
        if log:
            log("Original colours reapplied (the detail is kept).")
    return out
