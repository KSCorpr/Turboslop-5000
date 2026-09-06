#!/usr/bin/env python3
"""Runner « image → prompt » : un VLM regarde l'image et écrit un PROMPT.

La différence avec un sous-titrage classique est le cœur du sujet. Un modèle de
vision produit spontanément « une photo d'un chat sur un canapé » : c'est une
LÉGENDE — une phrase sur l'image. Un prompt, lui, est une CONSIGNE : il nomme le
médium, la lumière, l'objectif, la palette, le cadrage, parce que ce sont ces
mots-là qui pilotent un modèle de diffusion. Coller une légende dans le champ
Prompt donne une image vaguement ressemblante et plate.

Le system prompt ci-dessous fait donc trois choses : il interdit les formules de
légende, il impose de nommer le médium détecté (et de n'employer QUE son
vocabulaire — mêler « oil painting » et « 85mm lens » brouille le rendu), et il
force l'anglais quelle que soit la langue de l'interface, parce que c'est la
langue d'entraînement des modèles.

Lancé en sous-process, comme l'améliorateur : le modèle est chargé puis
déchargé, donc aucune VRAM n'est retenue pendant la génération sd.cpp.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import label, pick_device, pick_dtype  # noqa: E402

# --------------------------------------------------------------------------- #
#  Règles communes aux trois modes
# --------------------------------------------------------------------------- #
_COMMON = (
    "You write prompts for text-to-image diffusion models. You are shown ONE "
    "image and you answer with the prompt itself — nothing else.\n"
    "\n"
    "HARD RULES:\n"
    "• Answer in ENGLISH, whatever language the request is in. Diffusion models "
    "are trained on English.\n"
    "• NEVER start with caption formulas: no 'This image shows', 'A photo of', "
    "'The picture depicts', 'In this image'. Start directly with the content.\n"
    "• No preamble, no explanation, no quotes, no bullet list, no markdown. One "
    "single flowing block of comma-separated phrases.\n"
    "• Describe ONLY what is visible. Never invent a brand, a place, a name or "
    "a date you cannot actually see. This applies to COLOURS above all: name "
    "the colour that is actually there. If a shoe is black, it is black — not "
    "'dark red' because that would sound better.\n"
    "• NEVER repeat yourself. Each comma-separated fragment must add something "
    "the previous ones did not. Once you have said 'wet', 'high heels' or "
    "'fashion', that idea is spent — do not restate it in another form. When "
    "you have nothing left to add, STOP; a short prompt beats a padded one.\n"
    "• Decide the MEDIUM once, say it FIRST, and never contradict it. A "
    "photograph gets lens, aperture, depth of field, film grain, real texture "
    "— and NEVER 'oil painting', 'brushstrokes', 'canvas texture', 'painterly' "
    "or 'render'. A painting gets brushwork, canvas and pigment, and never a "
    "lens or an aperture. Mixing the two is the single most common way to ruin "
    "a prompt, and it happens at the END, when there is nothing real left to "
    "say. If you reach that point, stop writing instead.\n"
)

_MODE_FULL = _COMMON + (
    "\n"
    "TASK — write a prompt that would REPRODUCE this image on a model that has "
    "never seen it. Cover, in this order and only when relevant:\n"
    "1. the MEDIUM, first and in two or three words (photograph, oil painting, "
    "anime cel, 3D render, pencil sketch…). Deciding it first keeps the rest "
    "coherent;\n"
    "2. the main subject, precisely (species, age, build, pose, expression, "
    "clothing, materials);\n"
    "3. what surrounds it: setting, background, secondary objects;\n"
    "4. the LIGHT — direction, hardness, colour temperature, time of day;\n"
    "5. the COMPOSITION and framing: shot type, angle, depth of field;\n"
    "6. the COLOURS actually present, named precisely (crimson, muted sage, "
    "warm ochre — not 'nice colours', and not a colour you wish were there).\n"
    "Aim for 60 to 110 words. Dense, concrete, no filler adjectives."
)

_MODE_STYLE = _COMMON + (
    "\n"
    "TASK — extract ONLY THE STYLE, so it can be applied to a completely "
    "different subject.\n"
    "• Say NOTHING about what the picture is of. No subject, no object, no "
    "character, no place. If the image shows a red car in Rome, the words "
    "'car' and 'Rome' must NOT appear.\n"
    "• Describe only: the medium and technique, the light, the colour palette, "
    "the contrast and grain, the level of detail, the framing habits, the era "
    "or movement it evokes.\n"
    "• Write it so that it can be pasted in front of any subject.\n"
    "Aim for 25 to 50 words."
)

_MODE_PLAIN = _COMMON.replace(
    "You write prompts for text-to-image diffusion models. You are shown ONE "
    "image and you answer with the prompt itself — nothing else.",
    "You describe images in plain, factual English.") + (
    "\n"
    "TASK — say what is in the image, plainly, for someone who cannot see it. "
    "Two or three sentences. No prompt vocabulary, no camera settings, no "
    "style jargon — just what is there."
)

MODES = {"full": _MODE_FULL, "style": _MODE_STYLE, "plain": _MODE_PLAIN}

# Le modèle sait lire de très grandes images, mais chaque pixel devient des
# jetons visuels : au-delà de ~1 Mpx on paie beaucoup de temps pour un détail
# qui ne changera pas le prompt. On borne donc le côté long.
MAX_SIDE = 1024

# Amorces de légende que le modèle ressort malgré la consigne. On les coupe
# après coup plutôt que d'espérer : la consigne réduit la fréquence, elle ne la
# met pas à zéro, et une seule occurrence suffit à polluer le champ Prompt.
_LEAD_INS = (
    "this image shows", "this image depicts", "this image features",
    "the image shows", "the image depicts", "the image features",
    "the picture shows", "the picture depicts", "this picture shows",
    "here is a prompt", "here's a prompt", "prompt:", "sure,", "certainly,",
    "in this image,", "the photo shows", "this photograph shows",
)


# Mots vides d'un fragment : « wet pavement » et « the wet pavement » sont le
# même segment pour nous, et un prompt n'a aucun besoin des deux.
_FILLER = {"a", "an", "the", "of", "with", "and", "in", "on", "at", "to"}


def _key(fragment: str) -> str:
    """Forme canonique d'un fragment, pour comparer deux segments."""
    words = [w for w in re.findall(r"[a-z0-9+]+", fragment.lower())
             if w not in _FILLER]
    return " ".join(words)


def _dedupe(text: str) -> str:
    """Retire les segments répétés d'une liste séparée par des virgules.

    LE bug observé en vrai : après un bon début, le modèle s'enferme dans une
    boucle et répète « high heels, fashion, modern, wet, rain » jusqu'à épuiser
    son budget de jetons. Les pénalités de répétition passées au générateur
    réduisent le phénomène ; elles ne le suppriment pas, parce qu'une liste de
    mots-clés est précisément le format où boucler est le plus tentant.

    Ici on ne parie pas : la sortie EST une liste séparée par des virgules, donc
    on peut la dédupliquer sans rien perdre. L'ordre est conservé — le début est
    la partie utile — et un fragment déjà dit, sous quelque forme que ce soit,
    ne repasse pas.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for raw in text.split(","):
        frag = raw.strip()
        key = _key(frag)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(frag)
    return ", ".join(kept)


# --------------------------------------------------------------------------- #
#  Cohérence de MÉDIUM
# --------------------------------------------------------------------------- #
# Deuxième panne observée en vrai, et plus grave que la boucle : sur une PHOTO,
# le modèle a terminé par « oil painting style, brushstrokes visible, canvas
# texture evident ». Trois fragments qui contredisent tout ce qui précède, et
# qui suffisent à faire produire une peinture au modèle de diffusion.
#
# La consigne l'interdit déjà, en majuscules. Ça ne suffit pas — et ça ne
# suffira jamais : la dérive arrive en FIN de génération, quand le modèle n'a
# plus rien de réel à dire. On vérifie donc après coup, comme pour les
# répétitions.
#
# Marqueurs choisis pour être SANS AMBIGUÏTÉ : « texture » seul ne dit rien,
# « canvas texture » désigne une toile. Un terme qui pourrait appartenir à deux
# médiums n'a rien à faire ici — le prix d'une erreur est de supprimer un
# fragment légitime.
_MEDIA = {
    "photo": ("photograph", "photography", "photorealistic", "dslr",
              "depth of field", "bokeh", "film grain", "shot on", "camera",
              "lens", "aperture", "shutter", "iso ", "long exposure",
              "macro shot", "telephoto", "wide-angle"),
    "painting": ("oil painting", "acrylic", "watercolor", "watercolour",
                 "gouache", "brushstroke", "brush stroke", "brushwork",
                 "impasto", "canvas texture", "painterly", "palette knife",
                 "on canvas"),
    "3d": ("3d render", "3d rendering", "octane", "blender render", "cgi",
           "ray-traced", "raytraced", "clay render", "subsurface scattering"),
    "anime": ("anime", "manga", "cel shading", "cel-shaded", "cel shaded"),
    "drawing": ("pencil sketch", "line art", "ink drawing", "charcoal",
                "vector art", "flat design", "linocut", "woodcut"),
}


def _medium_of(fragment: str) -> str | None:
    low = fragment.lower()
    for medium, markers in _MEDIA.items():
        if any(m in low for m in markers):
            return medium
    return None


def _one_medium(text: str) -> str:
    """Ne garde que le médium ÉTABLI EN PREMIER, et jette les contradictions.

    Le premier marqueur gagne, et ce n'est pas arbitraire : le modèle décrit
    d'abord ce qu'il voit réellement, la confabulation vient après. Dans le cas
    observé, « depth of field » (photo) arrive neuf fragments avant « oil
    painting style » — l'ordre porte l'information.

    S'il n'y a qu'un seul médium, ou aucun, rien n'est touché.
    """
    frags = [f.strip() for f in text.split(",")]
    first = next((m for m in map(_medium_of, frags) if m), None)
    if first is None:
        return text
    kept = [f for f in frags
            if f and _medium_of(f) in (None, first)]
    return ", ".join(kept)


def _drop_dangling(text: str) -> str:
    """Coupe le dernier fragment d'une génération ARRÊTÉE par la limite.

    On ne devine pas si la fin est tronquée — « shallow dep » et « shallow »
    sont indiscernables sans dictionnaire, et couper un fragment légitime est
    pire que laisser un moignon. L'appelant SAIT : si le modèle n'a pas émis
    son jeton de fin, c'est la limite de jetons qui l'a arrêté, donc le dernier
    segment est coupé au milieu. Cette fonction n'est appelée que dans ce cas.
    """
    head, sep, _tail = text.rpartition(",")
    return (head if sep else text).rstrip(" ,;-")


def _clean(text: str, truncated: bool = False) -> str:
    """Retire les amorces de légende, les guillemets, puis les répétitions."""
    out = (text or "").strip()
    # Certains modèles encadrent leur réponse de guillemets ou de ```.
    if out.startswith("```"):
        out = out.split("\n", 1)[-1]
        out = out.rsplit("```", 1)[0]
    out = out.strip().strip('"').strip("'").strip()
    changed = True
    while changed:
        changed = False
        low = out.lower()
        for lead in _LEAD_INS:
            if low.startswith(lead):
                out = out[len(lead):].lstrip(" :,-—").lstrip()
                changed = True
                break
    # La déduplication ne s'applique qu'aux LISTES. Le mode « décrire
    # simplement » rend des phrases, où une virgule ne sépare pas des segments
    # interchangeables — y couper des morceaux casserait la grammaire.
    if out.count(",") >= 4 and out.count(".") <= 1:
        out = _one_medium(_dedupe(out))
        if truncated:
            out = _drop_dangling(out)
    # Une majuscule initiale perdue en coupant l'amorce se rattrape.
    return (out[:1].upper() + out[1:]) if out else out


def _load_image(path: str):
    from PIL import Image
    img = Image.open(path)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="full", choices=sorted(MODES))
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = pick_device(torch)
    dtype = pick_dtype(torch, device)
    print(f"[image→prompt] chargement du modèle sur {label(device)}…",
          flush=True)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=dtype).to(device).eval()

    image = _load_image(args.image)
    print(f"[image→prompt] image lue : {image.size[0]}×{image.size[1]}",
          flush=True)

    system = MODES[args.mode]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Write it now. Output the text only."},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    n = max(1, min(4, int(args.variants or 1)))
    # Budget CADRÉ sur la longueur demandée (~1,7 jeton par mot, virgules
    # comprises). L'ancien budget de 320 jetons pour 110 mots laissait 65 % de
    # marge, et cette marge, le modèle la remplit : c'est là qu'il se met à
    # répéter « high heels, fashion, modern, wet » jusqu'à la fin.
    budget = {"full": 200, "style": 110, "plain": 150}[args.mode]
    max_new = int(args.max_new_tokens) or budget
    print(f"[image→prompt] rédaction ({args.mode}, {n} proposition(s))…",
          flush=True)
    # DÉCRIRE N'EST PAS CRÉER. Le tirage aléatoire, c'est demander au modèle de
    # choisir parfois un jeton MOINS probable — donc, sur une description,
    # d'inventer. Observé en vrai : des escarpins noirs annoncés « dark red
    # shoes ». Sur une seule proposition on prend donc le jeton le plus probable
    # à chaque pas, sans tirage. Il n'y a rien à gagner à varier : l'image, elle,
    # ne varie pas.
    #
    # Le tirage ne revient que si l'on demande PLUSIEURS propositions, où
    # différer est justement l'objet — et encore, à température basse.
    sampling = ({"do_sample": False} if n == 1 else
                {"do_sample": True, "temperature": 0.3, "top_p": 0.85})
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            # Une liste de mots-clés est le format où boucler est le plus
            # tentant : rien ne signale au modèle qu'il a fini. Deux garde-fous
            # complémentaires — la pénalité décourage de réemployer un jeton
            # déjà sorti, le n-gramme interdit carrément de redire une suite de
            # six jetons. Le motif observé en vrai (« high heels, fashion,
            # modern, wet, rain ») en fait une dizaine : il est couvert.
            repetition_penalty=1.1, no_repeat_ngram_size=6,
            num_return_sequences=n, **sampling)
    start = inputs["input_ids"].shape[1]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    results, seen = [], set()
    for row in out:
        body = row[start:]
        # Pas de jeton de fin = c'est max_new_tokens qui a arrêté le modèle,
        # donc le dernier fragment est coupé au milieu d'un mot.
        cut = len(body) >= max_new and int(body[-1]) not in eos_ids
        cand = _clean(processor.tokenizer.decode(body,
                                                 skip_special_tokens=True),
                      truncated=cut)
        key = cand.lower()
        if cand and key not in seen:
            seen.add(key)
            results.append(cand)
    if not results:
        sys.exit("[image→prompt] le modèle n'a produit aucun texte.")

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"[image→prompt] {len(results)} proposition(s) écrite(s) : {dest}",
          flush=True)


if __name__ == "__main__":
    main()
