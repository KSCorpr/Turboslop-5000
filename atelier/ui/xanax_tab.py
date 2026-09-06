"""Onglets « Xanax » : le style est CODÉ EN DUR, il n'y a rien à régler.

Transcription du system prompt Gemini de l'utilisateur, réduite à l'essentiel :
une phrase en entrée, une image en sortie. Contrairement aux onglets de
génération classiques, il n'y a ici ni menu de préréglages, ni banques de
styles, ni prompt système modifiable — le style est « fixe et non négociable »,
donc il n'est pas exposé.

Un point mérite d'être explicité, parce qu'il surprend : **la traduction FR→EN
a besoin de l'améliorateur**. Le style est un préfixe anglais collé devant le
texte ; à lui seul il ne traduit rien. Si l'améliorateur est installé on s'en
sert (avec le style transmis en contrainte, pour qu'il n'écrive rien qui le
contredise) ; sinon on le dit clairement au lieu de laisser croire que ça marche.
"""
from __future__ import annotations

import queue
import random
import re
import threading

import gradio as gr

from .. import registry, settings
from ..engine import generate as gen_engine
from ..engine import tools
from ..i18n import t
from . import widgets

# --------------------------------------------------------------------------- #
#  LE STYLE — fixe, non négociable, non exposé dans l'interface.
# --------------------------------------------------------------------------- #
XANAX_STYLE = (
    "amateur snapshot photograph, everyday life in provincial France between "
    "1995 and 2005, shot on a cheap low-end consumer point-and-shoot camera, "
    "naturalistic mundane unadorned photography, plain uncomposed careless "
    "framing, ordinary unremarkable people going about their day, raw "
    "unstaged daily life, flat grey overcast daylight, dull mundane "
    "surroundings, clean untouched image straight out of the camera, no grain, "
    "no filter, no post-processing"
)

# Format 4:3 imposé par le brief, sur la grille NATIVE de chaque modèle (32 px
# pour Flux.2, 64 px pour Krea 2 — sortir de la grille dégrade le rendu).
XANAX_SIZE = {"flux2": (1184, 880), "krea2": (1152, 896)}

# --------------------------------------------------------------------------- #
#  DE QUOI ON PART : une phrase de la vie courante, pas une description
# --------------------------------------------------------------------------- #
# Décrire une image (« un homme qui attend le bus devant un supermarché ») et
# raconter sa journée (« j'ai attendu le bus une plombe ») ne donnent pas le
# même résultat, et ce n'est pas une question de formulation. Une description
# est déjà cadrée : elle dit quoi montrer, donc le modèle met le sujet au
# milieu et compose. Une phrase de journal intime ne dit PAS ce qu'il faut
# montrer — il faut aller chercher le lieu, l'heure, les gens autour ; ce qui
# en sort ressemble à une photo prise en passant, ce que vise cet onglet.
#
# C'est pourquoi l'améliorateur reçoit ici son propre system prompt (style
# « xanax ») : les deux autres réclament un éclairage travaillé, un objectif
# nommé et une composition — la photo RÉUSSIE d'un photographe, alors qu'on
# veut la photo RATÉE d'un oncle.
ANECDOTES = [
    "had lunch at the motorway cafeteria with Gran",
    "rubbish day but at least I got my cigarettes",
    "did the shopping at the hypermarket, long queue at the till",
    "Grandad's birthday, we were all crammed in the conservatory",
    "waited twenty minutes for the bus in the rain",
    "New Year's Eve at my aunt's, we had the yule log",
    "washed the car in the driveway",
    "went to the school fête with my son",
    "hung around the benefits office all morning",
    "barbecue at the neighbours', then it started raining",
    "repainted the bedroom, not finished yet",
    "ate at the truck stop on the main road",
    "took the dog to the vet",
    "my cousin's communion, photo outside the church",
    "had a coffee at the corner café after the market",
    "moved my sister's sofa",
    "car boot sale on Sunday morning, sold nothing",
    "watched the match round at Kev's",
    "spent the afternoon at the launderette",
    "went to see the sea, it was grey",
    "assembled the flat-pack kitchen unit",
    "leaving drinks at work in the break room",
    "queued at the post office for a parcel",
    "had drinks in the garden, nothing special",
    "won three euros at the betting shop",
    "had pizza in front of the telly",
    "waited for my daughter outside the school gates",
    "stopped at the motorway services",
    "fixed the bike in the garage",
    "colleague's wedding, in the village hall",
    "mowed the lawn before the rain",
    "celebrated at the kebab shop downstairs",
]

# Barres de progression de sd.cpp : converties en ligne de statut, jamais
# écrites dans le journal (elles arrivent par centaines et le noient).
_PROGRESS_BAR = re.compile(r"\|[#=>\-\s]*\|")


def _size_for(family: str) -> tuple[int, int]:
    # startswith et non == : toute variante Krea partage la grille native du
    # Turbo. Un repli silencieux sur celle de Flux.2 ne se verrait qu'à
    # l'image produite, donc autant ne pas laisser le cas ouvert.
    if family.startswith("krea2"):
        return XANAX_SIZE["krea2"]
    return XANAX_SIZE.get(family, (1184, 880))


def build_prompt(subject: str) -> str:
    """Le prompt réellement envoyé : le style figé, puis le sujet."""
    return f"{XANAX_STYLE}, {(subject or '').strip().strip(',')}".strip(", ")


# Modèles proposés dans l'onglet, du plus rapide au plus lourd. Un SEUL onglet
# pour les deux : le style est identique, seul le moteur change — deux onglets
# jumeaux, c'était deux fois le même écran à maintenir et une case de plus à
# lire dans la barre.
XANAX_MODELS = [("⚡ Krea 2 Turbo", "krea2-turbo"),
                ("🟣 Flux.2 Klein 9B", "flux2-klein-9b")]


def _model_info(model_id: str):
    """(modèle, famille, défauts, largeur, hauteur, pas) pour un id donné."""
    model = registry.get_base_model(model_id, settings.load_prefs())
    family = model.family if model else "flux2"
    d = dict(model.defaults) if model else {}
    w, h = _size_for(family)
    return model, family, d, w, h, int(d.get("steps", 8) or 8)


def _recap_for(model_id: str) -> str:
    """État du modèle choisi + format qu'il produira, avant le clic."""
    model, _fam, _d, w, h, _st = _model_info(model_id)
    ready = model is not None and registry.model_is_ready(model)
    state = (t("● model ready") if ready
             else t("○ to download (Model Catalog tab)"))
    return f"{state} · {w}×{h}"


def build_xanax_tab(title: str = "💊 Xanax"):
    with gr.Tab(title):
        # Texte SÉPARÉ du titre : une f-string composée ne peut pas servir de
        # clé de traduction (elle ne correspondrait jamais au dictionnaire).
        gr.Markdown(
            "### Tell us about your day, we make a photo of it\nDo **not** "
            "write an image description — write a plain sentence about your "
            "day, the way you would in a diary: *“had lunch at the cafeteria "
            "with Gran”*, *“rubbish day but at least I got my cigarettes”*. "
            "That gap is what produces a photo taken in passing rather than a "
            "posed one.\n\n**The style is fixed and cannot be changed**: "
            "amateur snapshot, provincial France, 1995-2005, overcast, no "
            "post-processing, 4:3 on the model's native grid. That is the "
            "point of this tab — to tune anything at all, use a normal "
            "generation tab.")

        with gr.Row():
            with gr.Column(scale=3):
                model_pick = gr.Radio(
                    choices=[(t(lbl), mid) for lbl, mid in XANAX_MODELS],
                    value=XANAX_MODELS[0][1], label="Model")
                model_state = gr.Markdown(_recap_for(XANAX_MODELS[0][1]),
                                          elem_classes="hint")
                prompt = gr.Textbox(
                    label="What you did", lines=3,
                    placeholder="had lunch at the motorway cafeteria with Gran…",
                    info="One sentence about your day, in the first person. "
                         "Not “a man waits for the bus” but “waited ages for "
                         "the bus”.")
                dice = gr.Button("🎲 A random day", size="sm")
                with gr.Row(elem_classes="go-row"):
                    run = gr.Button("📷 Generate", variant="primary",
                                    size="lg", scale=4)
                    stop = gr.Button("⏹️ Stop", variant="stop", scale=1,
                                     min_width=90)
                status = gr.Markdown("")
                enhance = gr.Checkbox(
                    value=tools.enhance_is_installed(),
                    interactive=tools.enhance_is_installed(),
                    label="✨ Turn my sentence into a photo (AI enhancer)",
                    info=("Works out what the photo would SHOW: the place, "
                          "the people, the time of day. Translates along the "
                          "way, and knows what a French cafeteria chain is — "
                          "the image model does not."
                          if tools.enhance_is_installed() else
                          "Enhancer not installed — install it from a "
                          "generation tab. Without it your sentence is sent "
                          "AS IS: write in English then, and say what is "
                          "visible rather than what you did."))
                seed = gr.Number(value=-1, precision=0,
                                 label="Seed (-1 = random)",
                                 info="A fixed seed replays exactly the same "
                                      "photo.")
            with gr.Column(scale=4):
                result = gr.Image(label="Photo", type="filepath", height=460,
                                  format="png", buttons=widgets.IMAGE_BUTTONS)
                used_md = gr.Markdown("", elem_classes="hint")
                log = gr.Textbox(label="Log", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        model_pick.change(_recap_for, inputs=[model_pick],
                          outputs=[model_state])
        # Le dé ne sert pas qu'à dépanner l'inspiration : il montre le REGISTRE
        # attendu. Un exemple qu'on peut lire, modifier et relancer explique
        # mieux qu'un paragraphe ce que veut dire « pas une description ».
        dice.click(lambda: random.choice([t(a) for a in ANECDOTES]),
                   outputs=[prompt])

        def do_xanax(model_id, subject, use_enhance, seed_v):
            if not (subject or "").strip():
                raise gr.Error(t("Tell us something first — one sentence is "
                                 "enough."))
            settings.ensure_dirs()
            model, _family, d, width, height, steps = _model_info(model_id)
            if model is None:
                raise gr.Error(t("Model unavailable."))
            try:
                base_seed = int(seed_v)
            except (TypeError, ValueError):
                base_seed = -1
            if base_seed < 0:
                base_seed = random.randint(0, 2**31 - 1)

            logs: list[str] = []
            text = subject.strip()

            if use_enhance and tools.enhance_is_installed():
                yield (t("⏳ Working out what that moment looked like…"),
                       gr.update(), gr.update(), "\n".join(logs))
                try:
                    # style « xanax » : le seul des trois qui parte d'une
                    # phrase de journal au lieu d'une description, et le seul
                    # qui n'ajoute PAS d'objectif, d'éclairage ni de
                    # composition — ici, une belle photo serait ratée.
                    out = tools.enhance_prompt_variants(
                        text, style="xanax", level="medium", variants=1,
                        style_constraint=XANAX_STYLE, log=logs.append)
                    if out and out[0].strip():
                        text = out[0].strip()
                except Exception as exc:  # noqa: BLE001
                    # Un échec de l'améliorateur ne doit pas empêcher de générer.
                    logs.append(f"[improver unavailable] {exc}")

            full_prompt = build_prompt(text)
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    outs = gen_engine.generate(
                        model_id=model_id, prompt=full_prompt, negative="",
                        steps=steps,
                        cfg_scale=float(d.get("cfg_scale", 1.0) or 1.0),
                        width=width, height=height, seed=base_seed,
                        batch_count=1, sampler=d.get("sampler") or "euler",
                        schedule=("" if d.get("scheduler") in (None, "", "auto")
                                  else d["scheduler"]),
                        log=q.put)
                    state["outs"] = [str(p) for p in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            step_re = re.compile(rf"(\d+)\s*/\s*{steps}\b")
            logs.append(f"Seed : {base_seed}")
            logs.append(f"Prompt : {full_prompt}")
            yield (t("⏳ Loading the model…"), gr.update(), gr.update(),
                   "\n".join(logs))
            while True:
                line = q.get()
                if line is None:
                    break
                mt = step_re.search(line)
                if mt:
                    yield (t("🎨 Step {cur}/{total}").format(
                        cur=min(int(mt.group(1)), steps), total=steps),
                        gr.update(), gr.update(), "\n".join(logs[-400:]))
                    continue
                if _PROGRESS_BAR.search(line) or "\x1b" in line:
                    continue
                logs.append(line)
                yield (gr.update(), gr.update(), gr.update(),
                       "\n".join(logs[-400:]))

            if "err" in state or not state.get("outs"):
                logs.append(f"\n[ERROR] {state.get('err', 'no image')}")
                yield (t("❌ Failed — see the log."), gr.update(),
                       gr.update(), "\n".join(logs[-400:]))
                return

            # Le prompt accompagne l'image, comme demandé par la consigne
            # « écris le prompt après chaque image ».
            yield (t("✅ Photo generated (seed {s})").format(s=base_seed),
                   gr.update(value=state["outs"][0]),
                   gr.update(value=f"**Prompt used:** {full_prompt}"),
                   "\n".join(logs[-400:]))

        evt = run.click(do_xanax, inputs=[model_pick, prompt, enhance, seed],
                        outputs=[status, result, used_md, log])
        # QOL : Ctrl+Entrée depuis le champ de saisie lance la génération.
        prompt.submit(do_xanax, inputs=[model_pick, prompt, enhance, seed],
                      outputs=[status, result, used_md, log])
        widgets.stop_into_status(stop, gen_engine.cancel, status, [evt])
