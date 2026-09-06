"""Onglet de génération (un par modèle) : text-to-image / image-to-image,
presets sampler/scheduler, LoRA, fichiers locaux, envoi vers l'upscale.
"""
from __future__ import annotations

import random
import re

import gradio as gr

from .. import downloader, i18n, registry, sampling, settings, styles
from ..engine import generate as gen_engine
from ..engine import tools
from ..i18n import t
from . import widgets

# Les listes de samplers/schedulers ET leur documentation vivent dans
# atelier/sampling.py : un menu dont on ne sait pas quoi choisir n'est pas un
# menu, et la réponse dépend du MODÈLE (distillé, CFG 1.0, peu de pas).

# Préréglages de résolution PAR FAMILLE de modèle, alignés sur les résolutions
# natives d'entraînement (le modèle rend mieux sur ces formats).
#
#  • Flux.2 Klein : entraîné en ~1 MP, grille de 32 px, gère jusqu'à ~4 MP.
#    Formats natifs documentés : 1024², 1248×832, 1184×880, 1392×752, 1568×672…
#  • Krea 2 : famille SDXL/1024, multiples de 64 px : 1024², 1152×896, 1216×832,
#    1344×768, etc. (+ option 2K via VAE WAN).
RATIOS_FLUX2: dict[str, tuple[int, int]] = {
    "Square 1:1 — 1024×1024": (1024, 1024),
    "Square 1:1 — 1440×1440 (2K)": (1440, 1440),
    "Landscape 3:2 — 1248×832": (1248, 832),
    "Portrait 2:3 — 832×1248": (832, 1248),
    "Landscape 4:3 — 1184×880": (1184, 880),
    "Portrait 3:4 — 880×1184": (880, 1184),
    "Wide 16:9 — 1392×752": (1392, 752),
    "Vertical 9:16 — 752×1392": (752, 1392),
    "Cinema 21:9 — 1568×672": (1568, 672),
    "Custom (sliders)": (0, 0),
}
RATIOS_KREA2: dict[str, tuple[int, int]] = {
    "Square 1:1 — 1024×1024": (1024, 1024),
    "Square 1:1 — 1536×1536 (2K)": (1536, 1536),
    "Landscape 3:2 — 1216×832": (1216, 832),
    "Portrait 2:3 — 832×1216": (832, 1216),
    "Landscape 4:3 — 1152×896": (1152, 896),
    "Portrait 3:4 — 896×1152": (896, 1152),
    "Wide 16:9 — 1344×768": (1344, 768),
    "Vertical 9:16 — 768×1344": (768, 1344),
    "Custom (sliders)": (0, 0),
}
_CUSTOM_LABEL = "Custom (sliders)"

# Entrée « neutre » en tête du menu des styles perso : la sélectionner RETIRE
# le style appliqué (vide le champ système), sans rien supprimer d'enregistré.
_NONE_STYLE = "— None —"


def _style_choices() -> list[str]:
    return [_NONE_STYLE] + styles.list_styles()

# Barres de progression sd.cpp (« |####| 97/298 - 87MB/s[K », « |==>| 1/4 - … »).
# Capturées ligne par ligne, elles arrivent par CENTAINES et inondent le journal
# → re-rendu permanent = clignotement. On les DÉTECTE pour piloter la barre de
# progression, mais on ne les écrit PAS dans le texte du journal.
_PROGRESS_BAR = re.compile(r"\|[#=>\-\s]*\|")


def _ratios_for(family: str) -> dict[str, tuple[int, int]]:
    # startswith et non == : toute variante Krea (INT8 ConvRot aujourd'hui,
    # une autre demain) partage l'architecture du Turbo, donc ses résolutions
    # natives. Une égalité stricte lui donnerait la grille de Flux.2, hors de
    # sa grille d'entraînement — et ça ne se verrait qu'à l'image produite.
    return RATIOS_KREA2 if family.startswith("krea2") else RATIOS_FLUX2


def _defaults(model_id: str) -> dict:
    m = registry.get_base_model(model_id, settings.load_prefs())
    return m.defaults if m else {}


def _presets(model_id: str) -> list[dict]:
    m = registry.get_base_model(model_id, settings.load_prefs())
    return list(m.presets) if (m and m.presets) else []


def _ratio_label(ratios: dict[str, tuple[int, int]], w: int, h: int) -> str:
    for label, (rw, rh) in ratios.items():
        if rw == w and rh == h:
            return label
    return _CUSTOM_LABEL


def build_generative_tab(model_id: str, title: str,
                         # Les trois destinations vivent désormais SOUS le même
                         # onglet racine « 🧰 Outils » : c'est lui qu'on
                         # sélectionne ici, chaque onglet enfant se chargeant
                         # ensuite de se mettre au premier plan.
                         pending_toolkit=None, tabs=None, toolkit_tab_id="tools",
                         pending_3d=None, threed_tab_id="tools",
                         pending_outpaint=None, outpaint_tab_id="tools"):
    d = _defaults(model_id)

    # `id` EXPLICITE : « 📝 Image → prompt » renvoie son texte vers un onglet
    # nommé. Sans identifiant stable, la cible serait le libellé affiché, qui
    # change avec la langue — le retour tomberait à côté en anglais.
    with gr.Tab(title, id=model_id):
        m = registry.get_base_model(model_id, settings.load_prefs())
        ready = m is not None and registry.model_is_ready(m)
        family = m.family if m else "flux2"
        ratios = _ratios_for(family)
        # Capacité d'édition, déclarée par le catalogue (defaults.edit) :
        #   "full"     -> modèle d'édition natif (Flux.2) : UI images de référence ;
        #   "optional" -> édition possible AVEC un LoRA d'édition (Krea 2 Ostris
        #                 Edit) : UI img2img + case « Mode édition » ;
        #   absent     -> img2img classique seulement.
        _edit_kind = (m.defaults.get("edit") if m else None) or \
            ("full" if (m and m.family == "flux2") else "")
        is_edit = _edit_kind in (True, "full")
        edit_optional = _edit_kind == "optional"
        status = (f"<span class='status-ok'>{t('● model ready')}</span>" if ready
                  else "<span class='status-missing'>"
                       f"{t('○ to download (Model catalog tab)')}</span>")
        _mode = t("image editing") if is_edit else t("image-to-image")
        gr.Markdown(t("### {title} — text-to-image & {mode}  ·  {status}").format(
            title=title, mode=_mode, status=status))

        # Variante de moteur, pas un nouvel onglet : le workflow et tous les
        # réglages Krea restent identiques. GGUF demeure le défaut tant que le
        # test A/B local n'a pas démontré que l'INT8 vaut mieux sur cette carte.
        #
        # Le sélecteur n'apparaît QUE si la variante est réellement installée.
        # Proposer un choix dont une branche n'existe pas, c'est un piège :
        # l'utilisateur clique, la génération échoue. Et sur l'onglet le plus
        # utilisé de l'application, un réglage expérimental n'a rien à faire
        # au-dessus du prompt tant qu'il ne concerne personne.
        _variant = registry.get_base_model("krea2-turbo-int8",
                                           settings.load_prefs())
        _variant_ready = (model_id == "krea2-turbo" and _variant is not None
                          and registry.model_is_ready(_variant))
        if _variant_ready:
            variant_model = gr.Radio(
                [(t("GGUF — recommended and proven"), "krea2-turbo"),
                 (t("INT8 ConvRot — experimental, RTX 30xx"),
                  "krea2-turbo-int8")],
                value="krea2-turbo", label="Diffusion model format",
                info="Compare the two with the A/B test in Settings.")
        else:
            variant_model = gr.State(model_id)

        with gr.Row():
            # ----- Entrées -----
            with gr.Column(scale=3):
                # Le PROMPT d'abord : c'est le champ principal, il ne doit pas
                # être enterré sous des accordéons. Tout le reste (styles,
                # réglages de l'améliorateur) est replié en dessous, groupé par
                # intention : « 🎨 Styles » d'un côté, « ✨ Améliorateur » de
                # l'autre. Les explications longues vivent dans l'onglet
                # « 🧹 Gestion & aide » ; ici on s'en tient à une ligne par bloc.
                prompt = gr.Textbox(label="Prompt", lines=3,
                                    placeholder="Describe the image…")
                negative = gr.Textbox(label="Negative prompt", lines=1,
                                      visible=d.get("supports_negative", False))

                # GÉNÉRER juste sous le prompt : c'est l'action principale, elle
                # ne doit pas être au bout d'une colonne de réglages.
                with gr.Row(elem_classes="go-row"):
                    # Libellé court : le modèle est déjà écrit sur l'onglet.
                    run = gr.Button("🎨 Generate", variant="primary", size="lg",
                                    scale=4)
                    stop = gr.Button("⏹️ Stop", variant="stop", scale=1,
                                     min_width=90)
                    clear_prompt = gr.Button("🗑️ Effacer", scale=1,
                                             min_width=110)
                status_md = gr.Markdown("")

                # ----- ✨ Amélioration du prompt -----
                with gr.Row():
                    enhance_btn = gr.Button("✨ Enhance prompt (AI)",
                                            size="sm", scale=3)
                    enh_level = gr.Dropdown(
                        [(t("Light"), "light"), (t("Medium"), "medium"),
                         (t("Strong"), "strong")],
                        value="medium", label="Strength", scale=2)
                # Récapitulatif VIVANT : dit en clair ce que le bouton va faire,
                # AVANT de cliquer. Sans ça on règle des menus sans savoir ce
                # qu'ils changent concrètement sur le prompt.
                enh_recap = gr.Markdown("", elem_classes="hint")
                enh_msg = gr.Markdown("", elem_classes="feedback")
                enh_undo = gr.Button("↩️ Restore the original prompt",
                                     size="sm", visible=False)
                enh_props = gr.Radio([], label="Suggestions — click one to "
                                               "use it", visible=False)
                # Prompt d'avant amélioration, pour pouvoir revenir en arrière.
                prompt_before = gr.State("")

                # ----- 🎨 Styles : UN repli, puis des ONGLETS -----
                # Il y avait ici trois accordéons dans un quatrième : choisir
                # un style photo demandait deux dépliages, et rien ne montrait
                # que les deux autres banques existaient. Des onglets à
                # l'intérieur du repli coûtent le même espace une fois fermés,
                # mais ouverts ils montrent les trois pistes d'un coup d'œil.
                _photo_labels = styles.photo_style_labels()
                _art_labels = styles.bank_labels(styles.ART_STYLES_FILE)
                _photo_neg_note = ("" if d.get("supports_negative", False) else
                                   " · negatives have no effect here (CFG 1.0)")
                # Libellé traduit AVANT interpolation : une f-string composée
                # ne peut pas servir de clé de traduction (le nombre change).
                with gr.Accordion(
                        t("🎨 Styles — presets, photo, artistic ({n} styles)").format(
                              n=len(_photo_labels) + len(_art_labels)),
                        open=False):
                    gr.Markdown(
                        "All three **stack**: the preset is prepended, the "
                        "photo and artistic styles dress up your subject. "
                        "Details of each in “🧹 Manage & help”.")

                    with gr.Tabs():
                        with gr.Tab("🎭 Custom preset"):
                            system_prompt = gr.Textbox(
                                label="Prepended to every generation",
                                lines=2,
                                placeholder="e.g. watercolor style, pastel "
                                            "palette, soft lighting")
                            with gr.Row():
                                style_pick = gr.Dropdown(
                                    _style_choices(), value=_NONE_STYLE, scale=3,
                                    label="Saved presets",
                                    info="“— None —” removes the applied style.",
                                    allow_custom_value=False)
                                style_name = gr.Textbox(
                                    label="Name of the preset to save", scale=2,
                                    placeholder="e.g. Pastel watercolor")
                            with gr.Row():
                                # NE PAS APPLIQUER ≠ SUPPRIMER. Le bouton rouge
                                # efface le préréglage du disque ; c'est rarement ce
                                # qu'on veut, donc l'action courante — juste ne plus
                                # l'appliquer — a son propre bouton, en premier.
                                style_off = gr.Button("✖️ Stop applying",
                                                      size="sm")
                                style_save = gr.Button("💾 Save", size="sm")
                                style_refresh = gr.Button("↻ Refresh", size="sm")
                            with gr.Row():
                                style_del = gr.Button(
                                    "🗑️ Delete this preset (permanent)",
                                    size="sm", variant="stop")
                            style_msg = gr.Markdown("", elem_classes="feedback")
                            # Suppression en DEUX temps : le 1er clic arme, le 2e
                            # confirme. Un clic distrait ne détruit plus rien.
                            style_armed = gr.State(None)

                        # Banque photo (© ghleg, MIT) : le sujet du prompt est
                        # inséré dans chaque style coché ; les négatifs des styles ne
                        # sont repris que si le modèle en tient compte (CFG > 1).
                        with gr.Tab(t("📷 Photo ({n})").format(n=len(_photo_labels))):
                            photo_pick = gr.Dropdown(
                                _photo_labels, value=[], multiselect=True,
                                label="Styles to combine (by category)",
                                info="Quality, light, lens, film stock, mood… "
                                     "Your subject is inserted into every "
                                     "ticked style." + _photo_neg_note
                                     + " · Banque © ghleg, MIT.",
                                allow_custom_value=False)

                        # Banque artistique : sans négatifs, la description du style
                        # est simplement ajoutée après le sujet. 🎲 = wildcard.
                        with gr.Tab(t("🖍️ Artistic ({n})").format(n=len(_art_labels))):
                            art_pick = gr.Dropdown(
                                _art_labels, value=[], multiselect=True,
                                label="Styles to combine (by category)",
                                info="Anime, cartoon, comics, drawing, "
                                     "design, painting… Appended after your "
                                     "subject. Provenance of the collection "
                                     "is unconfirmed.",
                                allow_custom_value=False)
                            art_dice = gr.Button("🎲 Random (wildcard)", size="sm")

                # ----- ✨ Améliorateur : réglages ET installation ensemble -----
                # C'était éclaté en deux accordéons séparés par le champ négatif.
                # Ouvert d'office tant que l'add-on n'est pas installé.
                _enh_ready = tools.enhance_is_installed()
                with gr.Accordion("✨ Prompt improver — settings"
                                  + ("" if _enh_ready else " & installation"),
                                  open=not _enh_ready):
                    gr.Markdown(
                        "The grey line under the menus **says plainly what "
                        "the button will do**.")
                    with gr.Row():
                        enh_variants = gr.Radio(
                            [("1", 1), ("2", 2), ("4", 4)], value=4,
                            label="Propositions",
                            info="How many prompts to suggest. Produced in a "
                                 "single model load: 4 cost almost the same "
                                 "time as 1.")

                    # Ce repli n'a d'objet que tant que l'add-on manque. Une
                    # fois installé, « ⬇️ Installation ✅ déjà installé » est un
                    # accordéon qui ne dit rien et qu'on ouvre pour rien : il
                    # disparaît. La réinstallation passe par « 🧹 Gestion &
                    # aide », qui est l'endroit prévu pour réparer.
                    with gr.Accordion("⬇️ Installation (1 clic)",
                                      open=not _enh_ready,
                                      visible=not _enh_ready):
                        gr.Markdown(
                            "A small LLM (**Qwen2.5-3B-Instruct**, PyTorch, "
                            "~6 GB) loaded and then unloaded on every call: "
                            "**no VRAM conflict** with generation. Nothing to "
                            "type.")
                        enh_log = gr.Textbox(
                            label="Install log", lines=6,
                            autoscroll=True, elem_classes="log-box")
                        enh_inst = gr.Button(
                            "⬇️ Install the prompt enhancer")

                        def _install_enh():
                            for msg in tools.install_enhance_stream():
                                yield msg

                        enh_inst.click(_install_enh, outputs=[enh_log])

                _acc_title = ("🖼️ Reference images (image editing)" if is_edit
                              else "🖼️ Reference / starting image (image-to-image)")
                # Ouvert par défaut sur un modèle d'édition (Flux.2) : l'édition
                # est une capacité phare, on la met en avant.
                with gr.Accordion(_acc_title, open=is_edit):
                    if is_edit:
                        gr.Markdown(
                            "**Edit an image**: load it and describe **the "
                            "change** in the prompt (e.g. *“change the car "
                            "color to red”*, *“add snow”*). Editing is "
                            "prompt-driven (no strength slider). The output "
                            "aspect follows your image. You can add **2 extra "
                            "reference images** to combine elements (e.g. "
                            "*“put the character from image 1 into the scene "
                            "of image 2”*).")
                    else:
                        gr.Markdown(
                            "**Reference image** (image-to-image): load a "
                            "photo, describe the desired result, and set the "
                            "**transformation strength** — **low (0.2–0.4)** "
                            "= keeps the reference's structure; **high "
                            "(0.7–1.0)** = reinvented. The output aspect "
                            "follows your image.")
                        if edit_optional:
                            gr.Markdown(
                                "✏️ **Or: Edit mode (Ostris Edit)** — the "
                                "image becomes a **context reference** "
                                "(style, subject…) instead of a starting "
                                "point. Requires a **Krea 2 editing LoRA** — "
                                "one-click install button below — and an "
                                "up-to-date sd.cpp engine "
                                "(`update-engine.bat`). 💡 Editing costs more "
                                "VRAM (reference tokens): the reference is "
                                "downscaled automatically, and the automatic "
                                "quantization (Q4_K_M on 11–12 GB) leaves the "
                                "headroom it needs.")
                    init_image = gr.Image(
                        label="Image to edit" if is_edit else "Starting image",
                        type="pil",
                        buttons=widgets.IMAGE_VIEW_ONLY)
                    if is_edit:
                        with gr.Row():
                            ref_image2 = gr.Image(
                                label="Reference 2 (optional)", type="pil",
                                buttons=widgets.IMAGE_VIEW_ONLY)
                            ref_image3 = gr.Image(
                                label="Reference 3 (optional)", type="pil",
                                buttons=widgets.IMAGE_VIEW_ONLY)
                    else:
                        ref_image2 = gr.State(None)
                        ref_image3 = gr.State(None)
                    strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05,
                                         label="Transformation strength",
                                         visible=not is_edit)
                    if edit_optional:
                        edit_mode = gr.Checkbox(
                            value=False,
                            label="✏️ Edit mode (editing LoRA required)")
                        if d.get("edit_lora"):
                            edit_lora_btn = gr.Button(
                                "⬇️ Install the official editing LoRA (one click)",
                                size="sm")
                            edit_lora_msg = gr.Markdown("")
                    else:
                        edit_mode = gr.State(False)
                    if is_edit:
                        outpaint = gr.Slider(
                            1.0, 2.0, value=1.0, step=0.1,
                            label="🧩 Centred outpaint — extend the canvas "
                                  "(1.0 = off; ⚠️ experimental)",
                            info="Enlarges the canvas symmetrically and lets "
                                 "the model fill the borders. For a "
                                 "directional outpaint "
                                 "(left/right/top/bottom), with no prompt and "
                                 "with any model, use the “🖼️ Outpaint” tab.")
                    else:
                        outpaint = gr.State(1.0)

                with gr.Accordion("🧩 LoRA", open=False):
                    with gr.Row():
                        lora1 = gr.Dropdown(label="LoRA 1",
                                            choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora1_w = gr.Slider(0.0, 5000.0, value=0.8, step=0.05,
                                            label="Weight")
                    with gr.Row():
                        lora2 = gr.Dropdown(label="LoRA 2",
                                            choices=gen_engine.list_loras(),
                                            value=None, allow_custom_value=False)
                        lora2_w = gr.Slider(0.0, 5000.0, value=0.8, step=0.05,
                                            label="Weight")
                    with gr.Row():
                        refresh_lora = gr.Button("↻ Refresh list", size="sm")
                        clear_lora = gr.Button("✖ Clear LoRAs", size="sm")
                    gr.Markdown(t("Drop your LoRA files into `{dir}`")
                                .format(dir=settings.LORA_DIR))
                    with gr.Row():
                        civitai_ref = gr.Textbox(
                            label="Import a Civitai LoRA (URL or version ID)",
                            scale=3, placeholder="https://civitai.com/…"
                                                 "?modelVersionId=3067151")
                        civitai_btn = gr.Button("⬇️ Import", scale=1)
                    civitai_msg = gr.Markdown("")

                with gr.Accordion("📂 Local files (custom model)", open=False):
                    gr.Markdown(t(
                        "To use a model **downloaded elsewhere**: drop the "
                        "file(s) into `{dir}` then select it below. Empty = "
                        "catalog model.")
                        .format(dir=settings.CUSTOM_DIR))
                    custom_diff = gr.Dropdown(gen_engine.list_custom_models(),
                                              value=None, label="Diffusion (local)")
                    with gr.Row():
                        custom_vae = gr.Dropdown(gen_engine.list_custom_models(),
                                                 value=None, label="VAE (local)")
                        custom_enc = gr.Dropdown(gen_engine.list_custom_models(),
                                                 value=None, label="Encoder (local)")
                    with gr.Row():
                        refresh_custom = gr.Button("↻ Refresh local files",
                                                   size="sm")
                        clear_custom = gr.Button("✖ Clear custom fields",
                                                 size="sm")

                ratio = gr.Dropdown(
                    [t(k) for k in ratios],
                    value=t(_ratio_label(ratios, d.get("width", 1024),
                                         d.get("height", 1024))),
                    label="Aspect ratio")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=d.get("width", 1024), step=16,
                                      label="Width")
                    height = gr.Slider(256, 2048, value=d.get("height", 1024), step=16,
                                       label="Height")
                with gr.Row():
                    steps = gr.Slider(1, 60, value=d.get("steps", 8), step=1,
                                      label="Steps")
                    cfg = gr.Slider(0.0, 12.0, value=d.get("cfg_scale", 1.0),
                                    step=0.1, label="CFG",
                                    info="On sd.cpp, CFG disabled = 1.0 "
                                         "(normal for distilled models). 0.0 "
                                         "= pure unconditional: may IGNORE "
                                         "the prompt (Krea's “cfg 0” is its "
                                         "own convention, ≠ sd.cpp). >1 = "
                                         "guidance.")
                preset_list = _presets(model_id)
                preset = gr.Dropdown(
                    [t(p["name"]) for p in preset_list],
                    value=(t(preset_list[0]["name"]) if preset_list else None),
                    label="Preset (sampler/scheduler/steps)",
                    visible=bool(preset_list))
                # Menus ANNOTÉS (⭐ recommandé · △ peu adapté · ⚠️ déconseillé)
                # et fiche qui suit la sélection. Le verdict dépend du MODÈLE :
                # distillé à CFG 1.0, en flow matching, sur 4 à 8 pas — trois
                # propriétés qui disqualifient la moitié du menu.
                with gr.Row():
                    sampler = gr.Dropdown(
                        sampling.choices("sampler", family),
                        value=d.get("sampler", "euler"), label="Sampler",
                        info="⭐ recommended · △ poorly suited · ⚠️ "
                             "discouraged for THIS model")
                    schedule = gr.Dropdown(
                        sampling.choices("schedule", family),
                        value=d.get("scheduler", "auto"),
                        label="Scheduler (sigmas)",
                        info="How the denoising steps are spread out")
                with gr.Row():
                    sampler_doc = gr.Markdown(
                        sampling.describe("sampler", d.get("sampler", "euler"),
                                          family), elem_classes="hint")
                    schedule_doc = gr.Markdown(
                        sampling.describe("schedule",
                                          d.get("scheduler", "auto"), family),
                        elem_classes="hint")
                sampler.change(
                    lambda k: sampling.describe("sampler", k, family),
                    inputs=[sampler], outputs=[sampler_doc])
                schedule.change(
                    lambda k: sampling.describe("schedule", k, family),
                    inputs=[schedule], outputs=[schedule_doc])
                with gr.Accordion("📖 Why half of this menu is useless here",
                                  open=False):
                    gr.Markdown(sampling.rationale(family))
                flow_shift = gr.Slider(
                    0.0, 12.0, value=float(d.get("flow_shift", 0.0)), step=0.1,
                    label="Flow shift",
                    info="Leave at 0 (auto): the model picks the right value "
                         "for the resolution. Too low (1–2) leaves "
                         "GRAIN/noise at high resolution; ~3–4 reinforces "
                         "structure.")
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 = random)",
                                     precision=0)
                    batch = gr.Slider(1, 8, value=1, step=1, label="Images")

                # (« Générer » / « Annuler » sont en haut, sous le prompt.)

            # ----- Sorties : aperçu temps réel (Image dédiée) + résultats (Gallery)
            # L'aperçu est une gr.Image SÉPARÉE (un seul <img> mis à jour sur
            # place) et non fusionné dans la galerie : mettre à jour une Gallery
            # à chaque frame reconstruit toute la grille et fait « clignoter ».
            with gr.Column(scale=4):
                # (La ligne de statut est sous le bouton Générer, à gauche : la
                # progression se lit là où on vient de cliquer. Elle reste une
                # mise à jour « valeur seule », donc AUCUN overlay sur l'aperçu —
                # c'est l'overlay de gr.Progress qui le faisait clignoter.)
                preview_img = gr.Image(
                    label="Live preview", visible=False, height=560,
                    format="png", buttons=widgets.IMAGE_VIEW_ONLY,
                    show_label=True, interactive=False)
                gallery = gr.Gallery(
                    label="Results (caption = seed)",
                    columns=2, height=560, object_fit="contain", show_label=True,
                    format="png", buttons=widgets.IMAGE_BUTTONS)
                with gr.Row():
                    seed_box = gr.Textbox(label="Selected image's seed",
                                          interactive=False,
                                          buttons=widgets.TEXT_COPY,
                                          scale=2)
                    seed_reuse = gr.Button("♻️ Reuse this seed", size="sm",
                                           scale=1)
                send_tool = gr.Dropdown(
                    [(t("🌐 Depth"), "depth"),
                     (t("✂️ Background removal"), "bg"),
                     (t("🪄 Cut out an object (SAM)"), "sam"),
                     (t("🔼 Agrandir (ESRGAN)"), "esrgan"),
                     (t("✨ Creative upscale (SDXL)"), "creative")],
                    value=None, label="📤 Send the selection to the Toolkit",
                    visible=pending_toolkit is not None)
                send_3d = gr.Button("🧊 Send the selection to Image → 3D",
                                    size="sm",
                                    visible=pending_3d is not None)
                send_op = gr.Button("🖼️ Send the selection to Outpaint",
                                    size="sm",
                                    visible=pending_outpaint is not None)
                logbox = gr.Textbox(label="Log", lines=10, max_lines=24,
                                    autoscroll=True, elem_classes="log-box")

        last_paths = gr.State([])
        last_seeds = gr.State([])
        sel_index = gr.State(0)

        # ----- Comportements -----
        def refresh_loras():
            choices = gen_engine.list_loras()
            return (gr.update(choices=choices), gr.update(choices=choices), "")

        refresh_lora.click(refresh_loras, outputs=[lora1, lora2, civitai_msg])

        def clear_loras():
            return (gr.update(value=None), gr.update(value=0.8),
                    gr.update(value=None), gr.update(value=0.8))

        clear_lora.click(clear_loras, outputs=[lora1, lora1_w, lora2, lora2_w])

        def _civitai_import(ref):
            if not (ref or "").strip():
                raise gr.Error(t("Paste a Civitai URL or version ID."))
            try:
                name = downloader.download_lora_civitai(ref)
            except Exception as exc:  # noqa: BLE001
                raise gr.Error(str(exc))
            choices = gen_engine.list_loras()
            return (gr.update(choices=choices), gr.update(choices=choices),
                    t("✓ LoRA imported: **{name}** — select it above."
                      ).format(name=name))

        civitai_btn.click(_civitai_import, inputs=[civitai_ref],
                          outputs=[lora1, lora2, civitai_msg])

        # LoRA d'édition officiel (Mode édition) : téléchargé depuis le dépôt HF
        # déclaré par le catalogue (defaults.edit_lora), sélectionné en LoRA 1
        # (poids 1.0) et Mode édition coché — prêt à générer.
        if edit_optional and d.get("edit_lora"):
            def _get_edit_lora():
                try:
                    name = downloader.download_lora(d["edit_lora"])
                except Exception as exc:  # noqa: BLE001
                    raise gr.Error(str(exc))
                choices = gen_engine.list_loras()
                return (gr.update(choices=choices, value=name),
                        gr.update(choices=choices),
                        gr.update(value=1.0),
                        gr.update(value=True),
                        t("✅ Editing LoRA **{name}** installed, selected "
                          "(LoRA 1) and Edit mode switched on. Load an image "
                          "and describe the result you want.").format(name=name))

            edit_lora_btn.click(_get_edit_lora,
                                outputs=[lora1, lora2, lora1_w, edit_mode,
                                         edit_lora_msg])

        def _refresh_custom():
            c = gen_engine.list_custom_models()
            return gr.update(choices=c), gr.update(choices=c), gr.update(choices=c)

        refresh_custom.click(_refresh_custom,
                             outputs=[custom_diff, custom_vae, custom_enc])

        def _clear_custom():
            return gr.update(value=None), gr.update(value=None), gr.update(value=None)

        clear_custom.click(_clear_custom,
                           outputs=[custom_diff, custom_vae, custom_enc])

        # --- Styles enregistrés (prompt système) ---
        # « — Aucun — » n'est pas un style enregistré : le choisir VIDE le champ
        # système (retire le style appliqué) sans rien supprimer.
        def _load_style(name):
            if not name or name == _NONE_STYLE:
                return gr.update(value="")
            return gr.update(value=styles.get_style(name))

        style_pick.change(_load_style, inputs=[style_pick],
                          outputs=[system_prompt])

        def _save_style(name, text):
            try:
                saved = styles.save_style(name, text)
            except ValueError as exc:
                raise gr.Error(str(exc))
            return (gr.update(choices=_style_choices(), value=saved),
                    gr.update(value=""))

        style_save.click(_save_style, inputs=[style_name, system_prompt],
                         outputs=[style_pick, style_name])

        # Supprimer : retire le style enregistré ET réinitialise (« Aucun » +
        # champ système vidé) — le texte appliqué ne « survit » plus à la
        # suppression de son preset.
        def _style_off():
            """Retire le style de la GÉNÉRATION. Ne touche à rien sur le disque."""
            return (gr.update(value=_NONE_STYLE), gr.update(value=""),
                    gr.update(value="✖️ " + t("Style removed from "
                                                  "generation. The preset is "
                                                  "kept.")),
                    None)

        style_off.click(_style_off,
                        outputs=[style_pick, system_prompt, style_msg,
                                 style_armed])

        def _delete_style(name, armed):
            if not name or name == _NONE_STYLE:
                return (gr.update(), gr.update(),
                        gr.update(value="⚠️ " + t("Pick a preset to "
                                                      "delete first.")),
                        None)
            if armed != name:
                # 1er clic : on arme et on prévient, sans rien détruire.
                return (gr.update(), gr.update(),
                        gr.update(value="⚠️ **" + t("Delete “{n}” for "
                                                        "good?").format(n=name)
                                  + "** " + t("Click again to confirm. To "
                                              "merely stop applying it, use "
                                              "“✖️ Stop applying”.")),
                        name)
            try:
                styles.delete_style(name)
            except ValueError as exc:      # préréglage livré : on explique
                return (gr.update(choices=_style_choices(), value=_NONE_STYLE),
                        gr.update(value=""), gr.update(value=f"ℹ️ {exc}"), None)
            return (gr.update(choices=_style_choices(), value=_NONE_STYLE),
                    gr.update(value=""),
                    gr.update(value="🗑️ " + t("“{n}” deleted.").format(n=name)),
                    None)

        style_del.click(_delete_style, inputs=[style_pick, style_armed],
                        outputs=[style_pick, system_prompt, style_msg,
                                 style_armed])

        def _refresh_styles():
            return gr.update(choices=_style_choices())

        style_refresh.click(_refresh_styles, outputs=[style_pick])

        # 🎲 Wildcard : ajoute un style artistique tiré au hasard (hors ceux déjà
        # sélectionnés) à la sélection courante.
        def _roll_art(current):
            lab = styles.random_bank_label(styles.ART_STYLES_FILE,
                                           exclude=current or [])
            if not lab:
                return gr.update()
            return gr.update(value=list(current or []) + [lab])

        art_dice.click(_roll_art, inputs=[art_pick], outputs=[art_pick])

        def on_ratio(label):
            w, h = ratios.get(i18n.to_source(label), (0, 0))
            if not w:
                return gr.update(), gr.update()
            return gr.update(value=w), gr.update(value=h)

        ratio.change(on_ratio, inputs=[ratio], outputs=[width, height])

        # --- Améliorateur de prompt (LLM) ---
        # System prompt adapté au modèle : Krea 2 -> guide Krea ; sinon générique.
        # Même guide de prompt pour Raw que pour le Turbo : c'est le même
        # modèle, seule la distillation les sépare.
        _enh_auto = "krea2" if family.startswith("krea2") else "generic"

        def _enhance(sys_prompt, text, level, variants):
            if not (text or "").strip():
                raise gr.Error(t("Enter a prompt to enhance first."))
            try:
                props = tools.enhance_prompt_variants(
                    text.strip(), style=_enh_auto, level=level or "medium",
                    variants=int(variants or 1),
                    style_constraint=(sys_prompt or "").strip())
            except tools.ToolError as exc:
                raise gr.Error(str(exc))
            except Exception as exc:  # noqa: BLE001
                raise gr.Error(f"Improvement failed: {exc}")

            # Feedback : ce qui a été fait, en une ligne repérable.
            bits = []
            if len(props) > 1:
                bits.append(t("{n} propositions").format(n=len(props)))
            if (sys_prompt or "").strip():
                bits.append(t("active style respected"))
            msg = "✅ **" + t("Improved prompt") + "**"
            if bits:
                msg += " — " + " · ".join(bits)
            if len(props) > 1:
                msg += "  \n" + t("Click a suggestion below to use it instead.")
            # La 1re proposition part dans le champ ; les autres restent
            # cliquables. Libellés numérotés : un prompt long casse un Radio.
            choices = [(f"{i + 1}. {p[:110]}{'…' if len(p) > 110 else ''}", p)
                       for i, p in enumerate(props)]
            return (gr.update(value=props[0]),
                    gr.update(choices=choices, value=props[0],
                              visible=len(props) > 1),
                    gr.update(value=msg), text or "", gr.update(visible=True))

        enhance_btn.click(
            _enhance,
            inputs=[system_prompt, prompt, enh_level, enh_variants],
            outputs=[prompt, enh_props, enh_msg, prompt_before, enh_undo])

        # --- Récapitulatif VIVANT : ce que « Améliorer » va faire ---------
        # On traduit les réglages en une phrase lisible, sinon impossible de
        # savoir ce qu'on règle sans lire la doc.
        def _recap(level, variants):
            n = int(variants or 1)
            parts = [{"light": t("light touch-up"),
                      "medium": t("balanced enrichment"),
                      "strong": t("full expansion")}.get(level or "medium", "")]
            parts.append(t("1 suggestion only") if n == 1 else
                         t("{n} propositions au choix").format(n=n))
            return "→ " + "  ·  ".join(p for p in parts if p)

        _recap_in = [enh_level, enh_variants]
        for _c in _recap_in:
            _c.change(_recap, inputs=_recap_in, outputs=[enh_recap])
        # Valeur de départ : la ligne doit être remplie AVANT toute interaction.
        enh_recap.value = _recap(enh_level.value, enh_variants.value)

        # --- Rétablir le prompt d'avant amélioration ---
        def _undo_enhance(before):
            return (gr.update(value=before or ""),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value="↩️ " + t("Original prompt restored.")),
                    gr.update(visible=False))

        enh_undo.click(_undo_enhance, inputs=[prompt_before],
                       outputs=[prompt, enh_props, enh_msg, enh_undo])

        # --- Tout effacer (prompt + traces d'amélioration) ---
        def _clear_all():
            return (gr.update(value=""), gr.update(value=""),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=""), gr.update(visible=False), "")

        clear_prompt.click(
            _clear_all,
            outputs=[prompt, negative, enh_props, enh_msg, enh_undo,
                     prompt_before])

        # Cliquer une proposition la met dans le champ Prompt (.input : ne se
        # redéclenche pas quand l'améliorateur remplit le Radio lui-même).
        enh_props.input(lambda p: gr.update(value=p) if p else gr.update(),
                        inputs=[enh_props], outputs=[prompt])


        def _fit_to_ref(img):
            """img2img : cale la sortie sur le format de l'image de départ
            (côté long plafonné à 1024 px, multiples de 16)."""
            if img is None:
                return gr.update(), gr.update(), gr.update()
            w0, h0 = img.size
            longest = max(w0, h0) or 1
            sc = 1024 / longest if longest > 1024 else 1.0
            w = max(256, min(2048, int(round(w0 * sc / 16)) * 16))
            h = max(256, min(2048, int(round(h0 * sc / 16)) * 16))
            return (gr.update(value=w), gr.update(value=h),
                    gr.update(value=t(_CUSTOM_LABEL)))

        init_image.upload(_fit_to_ref, inputs=[init_image],
                          outputs=[width, height, ratio])

        if preset_list:
            def apply_preset(name):
                name = i18n.to_source(name)
                for p in preset_list:
                    if p["name"] == name:
                        return (gr.update(value=p.get("sampler", "euler")),
                                gr.update(value=p.get("scheduler", "auto")),
                                gr.update(value=int(p.get("steps", 8))),
                                gr.update(value=float(p.get("cfg_scale", 1.0))))
                return gr.update(), gr.update(), gr.update(), gr.update()

            preset.change(apply_preset, inputs=[preset],
                          outputs=[sampler, schedule, steps, cfg])

        def do_generate(selected_model_id, system_prompt, prompt, negative,
                        photo_styles, art_styles,
                        init_image,
                        ref_image2, ref_image3, strength, outpaint, edit_mode,
                        width, height, steps, cfg, sampler, schedule, flow_shift,
                        seed, batch, lora1, lora1_w, lora2, lora2_w,
                        custom_diff, custom_vae, custom_enc):
            # NB : PAS de gr.Progress() ici — son overlay se dessine PAR-DESSUS
            # les sorties (dont l'aperçu) à chaque mise à jour → c'était LA cause
            # du clignotement « on voit la barre 1/8 entre deux pas ». Toute la
            # progression passe par la ligne de statut texte (status_md).
            import queue
            import threading
            import time

            if not (prompt or "").strip():
                raise gr.Error(t("Enter a prompt (describe the image, or the "
                                 "change to apply)."))

            full_prompt = prompt or ""
            if (system_prompt or "").strip():
                full_prompt = f"{system_prompt.strip()}, {full_prompt}".strip(", ")

            # 📷 Styles photo Krea 2 (cumulables) : le sujet est inséré dans
            # chaque style choisi ; les négatifs des styles ne sont combinés que
            # si le modèle en tient compte (supports_negative / CFG > 1). Sur un
            # modèle distillé CFG 1.0 (Krea 2 Turbo, champ négatif masqué) ils
            # seraient du poids mort : on les laisse tomber.
            neg_text = negative or ""
            if photo_styles:
                full_prompt, photo_negs = styles.apply_photo_styles(
                    full_prompt, photo_styles)
                if d.get("supports_negative", False) and photo_negs:
                    extra = ", ".join(dict.fromkeys(photo_negs))
                    neg_text = (f"{neg_text}, {extra}".strip(", ")
                                if neg_text.strip() else extra)
            # 🎨 Styles artistiques (sans négatifs) : ajoutés après le sujet,
            # après d'éventuels styles photo (les deux banques s'empilent).
            if art_styles:
                full_prompt, _ = styles.apply_style_bank(
                    full_prompt, art_styles, styles.ART_STYLES_FILE)

            try:
                base_seed = int(seed)
            except (TypeError, ValueError):   # champ vidé -> aléatoire
                base_seed = -1
            if base_seed < 0:
                base_seed = random.randint(0, 2**31 - 1)

            settings.ensure_dirs()
            # Modèle d'édition (Flux.2) -> image(s) via -r (ref_image) ; sinon
            # img2img classique via -i (init_image) + force.
            init_path = None
            ref_paths: list = []
            of = float(outpaint) if outpaint else 1.0
            if is_edit and init_image is not None and of > 1.01:
                # Outpaint : on agrandit la toile et on demande au modèle de
                # remplir les bords. Fond = image étirée+floutée (continuation
                # plausible), image d'origine recollée au centre.
                from PIL import Image as _PI, ImageFilter as _IF
                ow, oh = init_image.size
                nw = max(256, min(2048, int(round(ow * of / 16)) * 16))
                nh = max(256, min(2048, int(round(oh * of / 16)) * 16))
                bg = init_image.convert("RGB").resize((nw, nh), _PI.LANCZOS)
                bg = bg.filter(_IF.GaussianBlur(28))
                bg.paste(init_image.convert("RGB"), ((nw - ow) // 2, (nh - oh) // 2))
                rp = settings.TMP_DIR / "outpaint_ref.png"
                bg.save(rp)
                ref_paths.append(rp)
                width, height = nw, nh
                full_prompt = (full_prompt + ", extend and naturally fill the "
                               "outer borders to complete the scene seamlessly, "
                               "matching perspective, lighting and content").strip(", ")
            elif is_edit:
                # Édition multi-référence : jusqu'à 3 images (-r répété).
                for i, im in enumerate((init_image, ref_image2, ref_image3)):
                    if im is not None:
                        rp = settings.TMP_DIR / f"edit_ref{i + 1}.png"
                        im.save(rp)
                        ref_paths.append(rp)
            elif edit_mode and init_image is not None:
                # Mode édition Ostris Edit (Krea 2) : l'image passe en référence
                # de contexte (-r + mmproj vision), pas en img2img. Un LoRA
                # d'édition doit être chargé pour que ça fasse quelque chose.
                # La référence est RÉDUITE (≤ 832 px de côté long) : ses tokens
                # s'ajoutent à la séquence du DiT, et une réf 1024² fait déborder
                # la VRAM d'une carte 12 Go (OOM constaté sur RTX 3060 : buffer
                # de calcul 2,5 Go + modèle Q5 8,4 Go). Le modèle est entraîné
                # avec des réfs réduites : aucune perte réelle de qualité.
                im = init_image
                longest = max(im.size)
                if longest > 832:
                    from PIL import Image as _PI
                    sc = 832 / longest
                    nw = max(64, int(round(im.width * sc / 16)) * 16)
                    nh = max(64, int(round(im.height * sc / 16)) * 16)
                    im = im.convert("RGB").resize((nw, nh), _PI.LANCZOS)
                rp = settings.TMP_DIR / "edit_ref1.png"
                im.save(rp)
                ref_paths.append(rp)
            elif init_image is not None:
                init_path = settings.TMP_DIR / "i2i_init.png"
                init_image.save(init_path)

            loras = [(lora1, float(lora1_w)), (lora2, float(lora2_w))]
            loras = [(n, w) for n, w in loras if n]

            preview_path = settings.TMP_DIR / f"preview_{int(time.time()*1000)}.png"
            try:
                preview_path.unlink()
            except OSError:
                pass

            total = max(1, int(steps))
            step_re = re.compile(rf"(\d+)\s*/\s*{total}\b")
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    outs = gen_engine.generate(
                        model_id=selected_model_id or model_id, prompt=full_prompt,
                        negative=neg_text or "", steps=int(steps),
                        cfg_scale=float(cfg), width=int(width), height=int(height),
                        seed=base_seed, batch_count=int(batch), sampler=sampler,
                        schedule=schedule, flow_shift=float(flow_shift or 0.0),
                        init_image=init_path, strength=float(strength),
                        ref_image=(ref_paths or None), loras=loras,
                        diffusion_override=gen_engine.custom_path(custom_diff),
                        vae_override=gen_engine.custom_path(custom_vae),
                        encoder_override=gen_engine.custom_path(custom_enc),
                        preview_path=preview_path, log=q.put)
                    state["outs"] = [str(p) for p in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            last_mtime = None
            last_emit = 0.0
            last_log_len = 0
            status = t("⏳ Loading the model…")
            last_status = ""
            # UNE seule bascule d'affichage : aperçu VISIBLE, galerie MASQUÉE.
            # Ensuite, par frame, on ne change QUE des VALEURS (statut texte,
            # image de l'aperçu) → aucun overlay, aucun re-montage, aucun reflow.
            yield (status, gr.update(visible=True, value=None),
                   gr.update(visible=False), "", gr.update(), gr.update())
            while True:
                try:
                    line = q.get(timeout=0.3)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    mt = step_re.search(line)
                    if mt:
                        cur = min(int(mt.group(1)), total)
                        status = t("🎨 Step {cur}/{total}").format(
                            cur=cur, total=total)
                    # Barres de progression sd.cpp : converties en statut texte,
                    # jamais ajoutées au journal (flood) ni affichées en overlay.
                    is_bar = bool(_PROGRESS_BAR.search(line)) or "\x1b" in line
                    if is_bar and not mt:  # barre de CHARGEMENT (tenseurs)
                        lm = re.search(r"(\d+)\s*/\s*(\d+)", line)
                        if lm and int(lm.group(2)) > 0:
                            status = t("⏳ Loading the model… {c}/{t}").format(
                                c=lm.group(1), t=lm.group(2))
                    if not is_bar:
                        logs.append(line)
                # Aperçu dans l'Image DÉDIÉE : nouvelle frame SEULEMENT si le
                # fichier a changé (mtime). On lit en mémoire (copie PIL) car sous
                # Windows sd-cli écrit ce fichier en continu (verrou en écriture).
                prev = gr.update()
                new_prev = False
                if preview_path.exists():
                    try:
                        m = preview_path.stat().st_mtime
                        if m != last_mtime:
                            from PIL import Image
                            with Image.open(preview_path) as _pim:
                                _img = _pim.copy()
                            # Valeur SEULE (visibilité déjà True) → l'<img> est
                            # remplacé sur place, sans re-montage ni reflow.
                            prev = gr.update(value=_img)
                            last_mtime = m
                            new_prev = True
                    except (OSError, ValueError):
                        pass
                # On n'émet QUE s'il y a du NOUVEAU : frame d'aperçu (tout de
                # suite), statut qui change (≤4x/s, ex. compteur de chargement),
                # ou vraie ligne de journal (≤1x/s). Sinon rien → zéro re-render.
                now = time.time()
                log_changed = len(logs) != last_log_len
                status_changed = status != last_status
                if new_prev or (status_changed and now - last_emit >= 0.25) \
                        or (log_changed and now - last_emit >= 1.0):
                    last_emit = now
                    last_log_len = len(logs)
                    last_status = status
                    yield (status, prev, gr.update(),
                           "\n".join(logs[-400:]), gr.update(), gr.update())

            if "err" in state:
                logs.append(f"\n[ERROR] {state['err']}")
                # Fin (erreur) : on remet l'affichage normal (galerie visible).
                yield (t("❌ Error — see the log below."),
                       gr.update(visible=False), gr.update(visible=True),
                       "\n".join(logs), gr.update(), gr.update())
                return
            paths = state.get("outs", [])

            seeds = [base_seed + i for i in range(len(paths))]
            items = [(p, f"seed {s}") for p, s in zip(paths, seeds)]
            # Fin : on masque l'aperçu et on RÉAFFICHE la galerie avec les résultats
            # (elle avait été masquée au début → il FAUT visible=True ici).
            yield (t("✅ Done — {n} image(s).").format(n=len(paths)),
                   gr.update(visible=False, value=None),
                   gr.update(value=items, visible=True),
                   "\n".join(logs), paths, seeds)

        _gen_io = dict(
            fn=do_generate,
            inputs=[variant_model, system_prompt, prompt, negative,
                    photo_pick, art_pick,
                    init_image,
                    ref_image2, ref_image3, strength, outpaint, edit_mode, width,
                    height, steps, cfg, sampler, schedule, flow_shift, seed, batch,
                    lora1, lora1_w, lora2, lora2_w,
                    custom_diff, custom_vae, custom_enc],
            outputs=[status_md, preview_img, gallery, logbox,
                     last_paths, last_seeds],
        )
        gen_evt = run.click(**_gen_io)
        # QOL : Ctrl+Entrée (ou Cmd+Entrée) depuis le prompt lance la
        # génération. Un champ multiligne n'a pas de « validation » naturelle,
        # et faire l'aller-retour jusqu'au bouton à chaque essai est le geste
        # qu'on répète le plus dans cette application.
        prompt.submit(**_gen_io)
        widgets.stop_into_status(stop, gen_engine.cancel, status_md,
                                 [gen_evt])

        # --- Seed : vidé -> -1 ; sélection -> affichage copiable ; réutiliser ---
        def _seed_default(v):
            return -1 if v in (None, "") else gr.update()

        seed.change(_seed_default, inputs=[seed], outputs=[seed])

        def _on_select(seeds, evt: gr.SelectData):
            i = evt.index if isinstance(evt.index, int) else 0
            s = seeds[i] if (seeds and 0 <= i < len(seeds)) else ""
            return i, str(s)

        gallery.select(_on_select, inputs=[last_seeds],
                       outputs=[sel_index, seed_box])

        def _reuse_seed(seeds, idx):
            if seeds and isinstance(idx, int) and 0 <= idx < len(seeds):
                return gr.update(value=int(seeds[idx]))
            return gr.update()

        seed_reuse.click(_reuse_seed, inputs=[last_seeds, sel_index],
                         outputs=[seed])

        if pending_toolkit is not None and tabs is not None:
            def _send_toolkit(paths, idx, dest):
                if not paths or not dest:
                    raise gr.Error(t("Generate then select an image."))
                i = idx if isinstance(idx, int) and 0 <= idx < len(paths) else 0
                return ((paths[i], dest), gr.Tabs(selected=toolkit_tab_id),
                        gr.update(value=None))

            send_tool.change(
                _send_toolkit, inputs=[last_paths, sel_index, send_tool],
                outputs=[pending_toolkit, tabs, send_tool])

        # Envoi de l'image sélectionnée vers l'onglet « Image → 3D ».
        if pending_3d is not None and tabs is not None:
            def _send_3d(paths, idx):
                if not paths:
                    raise gr.Error(t("Generate then select an image."))
                i = idx if isinstance(idx, int) and 0 <= idx < len(paths) else 0
                return paths[i], gr.Tabs(selected=threed_tab_id)

            send_3d.click(_send_3d, inputs=[last_paths, sel_index],
                          outputs=[pending_3d, tabs])

        # Envoi de l'image sélectionnée vers l'onglet « Outpaint ».
        if pending_outpaint is not None and tabs is not None:
            def _send_outpaint(paths, idx):
                if not paths:
                    raise gr.Error(t("Generate then select an image."))
                i = idx if isinstance(idx, int) and 0 <= idx < len(paths) else 0
                return paths[i], gr.Tabs(selected=outpaint_tab_id)

            send_op.click(_send_outpaint, inputs=[last_paths, sel_index],
                          outputs=[pending_outpaint, tabs])

        # Le champ Prompt est RENVOYÉ à l'appelant. C'est ce qui permet à
        # « 📝 Image → prompt » d'y déposer son texte directement, au clic,
        # plutôt que d'attendre un événement de changement d'onglet : une
        # sélection programmatique ne déclenche pas `Tabs.select`, donc le
        # texte n'arrivait jamais (vérifié dans le navigateur avant de câbler
        # autrement).
        return prompt
