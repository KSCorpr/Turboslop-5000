"""Onglet Toolkit : profondeur, détourage, SAM et upscale ESRGAN."""
from __future__ import annotations

import queue
import threading

import gradio as gr

from .. import downloader, hardware, registry, settings
from ..engine import generate as gen_engine
from ..engine import highres
from ..engine import tools
from ..i18n import t
from . import widgets
from . import preview

# Préréglages de l'upscale créatif.
#
# Un préréglage ne se réduit PAS à « un prompt + une créativité » : sur du
# dessin, ce qui décide de la propreté du résultat est ailleurs — le
# pré-agrandissement (Lanczos interpole, un ESRGAN dessin non), le négatif (le
# défaut est orienté photo et fait poser du grain sur les aplats), le CFG et le
# verrouillage de structure. D'où un dictionnaire d'options par préréglage, dont
# tous les champs sont facultatifs.
#
# Champs : prompt · negative · denoise · cfg · steps · controlnet · cn_scale ·
#          esrgan ("drawing" = choisir automatiquement un modèle dessin installé)
UPSCALE_PRESETS = [
    {"name": "🔍 Sharp & faithful (no additions)",
     "prompt": "sharp focus, clean precise detail, faithful to the original, "
               "no added elements, high fidelity",
     "denoise": 0.20},
    {"name": "✨ Add detail",
     "prompt": "highly detailed, intricate fine textures, crisp micro-detail, "
               "enhanced clarity, sharp focus",
     "denoise": 0.40},
    {"name": "🧴 Realistic skin (portrait)",
     "prompt": "highly detailed realistic skin with fine pores, natural "
               "complexion, sharp eyes and individual hair strands, "
               "true-to-life photographic detail",
     "denoise": 0.35},
    {"name": "🌿 Nature / landscape",
     "prompt": "crisp natural textures, detailed foliage and rock, fine "
               "vegetation, clear sharp landscape detail",
     "denoise": 0.40},
    {"name": "🏙️ Architecture / product",
     "prompt": "clean sharp edges, precise material textures, accurate "
               "reflections, crisp surface detail",
     "denoise": 0.30},

    # ---- Dessin : le préréglage qui traite VRAIMENT le problème -------------
    # Sur une illustration, un upscale « photo » fait trois dégâts :
    #  1. la base Lanczos interpole -> traits mous, aplats baveux ;
    #  2. le négatif par défaut ne défend pas les aplats -> SDXL y pose du grain
    #     et de la matière « photo » ;
    #  3. un débruitage à 0,40 redessine le trait, qui se met à onduler.
    # On corrige les trois : ESRGAN dessin en base (agrandissement réel, pas une
    # interpolation), négatif anti-photo/anti-grain, débruitage bas et structure
    # verrouillée par ControlNet — SDXL ne fait plus que nettoyer.
    {"name": "🖍️ Illustration / comics — crisp linework, no interpolation",
     "prompt": "clean crisp linework, flat solid color areas, smooth even "
               "fills, sharp precise edges, clean vector-like illustration, "
               "no texture on flat colors",
     "negative": "photorealistic, photo texture, film grain, noise, gradient "
                 "banding, jpeg artifacts, blurry soft edges, halo, ringing, "
                 "oversharpened, painterly brush texture on flat areas, "
                 "3d render, deformed lines",
     "denoise": 0.18, "cfg": 4.0, "steps": 20,
     "controlnet": True, "cn_scale": 0.85, "esrgan": "drawing"},
    {"name": "🎨 Painted illustration / concept art",
     "prompt": "crisp clean brushwork, refined shapes, vivid consistent "
               "colors, sharp stylized detail",
     "negative": "photorealistic, film grain, noise, jpeg artifacts, "
                 "blurry, oversharpened, halo",
     "denoise": 0.35, "cfg": 5.0,
     "controlnet": True, "cn_scale": 0.7, "esrgan": "drawing"},

    {"name": "🚀 Maximum detail (creative)",
     "prompt": "ultra detailed, hyper-detailed intricate surfaces, rich fine "
               "texture everywhere, razor sharp",
     "denoise": 0.55},
    {"name": "🪶 Soft & clean (anti-grain)",
     "prompt": "clean smooth surfaces, gently denoised, soft natural detail, "
               "no artifacts, no grain",
     "denoise": 0.25},
]


# Titre de l'aperçu de secours, partagé par les outils où l'on CLIQUE sur
# l'image : c'est là que ne pas la voir est bloquant, pas seulement gênant.
_FALLBACK_TITLE = "🖼️ Fallback preview (if the image will not display)"


def _installer_block(title: str, note: str, stream_fn, installed: bool):
    """Installation 1 clic, commune aux outils.

    Un bloc d'installation ne sert QU'UNE FOIS. Le laisser en permanence dans
    l'onglet, c'est faire payer à vie un accordéon de plus à quelqu'un qui a
    déjà tout installé — et ils sont sept. Quand l'outil est là, le bloc
    disparaît ; quand il ne l'est pas, il est déplié d'emblée, parce qu'à ce
    moment-là c'est la seule chose à faire dans cet onglet.

    On garde `visible=` plutôt qu'un `if` : le bloc doit exister dans l'arbre
    pour pouvoir réapparaître si l'installation échoue plus tard.
    """
    # Déjà installé : une seule petite ligne, qui redonne accès au bloc si
    # l'installation est à refaire. Sans elle, « réparer » deviendrait
    # impossible depuis l'interface.
    # `title` est un nom d'outil français fourni par l'appelant : il se traduit
    # séparément, sinon seule l'enveloppe passerait en anglais.
    repair = gr.Button(
        t("⚙️ Reinstall / repair {title}").format(title=t(title)),
        size="sm", variant="secondary", visible=installed)
    with gr.Accordion(
            t("⚙️ Install {title} (1 click)").format(title=t(title)),
            open=not installed, visible=not installed) as box:
        gr.Markdown(note)
        log = gr.Textbox(label="Install log", lines=10,
                         autoscroll=True, elem_classes="log-box")
        btn = gr.Button(t("⬇️ Install {title}").format(title=t(title)))

        def _install():
            for msg in stream_fn():
                yield msg

        btn.click(_install, outputs=[log])

    repair.click(lambda: (gr.update(visible=True, open=True),
                          gr.update(visible=False)),
                 outputs=[box, repair])
    return box


def build_toolkit_tab(tab_id="toolkit", pending_toolkit=None, tabs=None,
                      parent_tabs=None, prompt_boxes=None):
    """`parent_tabs` : le groupe « 🧰 Outils » qui contient cet onglet.

    Depuis le regroupement des onglets, atteindre un outil demande DEUX
    sélections — le groupe à la racine, puis l'outil dedans. L'appelant fournit
    donc le conteneur, sinon l'image arriverait dans un onglet resté masqué."""
    with gr.Tab("🧰 Toolkit", id=tab_id):
        gr.Markdown(
            "### Utility tools\n**Depth** map, **background removal** "
            "(transparent PNG), **click-to-cutout** (Segment Anything) and "
            "**ESRGAN upscale** (simple, 100% GPU).")

        with gr.Tabs() as sub_tabs:
            # ---------- Image -> prompt ----------
            # Placé EN TÊTE : c'est le seul outil du Toolkit qui ne transforme
            # pas l'image mais qui alimente la génération. Il vient donc avant
            # ceux qui la retouchent, dans l'ordre où on s'en sert.
            with gr.Tab("📝 Image → prompt", id="describe"):
                gr.Markdown(
                    "Hand it an image, get back the **prompt** that would "
                    "recreate it. This is not a caption: a vision model would "
                    "say “a photo of a cat on a sofa”, which pasted into the "
                    "Prompt field gives a flat image. Here we name the "
                    "**medium**, the **light**, the **lens**, the **palette** "
                    "and the **framing** — the words that actually steer "
                    "diffusion. Always in **English**: that is the models' "
                    "language.")
                _installer_block(
                    "Image → prompt",
                    "**Qwen2.5-VL-3B** vision-language model (~7.5 GB), same "
                    "family as the prompt enhancer. Loaded then unloaded on "
                    "each call: **no VRAM conflict** with generation. ⚠️ "
                    "*Qwen Research* licence — non-commercial, like the "
                    "enhancer already installed.",
                    tools.install_describe_stream, tools.describe_is_installed())

                with gr.Row():
                    with gr.Column(scale=2):
                        p_image = gr.Image(label="Image to read", type="pil",
                                           height=340,
                                           buttons=widgets.IMAGE_BUTTONS)
                        p_mode = gr.Radio(
                            [(t("📸 Recreate this image — subject AND style"),
                              "full"),
                             (t("🎨 Style only — to apply to something else"),
                              "style"),
                             (t("🔍 Plain description — what is in it"),
                              "plain")],
                            value="full", label="What you want out of it")
                        p_hint = gr.Markdown("", elem_classes="hint")
                        with gr.Row():
                            p_run = gr.Button("📝 Read the image",
                                              variant="primary")
                            p_stop = gr.Button("⏹️ Cancel", variant="stop")
                    with gr.Column(scale=3):
                        p_out = gr.Textbox(
                            label="Resulting prompt", lines=9,
                            buttons=widgets.TEXT_COPY,
                            placeholder="The text will appear here — read it "
                                        "before sending, it is a starting "
                                        "point, not a verdict.")
                        gr.Markdown(t(
                            "**Send it straight to a generation tab** — the "
                            "prompt replaces the field there, and you "
                            "generate whenever you like."))
                        with gr.Row():
                            p_to_krea = gr.Button("→ ⚡ Krea 2 Turbo")
                            p_to_flux = gr.Button("→ 🟣 Flux.2 Klein")
                        p_log = gr.Textbox(label="Log", lines=6,
                                           autoscroll=True,
                                           elem_classes="log-box")

                # Ce que le mode va faire, écrit AVANT le clic : « juste le
                # style » et « refaire cette image » ne se devinent pas d'après
                # leur libellé seul, et se tromper coûte une minute de calcul.
                _MODE_HINT = {
                    "full": "Subject, setting, light, colours, medium — "
                            "enough to recreate a close image on another "
                            "model.",
                    "style": "**Not a word about the subject**: not the cat, "
                             "not the car, not the place. Only the look, to "
                             "paste in front of your own subject.",
                    "plain": "Two or three sentences, no prompt vocabulary. "
                             "To know what is in the image, not to regenerate "
                             "it.",
                }

                def _mode_hint(mode):
                    return t(_MODE_HINT.get(mode, ""))

                p_mode.change(_mode_hint, inputs=[p_mode], outputs=[p_hint])

                def do_describe(image, mode):
                    if image is None:
                        raise gr.Error(t("Load an image first."))
                    if not tools.describe_is_installed():
                        raise gr.Error(t("Install “Image → prompt” first."))
                    q: "queue.Queue[str | None]" = queue.Queue()
                    state: dict = {}

                    def worker():
                        try:
                            state["out"] = tools.image_to_prompt(
                                image, mode=mode, log=q.put)
                        except Exception as exc:  # noqa: BLE001
                            state["err"] = str(exc)
                        finally:
                            q.put(None)

                    threading.Thread(target=worker, daemon=True).start()
                    logs: list[str] = []
                    while True:
                        line = q.get()
                        if line is None:
                            break
                        logs.append(line)
                        yield gr.update(), "\n".join(logs[-500:])
                    if "err" in state:
                        logs.append(f"\n[ERROR] {state['err']}")
                        yield gr.update(), "\n".join(logs[-500:])
                        return
                    yield state["out"][0], "\n".join(logs[-500:])

                p_evt = p_run.click(do_describe, inputs=[p_image, p_mode],
                                    outputs=[p_out, p_log])
                widgets.stop_into_log(p_stop, tools.cancel, p_log, [p_evt])

                # Envoi vers un onglet de génération. On écrit DIRECTEMENT
                # dans son champ Prompt (l'appelant nous l'a passé) puis on
                # bascule. Le détour par un State consommé au changement
                # d'onglet ne marche pas : une sélection programmatique ne
                # déclenche pas `Tabs.select`, le texte n'arrivait jamais.
                def _send_to(model_id, button):
                    box = (prompt_boxes or {}).get(model_id)
                    if box is None or tabs is None:
                        button.visible = False
                        return

                    def _go(text):
                        if not (text or "").strip():
                            raise gr.Error(t("Read an image first."))
                        return text.strip(), gr.Tabs(selected=model_id)

                    button.click(_go, inputs=[p_out], outputs=[box, tabs])

                _send_to("krea2-turbo", p_to_krea)
                _send_to("flux2-klein-9b", p_to_flux)

            # ---------- Profondeur ----------
            with gr.Tab("🌐 Depth", id="depth"):
                gr.Markdown(
                    "*Depth Anything V2* — depth map (light = near, dark = "
                    "far). Download the result to reuse it.")
                _installer_block(
                    "Depth Anything V2",
                    "Uses PyTorch + transformers (~100 MB model). No command "
                    "to type.",
                    tools.install_depth_stream, tools.depth_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        d_image = gr.Image(label="Source image", type="pil",
                                           buttons=widgets.IMAGE_VIEW_ONLY)
                        d_run = gr.Button("🌐 Generate depth",
                                          variant="primary", size="lg")
                    with gr.Column(scale=4):
                        d_result = gr.Image(label="Depth map", height=520,
                                            format="png",
                                            buttons=widgets.IMAGE_BUTTONS)
                        d_log = gr.Textbox(label="Log", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_depth(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    logs: list[str] = []
                    progress(0.1, desc="Depth…")
                    try:
                        out = tools.depth_map(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    return str(out), "\n".join(logs)

                d_run.click(do_depth, inputs=[d_image], outputs=[d_result, d_log])

            # ---------- Suppression d'arrière-plan ----------
            with gr.Tab("✂️ Background removal", id="bg"):
                gr.Markdown(
                    "*RMBG-1.4* — cuts out the subject and returns a "
                    "**transparent PNG**.  \n⚠️ **Non-commercial** license "
                    "(BRIA RMBG-1.4).")
                _installer_block(
                    "RMBG-1.4",
                    "Uses PyTorch + transformers (~176 MB model). No command "
                    "to type.",
                    tools.install_bg_stream, tools.bg_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        b_image = gr.Image(label="Source image", type="pil",
                                           buttons=widgets.IMAGE_VIEW_ONLY)
                        b_run = gr.Button("✂️ Cut out", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        b_result = gr.Image(label="Cutout subject (transparent PNG)",
                                            height=520, format="png",
                                            buttons=widgets.IMAGE_BUTTONS,
                                            image_mode="RGBA")
                        b_log = gr.Textbox(label="Log", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_bg(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    logs: list[str] = []
                    progress(0.1, desc="Cutting out…")
                    try:
                        out = tools.bg_remove(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    return str(out), "\n".join(logs)

                b_run.click(do_bg, inputs=[b_image], outputs=[b_result, b_log])

            # ---------- Segment Anything (clic) ----------
            with gr.Tab("🪄 Cut out (SAM)", id="sam"):
                gr.Markdown(
                    "*Segment Anything* — **click an object**: SAM instantly "
                    "shows the **selected area highlighted**. Re-click to "
                    "adjust, then “Extract” for the **transparent PNG**.")
                _installer_block(
                    "Segment Anything",
                    "PyTorch + transformers (~375 MB, facebook/sam-vit-base). "
                    "No command to type.",
                    tools.install_sam_stream, tools.sam_is_installed())

                s_cut = gr.State(None)     # chemin du découpage déjà calculé
                with gr.Row():
                    with gr.Column(scale=3):
                        s_image = gr.Image(label="Image — click the object",
                                           type="pil",
                                           buttons=widgets.IMAGE_VIEW_ONLY)
                        s_info = gr.Markdown("Click a point on the image.")
                        s_overlay = gr.Image(label="Selected area (preview)",
                                             height=300, interactive=False,
                                             buttons=widgets.IMAGE_VIEW_ONLY)
                        s_run = gr.Button("🪄 Extract object", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        s_result = gr.Image(label="Extracted object (transparent PNG)",
                                            height=520, format="png",
                                            buttons=widgets.IMAGE_BUTTONS,
                                            image_mode="RGBA")
                        s_log = gr.Textbox(label="Log", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def _on_click(img, evt: gr.SelectData, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not tools.sam_is_installed():
                        raise gr.Error(t("Segment Anything is not installed "
                                         "(the “Install” button above)."))
                    x, y = int(evt.index[0]), int(evt.index[1])
                    progress(0.2, desc="Segmentation…")
                    try:
                        cut, overlay = tools.sam_segment(img, x, y)
                    except Exception as exc:  # noqa: BLE001
                        raise gr.Error(str(exc))
                    progress(1.0, desc="Done")
                    return (overlay or gr.update(), str(cut),
                            t("Area selected at ({x}, {y}). Click “Extract” "
                              "or re-click elsewhere.").format(x=x, y=y))

                s_image.select(_on_click, inputs=[s_image],
                               outputs=[s_overlay, s_cut, s_info])

                # Aperçu de secours : si la vignette du composant reste une
                # icône cassée, l'image est quand même visible ici — les
                # pixels sont dans la page, aucune requête n'est faite (voir
                # atelier/ui/preview.py). Replié : doublon inutile quand
                # l'aperçu normal fonctionne.
                with gr.Accordion(_FALLBACK_TITLE, open=False):
                    s_fallback = gr.HTML(preview.html(None))
                s_image.change(preview.html, inputs=[s_image],
                               outputs=[s_fallback])

                def do_sam(cut):
                    if not cut:
                        raise gr.Error(t("Click an object in the image first."))
                    return str(cut), f"✅ {cut}"

                s_run.click(do_sam, inputs=[s_cut], outputs=[s_result, s_log])

            # ---------- Agrandir (ESRGAN, sd.cpp) ----------
            with gr.Tab("🔼 Upscale", id="esrgan"):
                gr.Markdown(
                    "**Plain** enlargement by an ESRGAN network, native to "
                    "**sd.cpp**: deterministic, **100% GPU**, no PyTorch and "
                    "no prompt. The factor (×2 or ×4) comes from the model "
                    "you pick; “Repeat” applies the model a second time (×2 "
                    "twice = ×4).\n\n🎨 **Comics, illustration, line art**: "
                    "pick a model marked **drawing / anime**. Photo models "
                    "(Remacri, Nomos, UltraSharp…) are trained on natural "
                    "textures: on a flat colour area they invent grain, and "
                    "along a crisp line they lay down a halo. That is what "
                    "“horrible interpolation” looks like.\n\n📥 **Adding your "
                    "own models**: drop a `.pth`, `.safetensors` or `.gguf` "
                    "file into the upscalers folder, then “↻ Refresh”. sd.cpp "
                    "reads most `.pth` files directly — so the whole "
                    "[OpenModelDB](https://openmodeldb.info) catalog is "
                    "usable; filter it on *anime* / *manga* / *cartoon*. GGUF "
                    "loads faster and avoids executing a pickle, but it is "
                    "not required.")

                with gr.Accordion("⬇️ Download the upscalers (1 click)",
                                  open=not registry.upscalers_ready()):
                    gr.Markdown(
                        "Fetches **all** the GGUF ESRGAN models (~1 GB total) "
                        "from `wbruna/upscalers-sdcpp-gguf`. Reusable offline "
                        "afterwards.")
                    u_inst_log = gr.Textbox(label="Download log",
                                            lines=8, autoscroll=True,
                                            elem_classes="log-box")
                    u_inst = gr.Button("⬇️ Download the upscalers")

                with gr.Row():
                    with gr.Column(scale=3):
                        u_image = gr.Image(
                            label="Image to upscale", type="pil",
                            buttons=widgets.IMAGE_VIEW_ONLY)
                        u_model = gr.Dropdown(
                            registry.upscaler_choices(),
                            value=registry.default_upscaler(),
                            label="Upscale model (×2 / ×4 by name)",
                            info="🎨 = trained for DRAWINGS (crisp linework, "
                                 "clean flat colours) · 📷 = photo. On a "
                                 "comics page, a photo model smears and lays "
                                 "down halos.")
                        u_repeats = gr.Radio(
                            [("×1 (natif)", 1), ("Repeat ×2", 2)],
                            value=1, label="Repeat",
                            info="⚠️ Repeating runs the network on its OWN "
                                 "output: it mistakes the high frequencies it "
                                 "just invented for real detail and "
                                 "re-emphasizes them. That is what produces "
                                 "the staircase on diagonals. A ×4 model "
                                 "always beats a repeated ×2.")
                        with gr.Row():
                            u_refresh = gr.Button("↻ Refresh list", size="sm")
                            u_run = gr.Button("🔼 Upscale", variant="primary",
                                              size="lg", scale=2)
                        u_stop = gr.Button("⏹️ Cancel", variant="stop", size="sm")
                    with gr.Column(scale=4):
                        u_result = gr.Image(
                            label="Result (full resolution in outputs/)",
                            height=520, format="png",
                            buttons=widgets.IMAGE_BUTTONS)
                        u_to_face = gr.Button("→ 🙂 Fix the faces",
                                              size="sm")
                        u_log = gr.Textbox(label="Log", lines=10,
                                           autoscroll=True, elem_classes="log-box")

                def _install_upscalers():
                    lines: list[str] = []
                    for msg in downloader.download_upscalers(log=lines.append):
                        lines.append(msg)
                        yield "\n".join(lines), gr.update()
                    yield ("\n".join(lines),
                           gr.update(choices=registry.upscaler_choices(),
                                     value=registry.default_upscaler()))

                u_inst.click(_install_upscalers, outputs=[u_inst_log, u_model])

                def _refresh_upscalers():
                    return gr.update(choices=registry.upscaler_choices(),
                                     value=registry.default_upscaler())

                u_refresh.click(_refresh_upscalers, outputs=[u_model])

                def do_upscale(img, model, repeats, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not model:
                        raise gr.Error(t("Choose an upscale model (download "
                                         "them first)."))
                    logs: list[str] = []
                    progress(0.1, desc="Agrandissement…")
                    try:
                        out = gen_engine.upscale_image(
                            img, model, repeats=int(repeats), log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    logs.append(f"\n✅ Image agrandie : {out}")
                    return str(out), "\n".join(logs)

                u_evt = u_run.click(do_upscale,
                                    inputs=[u_image, u_model, u_repeats],
                                    outputs=[u_result, u_log])
                widgets.stop_into_log(u_stop, gen_engine.cancel, u_log,
                                      [u_evt])

            # ---------- Décomposition en calques (PSD) ----------------------
            with gr.Tab("🧩 Layers", id="layers"):
                gr.Markdown(
                    "Cuts an image into **layers** and writes a **PSD** (or "
                    "separate transparent PNGs). Two ways to go about it: let "
                    "SAM sweep the image on its own, or **point at the "
                    "regions yourself** by clicking.\n\n⚠️ **Know this before "
                    "you click**: the layers are **flat cut-outs**. Moving an "
                    "object reveals a hole — the background behind it never "
                    "existed. This is meant for masking, retouching a region "
                    "or exporting an element, **not** for recomposing the "
                    "scene.\n\nThe stacking order comes from the **depth** "
                    "map when the Depth tool is installed; otherwise large "
                    "regions go to the back, which is only an "
                    "approximation.\n\nRegions are **cleaned up** before "
                    "being laid down: interior holes filled, scattered pieces "
                    "split into distinct regions, crumbs discarded, edges "
                    "softened. And the layers are **disjoint** — showing them "
                    "all reproduces the original image exactly, no pixel is "
                    "painted twice.")
                _installer_block(
                    "Segment Anything",
                    "Same add-on as “Cut out an object”. Required.",
                    tools.install_sam_stream, tools.sam_is_installed())
                _installer_block(
                    "CLIP (zone understanding)",
                    "**Optional, ~600 MB — but it is what brings the "
                    "intelligence.** Without CLIP the tool sees shapes only. "
                    "With it, it recognizes what it is cutting out and uses "
                    "that for three things:\n\n- **regrouping the pieces of "
                    "one object** — SAM returns “body”, “door” and “wheel” "
                    "separately; labelled “vehicle” and adjacent, they become "
                    "**a single layer** again;\n- **discarding what is "
                    "nothing** — a flat area, a patch of blur, a meaningless "
                    "fragment. A classifier cannot say “nothing”, so the "
                    "vocabulary carries junk categories whose job is to "
                    "absorb them;\n- **naming the layers**: “vehicle”, “sky”, "
                    "“person” instead of “foreground · centre · orange”. And "
                    "without the Depth tool, the stacking order is inferred "
                    "from meaning — the sky goes behind because it is the "
                    "sky, not because it is large.",
                    tools.install_clip_stream, tools.clip_is_installed())

                lay_masks = gr.State([])      # masques choisis à la main
                with gr.Row():
                    with gr.Column(scale=3):
                        lay_image = gr.Image(label="Image to decompose",
                                             type="pil",
                                             buttons=widgets.IMAGE_VIEW_ONLY)
                        lay_mode = gr.Radio(
                            [(t("Automatic — SAM sweeps the image"), "auto"),
                             (t("Manual — I click the areas"), "manual")],
                            value="auto", label="Mode")
                        with gr.Group(visible=False) as lay_manual_box:
                            lay_hint = gr.Markdown(
                                "**Click an object** in the image above: it "
                                "becomes a layer. Click elsewhere to add "
                                "more.", elem_classes="hint")
                            lay_list = gr.Markdown("*No area selected.*",
                                                   elem_classes="feedback")
                            with gr.Row():
                                lay_undo = gr.Button("↩️ Remove the last one",
                                                     size="sm")
                                lay_clear = gr.Button("🗑️ Clear all",
                                                      size="sm")
                        with gr.Group(visible=True) as lay_auto_box:
                            lay_points = gr.Slider(
                                6, 24, value=12, step=2,
                                label="Sweep density (points per side)",
                                info="↑ = more areas found, and much slower. "
                                     "12 is a good starting point.")
                            lay_minarea = gr.Slider(
                                0.1, 5.0, value=0.4, step=0.1,
                                label="Minimum layer area (% of the image)",
                                info="Raising this is the best way to avoid a "
                                     "soup of tiny layers.")
                            lay_max = gr.Slider(
                                4, 40, value=24, step=1,
                                label="Maximum number of layers")
                        with gr.Row():
                            lay_psd = gr.Checkbox(value=True, label="PSD file")
                            lay_png = gr.Checkbox(
                                value=False, label="Separate transparent PNGs")
                        with gr.Row(elem_classes="go-row"):
                            lay_run = gr.Button("🧩 Decompose", variant="primary",
                                                size="lg", scale=3)
                            lay_stop = gr.Button("⏹️ Cancel", variant="stop",
                                                 scale=1, min_width=90)
                    with gr.Column(scale=4):
                        lay_preview = gr.Image(
                            label="Selected areas (preview)", height=460,
                            interactive=False, format="png",
                            buttons=widgets.IMAGE_VIEW_ONLY)
                        lay_files = gr.File(label="Files produced",
                                            file_count="multiple")
                        lay_log = gr.Textbox(label="Log", lines=10,
                                             autoscroll=True,
                                             elem_classes="log-box")

                def _lay_mode(mode):
                    manual = (mode == "manual")
                    return (gr.update(visible=manual),
                            gr.update(visible=not manual))

                lay_mode.change(_lay_mode, inputs=[lay_mode],
                                outputs=[lay_manual_box, lay_auto_box])

                # Même aperçu de secours qu'en détourage : c'est ici qu'il
                # manque le plus, le mode manuel consistant à cliquer sur ce
                # qu'on ne voit pas.
                with gr.Accordion(_FALLBACK_TITLE, open=False):
                    lay_fallback = gr.HTML(preview.html(None))
                lay_image.change(preview.html, inputs=[lay_image],
                                 outputs=[lay_fallback])

                def _lay_overlay(img, masks):
                    """Teinte les zones choisies, pour voir ce qu'on a."""
                    import numpy as np
                    if img is None:
                        return None
                    arr = np.asarray(img.convert("RGB")).astype("float32")
                    palette = [(0, 200, 255), (255, 120, 90), (140, 230, 120),
                               (240, 200, 80), (200, 140, 255), (255, 150, 200)]
                    for i, m in enumerate(masks):
                        tint = np.array(palette[i % len(palette)], "float32")
                        sel = m[..., None]
                        arr = np.where(sel, arr * 0.55 + tint * 0.45, arr)
                    from PIL import Image as _PI
                    return _PI.fromarray(arr.clip(0, 255).astype("uint8"))

                def _lay_summary(masks):
                    if not masks:
                        return "*No area selected.*"
                    return (f"**{len(masks)} region(s) picked** — click "
                            "again to add more, then “Split”.")

                def _lay_click(img, mode, masks, evt: gr.SelectData):
                    if mode != "manual" or img is None:
                        return gr.update(), gr.update(), gr.update()
                    if not tools.sam_is_installed():
                        raise gr.Error(t("Segment Anything is not installed "
                                         "(the “Install” button above)."))
                    import numpy as np
                    from PIL import Image as _PI
                    x, y = int(evt.index[0]), int(evt.index[1])
                    try:
                        cut, _ = tools.sam_segment(img, x, y)
                    except Exception as exc:  # noqa: BLE001
                        raise gr.Error(str(exc))
                    alpha = np.asarray(_PI.open(cut).convert("RGBA"))[:, :, 3]
                    new = list(masks) + [alpha > 127]
                    return new, _lay_overlay(img, new), _lay_summary(new)

                lay_image.select(_lay_click,
                                 inputs=[lay_image, lay_mode, lay_masks],
                                 outputs=[lay_masks, lay_preview, lay_list])

                def _lay_undo(img, masks):
                    new = list(masks)[:-1]
                    return new, _lay_overlay(img, new), _lay_summary(new)

                lay_undo.click(_lay_undo, inputs=[lay_image, lay_masks],
                               outputs=[lay_masks, lay_preview, lay_list])
                lay_clear.click(lambda img: ([], _lay_overlay(img, []),
                                             _lay_summary([])),
                                inputs=[lay_image],
                                outputs=[lay_masks, lay_preview, lay_list])

                def do_layers(img, mode, masks, points, minarea, maxn,
                              want_psd, want_png, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not (want_psd or want_png):
                        raise gr.Error(t("Choose at least one output format "
                                         "(PSD or PNG)."))
                    logs: list[str] = []
                    progress(0.1, desc="Splitting…")
                    try:
                        if mode == "manual":
                            out = tools.masks_to_layers(
                                img, list(masks), want_psd=want_psd,
                                want_png=want_png, log=logs.append)
                        else:
                            out = tools.image_to_layers(
                                img, points_per_side=int(points),
                                max_layers=int(maxn),
                                min_area_pct=float(minarea),
                                want_psd=want_psd, want_png=want_png,
                                log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return gr.update(), "\n".join(logs)
                    progress(1.0, desc="Done")
                    logs.append("\n✅ " + " · ".join(str(p.name) for p in out))
                    # Un dossier ne se télécharge pas : on ne propose que les
                    # fichiers, et le journal donne le chemin du dossier PNG.
                    files = [str(p) for p in out if p.is_file()]
                    return (files or gr.update()), "\n".join(logs)

                lay_evt = lay_run.click(
                    do_layers,
                    inputs=[lay_image, lay_mode, lay_masks, lay_points,
                            lay_minarea, lay_max, lay_psd, lay_png],
                    outputs=[lay_files, lay_log])
                widgets.stop_into_log(lay_stop, tools.cancel, lay_log,
                                      [lay_evt])

            # ---------- HD natif sd.cpp (highres fix) -----------------------
            with gr.Tab("🚀 HD", id="hd"):
                gr.Markdown(
                    "**Native sd.cpp HD pass**: the image is enlarged, then "
                    "**re-denoised as a whole** by your generation model "
                    "(Krea 2, Flux.2). All in **a single command**, 100% GPU, "
                    "no PyTorch.\n\nTwo fundamental differences from the "
                    "creative SDXL upscale:\n- **no tiling at all** — the "
                    "second pass sees the whole image, so there is no seam "
                    "*possible*, and no mismatch between neighbouring "
                    "squares;\n- **your model** does the redrawing, not a "
                    "2023 SDXL: the added detail stays in the style the model "
                    "already knows.\n\nIn exchange, **it is VRAM-hungry**: "
                    "refusing to tile has a price. The second pass allocates "
                    "a buffer proportional to the pixel count, **on top of "
                    "the model weights** already on the card. The factor is "
                    "therefore budgeted from your VRAM and your model's size, "
                    "and lowered on its own if it does not fit — the log "
                    "states the value it settled on. On 11–12 GB with a Q5 "
                    "model, expect ×1.25 to ×1.5; a lighter quantization buys "
                    "factor.")

                _hd_models = [(m.name, m.id)
                              for m in registry.load_base_models(
                                  settings.load_prefs())
                              if registry.model_is_ready(m)]
                if not _hd_models:
                    gr.Markdown(
                        "> ⚠️ **No generation model installed.** Download "
                        "Krea 2 Turbo or Flux.2 Klein from the “Model "
                        "catalog” tab, then come back here.")

                with gr.Row():
                    with gr.Column(scale=3):
                        hd_image = gr.Image(label="Image to take to HD",
                                            type="pil",
                                            buttons=widgets.IMAGE_VIEW_ONLY)
                        hd_model = gr.Dropdown(
                            choices=_hd_models,
                            value=(_hd_models[0][1] if _hd_models else None),
                            label="Generation model")
                        hd_scale = gr.Slider(
                            1.25, 3.0, value=2.0, step=0.25,
                            label="Enlargement factor",
                            info="The final side is capped: beyond it the "
                                 "factor is reduced automatically and the log "
                                 "states the value it settled on.")
                        hd_denoise = gr.Slider(
                            0.15, 0.70, value=0.40, step=0.05,
                            label="Added detail (HD pass denoise)",
                            info="The ONLY setting that really matters. 0.2 = "
                                 "stays very close to the original; 0.5+ = "
                                 "the model frankly reinvents the material.")
                        _hd_ups = registry.list_upscalers()
                        hd_upscaler = gr.Dropdown(
                            choices=[("Latent (default — the gentlest)", "Latent"),
                                     ("Latent antialiased", "Latent (antialiased)"),
                                     ("Lanczos (image, neutral)", "Lanczos")]
                                    + [(f"ESRGAN — {u}", u) for u in _hd_ups],
                            value="Latent",
                            label="Intermediate enlargement",
                            info="What enlarges BEFORE the second denoise. "
                                 "“Latent” works in the model's own space and "
                                 "lets the denoise rebuild everything; an "
                                 "ESRGAN gives an already-crisp base (useful "
                                 "on line art), at the risk of freezing its "
                                 "own flaws.")
                        hd_prompt = gr.Textbox(
                            label="Description (optional)", lines=2,
                            placeholder="what the image shows, in a few words",
                            info="Guides the added detail. Empty works very "
                                 "well: the model starts from the image.")
                        hd_seed = gr.Number(value=-1, precision=0,
                                            label="Seed (-1 = random)")
                        with gr.Row():
                            hd_run = gr.Button("🚀 Take to HD",
                                               variant="primary", size="lg",
                                               scale=2)
                            hd_stop = gr.Button("⏹️ Cancel", variant="stop",
                                                size="sm")
                    with gr.Column(scale=4):
                        hd_result = gr.Image(
                            label="Live preview (full resolution in outputs/)", height=520, format="png",
                            buttons=widgets.IMAGE_BUTTONS)
                        hd_log = gr.Textbox(label="Log", lines=12,
                                            autoscroll=True,
                                            elem_classes="log-box")

                def do_hd(img, model_id, scale, denoise, upscaler, prompt,
                          seed_v, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not model_id:
                        raise gr.Error(t("No generation model installed: "
                                         "download one from the “Model "
                                         "catalog” tab."))
                    logs: list[str] = []
                    settings.ensure_dirs()
                    preview = settings.TMP_DIR / "hd_preview.png"
                    try:
                        seed_i = int(seed_v)
                    except (TypeError, ValueError):
                        seed_i = -1
                    progress(0.1, desc="HD pass…")
                    try:
                        outs = gen_engine.hd_upscale(
                            model_id, img, scale=float(scale),
                            upscaler=upscaler, denoise=float(denoise),
                            prompt=prompt or "", seed=seed_i,
                            preview_path=preview, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    if not outs:
                        logs.append("\n[ERROR] no image produced.")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    logs.append(f"\n✅ Image HD : {outs[0]}")
                    return str(outs[0]), "\n".join(logs)

                hd_evt = hd_run.click(
                    do_hd,
                    inputs=[hd_image, hd_model, hd_scale, hd_denoise,
                            hd_upscaler, hd_prompt, hd_seed],
                    outputs=[hd_result, hd_log])
                widgets.stop_into_log(hd_stop, gen_engine.cancel, hd_log,
                                      [hd_evt])

            # ---------- Haute résolution (Flux.2 en référence + départ) ------
            with gr.Tab("🔍 High resolution", id="highres"):
                gr.Markdown(
                    "The image is **run back through Flux.2 at its native "
                    "resolution**, serving as both the **reference** and the "
                    "**starting point**. Three details make all the "
                    "difference, and none of them is obvious:\n- we ask for "
                    "“**high resolution**”, not “upscale”: in the training "
                    "captions, *upscaled* labels images that really were "
                    "upscaled — and therefore carry the very artifacts we "
                    "want to avoid;\n- the pre-enlargement is **bilinear**, "
                    "deliberately bland: Lanczos adds ringing that the model "
                    "reads back as detail and amplifies;\n- the same image "
                    "serves as the **reference** (the content) **and** as the "
                    "**starting latent** (the structure).\n\n⚠️ **This is not "
                    "a restoration.** At a high denoise the model REDRAWS: "
                    "what is preserved is plausibility, not fidelity. To keep "
                    "a face the same person, go through **🌱 Restore** "
                    "(SeedVR2) instead.")

                _hr_models = highres.edit_models()
                if not _hr_models:
                    gr.Markdown(
                        "> ⚠️ **No editing model installed.** This method "
                        "needs a model that accepts a reference image (Flux.2 "
                        "Klein). Download one from the “Model catalog” tab.")

                with gr.Row():
                    with gr.Column(scale=3):
                        hr_image = gr.Image(label="Image to upscale", type="pil",
                                            buttons=widgets.IMAGE_VIEW_ONLY)
                        hr_model = gr.Dropdown(
                            choices=_hr_models,
                            value=(_hr_models[0][1] if _hr_models else None),
                            label="Editing model")
                        hr_factor = gr.Slider(
                            1.0, 3.0, value=2.0, step=0.25,
                            label="Enlargement factor",
                            info="Output never goes below 1 MP (Flux.2's own "
                                 "regime) and never past 3.7 MP — beyond that "
                                 "the model loses global coherence. Reduced "
                                 "by itself if the card refuses.")
                        hr_strength = gr.Slider(
                            0.4, 0.95, value=0.8, step=0.05,
                            label="Denoise",
                            info="0.7–0.9 is the method's range. Lower moves "
                                 "the image less but gains less detail; "
                                 "higher makes it a different image.")
                        hr_prompt = gr.Textbox(
                            value=highres.DEFAULT_PROMPT,
                            label="Prompt",
                            info="The words “high resolution” ARE the method. "
                                 "Add a description if the image deserves "
                                 "one.")
                        hr_colors = gr.Checkbox(
                            value=True,
                            label="Give the original its colours back",
                            info="At high denoise the model washes the image "
                                 "out. We put the starting colours back "
                                 "underneath the detail it just added.")
                        with gr.Row():
                            hr_run = gr.Button("🔍 Go high resolution",
                                               variant="primary", size="lg",
                                               scale=2)
                            hr_stop = gr.Button("⏹️ Cancel", variant="stop",
                                                size="sm")
                    with gr.Column(scale=4):
                        hr_result = gr.Image(label="Result", height=520,
                                             format="png",
                                             buttons=widgets.IMAGE_BUTTONS)
                        hr_log = gr.Textbox(label="Log", lines=12,
                                            autoscroll=True,
                                            elem_classes="log-box")

                def do_highres(img, model_id, factor, strength, prompt,
                               colors, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not model_id:
                        raise gr.Error(t("No editing model installed."))
                    logs: list[str] = []
                    progress(0.1, desc="High resolution…")
                    try:
                        out = highres.high_resolution(
                            img, model_id=model_id, factor=float(factor),
                            strength=float(strength), prompt=prompt,
                            match_colors=bool(colors), log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    logs.append(f"\n✅ Image : {out}")
                    return str(out), "\n".join(logs)

                hr_evt = hr_run.click(
                    do_highres,
                    inputs=[hr_image, hr_model, hr_factor, hr_strength,
                            hr_prompt, hr_colors],
                    outputs=[hr_result, hr_log])
                widgets.stop_into_log(hr_stop, gen_engine.cancel, hr_log,
                                      [hr_evt])

            # ---------- Restauration SeedVR2 (diffusion 1 étape) ------------
            with gr.Tab("🌱 Restore", id="seedvr2"):
                gr.Markdown(
                    "**SeedVR2** diffusion restoration: recovers more natural "
                    "detail than ESRGAN while staying more faithful than the "
                    "creative SDXL upscale. Compute stays on the RTX 3060; "
                    "the GTX 1080 Ti can hold the weights. The 3B is enough "
                    "most of the time; the 7B keeps fine textures (faces, "
                    "fabric) better but takes twice as long.")
                _installer_block(
                    "SeedVR2",
                    "An isolated install (Python 3.12 + PyTorch CUDA): it "
                    "does not touch the application's own dependencies. The "
                    "Q8/Q4 weights are downloaded on the first upscale.",
                    tools.install_seedvr2_stream, tools.seedvr2_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        seed_image = gr.Image(
                            label="Image to restore", type="pil",
                            buttons=widgets.IMAGE_VIEW_ONLY)
                        seed_model = gr.Radio(
                            [(t(label), value)
                             for label, value in tools.SEEDVR2_MODELS],
                            value=tools.SEEDVR2_MODELS[0][1], label="Model",
                            info="Weights download themselves on first use "
                                 "(4.8 GB for a 7B).")
                        seed_res = gr.Slider(
                            1024, 4096, value=2048, step=64,
                            label="Target resolution (short side)",
                            info="Start at 2048 px; 4K takes considerably longer.")
                        seed_offload = gr.Radio(
                            [("GTX 1080 Ti (recommended for this PC)", "secondary"),
                             ("System RAM (more compatible)", "cpu"),
                             ("No offload (fastest, risk of OOM)", "none")],
                            value=("secondary" if hardware.rtx3060_1080ti_combo()
                                   else "cpu"), label="Where the weights live")
                        seed_blocks = gr.Slider(
                            0, 36, value=16, step=1, label="Blocks to offload",
                            info="16 is right with 12 GB; try 24 then 36 if you hit OOM.")
                        with gr.Row():
                            seed_tile = gr.Slider(512, 1280, value=1024, step=64,
                                                  label="Tuile VAE")
                            seed_overlap = gr.Slider(64, 256, value=128, step=32,
                                                     label="Recouvrement")
                        seed_color = gr.Dropdown(
                            [("Wavelet — natural (recommended)", "wavelet"),
                             ("LAB — very faithful colours", "lab"),
                             ("Wavelet adaptatif", "wavelet_adaptive"),
                             ("No correction", "none")],
                            value="wavelet", label="Colour correction")
                        with gr.Row():
                            seed_run = gr.Button("🌱 Restore", variant="primary",
                                                 size="lg", scale=2)
                            seed_stop = gr.Button("⏹️ Cancel", variant="stop",
                                                  size="sm")
                    with gr.Column(scale=4):
                        seed_result = gr.Image(
                            label="SeedVR2 result", height=520, format="png",
                            buttons=widgets.IMAGE_BUTTONS)
                        seed_to_face = gr.Button("→ 🙂 Fix the faces",
                                                 size="sm")
                        seed_log = gr.Textbox(label="Log", lines=14,
                                              autoscroll=True,
                                              elem_classes="log-box")

                def do_seedvr2(img, resolution, model, blocks, tile, overlap,
                               offload, color, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not tools.seedvr2_is_installed():
                        raise gr.Error(t("Install SeedVR2 first."))
                    q: "queue.Queue[str | None]" = queue.Queue()
                    state: dict = {}

                    def worker():
                        try:
                            state["out"] = tools.seedvr2_upscale(
                                img, resolution=int(resolution), model=model,
                                blocks_to_swap=int(blocks), tile=int(tile),
                                overlap=int(overlap), offload=offload,
                                color_correction=color, log=q.put)
                        except Exception as exc:  # noqa: BLE001
                            state["err"] = str(exc)
                        finally:
                            q.put(None)

                    threading.Thread(target=worker, daemon=True).start()
                    logs: list[str] = []
                    progress(0.05, desc="Loading SeedVR2…")
                    while True:
                        line = q.get()
                        if line is None:
                            break
                        logs.append(line)
                        yield gr.update(), "\n".join(logs[-500:])
                    if "err" in state:
                        logs.append(f"\n[ERROR] {state['err']}")
                        yield gr.update(), "\n".join(logs[-500:])
                        return
                    progress(1.0, desc="Done")
                    out = state.get("out")
                    logs.append(f"\n✅ Image restored: {out}")
                    yield str(out), "\n".join(logs[-500:])

                seed_evt = seed_run.click(
                    do_seedvr2,
                    inputs=[seed_image, seed_res, seed_model, seed_blocks,
                            seed_tile, seed_overlap, seed_offload, seed_color],
                    outputs=[seed_result, seed_log])
                widgets.stop_into_log(seed_stop, tools.cancel, seed_log,
                                      [seed_evt])

                with gr.Accordion("📁 Restore a whole folder at once", open=False):
                    gr.Markdown(
                        "Pick a folder of images. SeedVR2 loads the model "
                        "**once**, keeps it cached and processes every file "
                        "without touching the originals. Results go to a "
                        "timestamped subfolder of `outputs/`.")
                    seed_batch_files = gr.File(
                        label="Image folder", file_count="directory",
                        file_types=["image"], type="filepath")
                    with gr.Row():
                        seed_batch_run = gr.Button(
                            "🌱 Restore the whole folder", variant="primary")
                        seed_batch_stop = gr.Button("⏹️ Cancel", variant="stop")
                    seed_batch_gallery = gr.Gallery(
                        label="Batch results", columns=4, height=420,
                        buttons=widgets.GALLERY_BUTTONS)
                    seed_batch_log = gr.Textbox(
                        label="Batch log", lines=12, autoscroll=True,
                        elem_classes="log-box")

                    def do_seedvr2_batch(files, resolution, model, blocks,
                                         tile, overlap, offload, color):
                        if not files:
                            raise gr.Error(t("Pick a folder of images."))
                        if not tools.seedvr2_is_installed():
                            raise gr.Error(t("Install SeedVR2 first."))
                        q: "queue.Queue[str | None]" = queue.Queue()
                        state: dict = {}

                        def worker():
                            try:
                                state["outs"] = tools.seedvr2_batch(
                                    files, resolution=int(resolution), model=model,
                                    blocks_to_swap=int(blocks), tile=int(tile),
                                    overlap=int(overlap), offload=offload,
                                    color_correction=color, log=q.put)
                            except Exception as exc:  # noqa: BLE001
                                state["err"] = str(exc)
                            finally:
                                q.put(None)

                        threading.Thread(target=worker, daemon=True).start()
                        logs: list[str] = []
                        while True:
                            line = q.get()
                            if line is None:
                                break
                            logs.append(line)
                            yield gr.update(), "\n".join(logs[-500:])
                        if "err" in state:
                            logs.append(f"\n[ERROR] {state['err']}")
                            yield gr.update(), "\n".join(logs[-500:])
                            return
                        outs = [str(p) for p in state.get("outs", [])]
                        logs.append(f"\n✅ {len(outs)} image(s) restored.")
                        yield outs, "\n".join(logs[-500:])

                    seed_batch_evt = seed_batch_run.click(
                        do_seedvr2_batch,
                        inputs=[seed_batch_files, seed_res, seed_model,
                                seed_blocks, seed_tile, seed_overlap,
                                seed_offload, seed_color],
                        outputs=[seed_batch_gallery, seed_batch_log])
                    widgets.stop_into_log(seed_batch_stop, tools.cancel,
                                          seed_batch_log, [seed_batch_evt])

            # ---------- Restauration des visages (CodeFormer) ----------------
            with gr.Tab("🙂 Faces", id="face"):
                gr.Markdown(
                    "Rebuilds **faces only**; the rest of the image is left "
                    "alone. It is the step missing after an enlargement: "
                    "neither ESRGAN nor SeedVR2 can rebuild clean eyes and a "
                    "clean mouth on a face that has gone small or blurry. "
                    "**Run it last**, after the upscale.  \nThree models to "
                    "choose from — they do not win on the same images, so "
                    "**compare on yours**. ⚠️ **If you sell your images, "
                    "avoid CodeFormer**: its S-Lab 1.0 licence forbids "
                    "commercial use. The other two are Apache-2.0.")
                _installer_block(
                    "face restoration",
                    "Five weight files (~1.5 GB total): three restorers, the "
                    "face detector and the segmentation used to blend the "
                    "face back in. They are installed together because none "
                    "wins on every image — you compare. Nothing to type.",
                    tools.install_face_stream, tools.face_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        f_image = gr.Image(label="Source image", type="pil",
                                           buttons=widgets.IMAGE_VIEW_ONLY)
                        _f_models = tools.face_models_installed() or list(
                            tools.FACE_MODELS)
                        f_model = gr.Radio(
                            [(f"{label}  ·  {licence}", filename)
                             for filename, label, licence in _f_models],
                            value=next((f for f, _, _ in _f_models
                                        if f == tools.FACE_DEFAULT),
                                       _f_models[0][0]),
                            label="Model",
                            info="The licence decides what you may do with "
                                 "the result. Two of the three carry no "
                                 "commercial restriction.")
                        f_fidelity = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05,
                            label="Faithfulness to the original face",
                            info="0.5 is almost always right. Lower it when "
                                 "the face is badly damaged (the model "
                                 "invents more), raise it if the person stops "
                                 "looking like themselves. **CodeFormer "
                                 "only**: the other two have no such dial.")
                        f_center = gr.Checkbox(
                            value=False, label="Main face only",
                            info="By default every detected face is restored.")
                        f_run = gr.Button("🙂 Restore faces",
                                          variant="primary", size="lg")
                    with gr.Column(scale=4):
                        f_result = gr.Image(label="Restored faces", height=520,
                                            format="png",
                                            buttons=widgets.IMAGE_BUTTONS)
                        f_log = gr.Textbox(label="Log", lines=8,
                                           autoscroll=True,
                                           elem_classes="log-box")

                def do_face(img, model, fidelity, center,
                            progress=gr.Progress()):
                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not tools.face_is_installed():
                        raise gr.Error(t("Install CodeFormer first."))
                    logs: list[str] = []
                    progress(0.1, desc="Visages…")
                    try:
                        out = tools.face_restore(
                            img, fidelity=float(fidelity), model=model,
                            only_center=bool(center), log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERROR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Done")
                    logs.append(f"\n✅ Image : {out}")
                    return str(out), "\n".join(logs)

                f_run.click(do_face,
                            inputs=[f_image, f_model, f_fidelity, f_center],
                            outputs=[f_result, f_log])

            # Agrandir puis réparer les visages est LA suite d'opérations
            # normale. Sans ce relais il faudrait retrouver le fichier dans
            # `outputs/` et le recharger à la main. Le câblage se fait ici,
            # une fois `f_image` créé : un bouton ne peut pas écrire dans un
            # composant qui n'existe pas encore.
            def _hand_over_to_faces(button, source):
                def _go(image):
                    if image is None:
                        raise gr.Error(t("Produce an image first."))
                    return image, gr.Tabs(selected="face")

                button.click(_go, inputs=[source], outputs=[f_image, sub_tabs])

            _hand_over_to_faces(u_to_face, u_result)
            _hand_over_to_faces(seed_to_face, seed_result)

            # ---------- Upscale créatif SDXL (tuilé, Ultimate SD Upscale) ----
            with gr.Tab("✨ SDXL upscale", id="creative"):
                gr.Markdown(
                    "**Creative** “Ultimate SD Upscale”: pre-enlarges then "
                    "**refines tile by tile** with SDXL img2img at low "
                    "denoise (model **resident** → fast tiles, overlap "
                    "feather blending). Invents fine detail, Magnific-style. "
                    "**100% GPU** (PyTorch).")
                _installer_block(
                    "Creative SDXL upscale",
                    "PyTorch + diffusers (~9.5 GB: SDXL base + VAE fp16-fix + "
                    "ControlNet Tile). Model resident on the GPU. No command "
                    "to type.",
                    tools.install_upscale_stream, tools.upscale_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        c_image = gr.Image(
                            label="Image to upscale", type="pil",
                            buttons=widgets.IMAGE_VIEW_ONLY)
                        _ckpts = tools.list_upscale_checkpoints()
                        c_model = gr.Dropdown(
                            choices=_ckpts,
                            value=(_ckpts[0][1] if _ckpts else None),
                            label="SDXL model (drop your .safetensors into "
                                  "tools_repo/upscale/checkpoints/)")
                        c_vae = gr.Radio(
                            [(t("VAE fp16-fix (external, recommended)"), False),
                             (t("Model's built-in VAE"), True)],
                            value=False, label="VAE")
                        _ups = registry.list_upscalers()
                        c_esrgan = gr.Dropdown(
                            choices=[(t("Lanczos (default)"), "")]
                                    + [(u, u) for u in _ups],
                            value="", label="Pre-upscale (base before SDXL)",
                            info="A single pass, always. A model whose factor "
                                 "EXCEEDS the enlargement you asked for (a ×4 "
                                 "for a ×2) is the best choice: the downscale "
                                 "that follows acts as anti-aliasing.")
                        c_refresh = gr.Button("↻ Refresh models", size="sm")
                        c_preset = gr.Dropdown(
                            choices=[(t(p["name"]), p["name"])
                                     for p in UPSCALE_PRESETS],
                            value=None,
                            label="Preset (sets prompt, negative, creativity, "
                                  "CFG and structure)")
                        c_preset_msg = gr.Markdown("", elem_classes="feedback")
                        c_prompt = gr.Textbox(
                            label="Prompt (optional — guides the detail, KEEP "
                                  "IT SHORT: ~77 tokens max for SDXL; no need "
                                  "to copy the generation prompt)", lines=2,
                            placeholder="highly detailed skin texture, sharp "
                                        "focus, photorealistic")
                        c_negative = gr.Textbox(
                            label="Negative prompt (empty = photo-oriented default)",
                            lines=2,
                            placeholder="photorealistic, film grain, noise…",
                            info="What SDXL is forbidden to add. On drawings, "
                                 "this is what keeps grain and photo texture "
                                 "off the flat color areas.")
                        c_scale = gr.Slider(
                            1.5, 8.0, value=2.0, step=0.5,
                            label="Enlargement factor",
                            info="Up to ~8K (capped at 8192 px). ×6–×8 = many "
                                 "tiles: very slow + ~1–2 GB RAM.")
                        c_denoise = gr.Slider(
                            0.15, 0.75, value=0.35, step=0.05,
                            label="Creativity (denoise — ↑ = invented detail)")
                        _cn_ok = tools.upscale_cn_is_installed()
                        c_controlnet = gr.Checkbox(
                            value=_cn_ok,
                            label="🔒 ControlNet Tile (locks structure — lets "
                                  "you raise creativity without drifting)")
                        if not _cn_ok:
                            gr.Markdown(
                                "> ℹ️ ControlNet **not downloaded yet**: "
                                "re-run “Install the creative SDXL upscale” "
                                "above (adds ~2.5 GB) then **restart** to "
                                "enable the structure lock.")
                        c_cnscale = gr.Slider(
                            0.2, 1.0, value=0.6, step=0.05,
                            label="ControlNet fidelity (↑ = more faithful)")
                        with gr.Row():
                            c_steps = gr.Slider(10, 40, value=24, step=1,
                                                label="Steps / tile")
                            c_cfg = gr.Slider(1.0, 12.0, value=6.0, step=0.5,
                                              label="CFG")
                        c_tile = gr.Slider(640, 1280, value=1024, step=64,
                                           label="Tile size")
                        with gr.Row():
                            c_run = gr.Button("✨ Upscale", variant="primary",
                                              size="lg", scale=2)
                            c_stop = gr.Button("⏹️ Cancel", variant="stop",
                                               size="sm")
                    with gr.Column(scale=4):
                        c_result = gr.Image(
                            label="Live preview (full resolution in outputs/)", height=520, format="png",
                            buttons=widgets.IMAGE_BUTTONS)
                        c_log = gr.Textbox(label="Log", lines=12,
                                           autoscroll=True, elem_classes="log-box")

                def _apply_preset(name):
                    """Applique TOUT le préréglage et dit ce qu'il a changé.

                    Un préréglage qui règle six choses en silence est
                    indéfendable : on affiche donc, en clair, ce qui vient
                    d'être posé — et notamment quel modèle de pré-agrandissement
                    a été choisi, puisque c'est lui qui fait le gros du travail
                    sur du dessin."""
                    pre = next((p for p in UPSCALE_PRESETS
                                if p["name"] == name), None)
                    if pre is None:
                        return ((gr.update(),) * 8) + (gr.update(value=""),)

                    done: list[str] = []
                    esrgan_up = gr.update()
                    want = pre.get("esrgan")
                    if want == "drawing":
                        model = registry.drawing_upscaler()
                        if model:
                            esrgan_up = gr.update(value=model)
                            done.append(t("pre-upscale **{m}** (drawing) "
                                          "instead of Lanczos").format(m=model))
                        else:
                            done.append(t("⚠️ no **drawing** upscaler "
                                          "installed — the base stays on "
                                          "Lanczos (softer linework). "
                                          "Download the upscalers from the “🔼 "
                                          "Upscale” tab."))
                    elif want:
                        esrgan_up = gr.update(value=want)

                    cn = pre.get("controlnet")
                    cn_up = gr.update()
                    if cn is not None:
                        # ControlNet ne peut être coché que s'il est téléchargé.
                        cn = bool(cn) and tools.upscale_cn_is_installed()
                        cn_up = gr.update(value=cn)
                        if pre.get("controlnet") and not cn:
                            done.append(t("⚠️ ControlNet Tile not installed: "
                                          "structure will not be locked."))
                        elif cn:
                            done.append(t("structure locked by ControlNet "
                                          "Tile ({v})").format(
                                v=pre.get("cn_scale", 0.6)))

                    done.append(t("creativity {d} · CFG {c} · {s} steps").format(
                        d=pre.get("denoise", 0.35), c=pre.get("cfg", 6.0),
                        s=pre.get("steps", 24)))
                    if pre.get("negative"):
                        done.append(t("matching negative"))

                    return (gr.update(value=pre.get("prompt", "")),
                            gr.update(value=pre.get("negative", "")),
                            gr.update(value=pre.get("denoise", 0.35)),
                            gr.update(value=pre.get("cfg", 6.0)),
                            gr.update(value=pre.get("steps", 24)),
                            cn_up,
                            gr.update(value=pre.get("cn_scale", 0.6)),
                            esrgan_up,
                            gr.update(value="✅ " + " · ".join(done)))

                c_preset.change(
                    _apply_preset, inputs=[c_preset],
                    outputs=[c_prompt, c_negative, c_denoise, c_cfg, c_steps,
                             c_controlnet, c_cnscale, c_esrgan, c_preset_msg])

                def _refresh_models():
                    ck = tools.list_upscale_checkpoints()
                    ups = registry.list_upscalers()
                    return (gr.update(choices=ck,
                                      value=(ck[0][1] if ck else None)),
                            gr.update(choices=[(t("Lanczos (default)"), "")]
                                              + [(u, u) for u in ups]))

                c_refresh.click(_refresh_models, outputs=[c_model, c_esrgan])

                def do_creative(img, prompt, negative, scale, denoise, steps,
                                cfg, tile, controlnet, cn_scale, model,
                                vae_integrated, esrgan,
                                progress=gr.Progress()):
                    import queue
                    import threading
                    import time
                    from PIL import Image as _PILImage

                    if img is None:
                        raise gr.Error(t("Provide an image."))
                    if not tools.upscale_is_installed():
                        raise gr.Error(t("Install the creative SDXL upscale "
                                         "first (accordion above)."))
                    if controlnet and not tools.upscale_cn_is_installed():
                        gr.Warning(t("ControlNet not installed: upscaling "
                                     "without it. Re-run the installer to "
                                     "enable it."))
                        controlnet = False
                    settings.ensure_dirs()
                    preview_path = (settings.TMP_DIR /
                                    f"usdu_preview_{int(time.time()*1000)}.png")
                    try:
                        preview_path.unlink()
                    except OSError:
                        pass
                    q: "queue.Queue[str | None]" = queue.Queue()
                    state: dict = {}

                    def worker():
                        try:
                            out = tools.ultimate_upscale(
                                img, scale=float(scale), prompt=prompt or "",
                                negative=negative or "",
                                denoise=float(denoise), steps=int(steps),
                                cfg=float(cfg), tile=int(tile),
                                use_controlnet=bool(controlnet),
                                cn_scale=float(cn_scale),
                                base_model=(model or None),
                                integrated_vae=bool(vae_integrated),
                                esrgan_model=(esrgan or None),
                                preview_path=preview_path, log=q.put)
                            state["out"] = str(out)
                        except Exception as exc:  # noqa: BLE001
                            state["err"] = str(exc)
                        finally:
                            q.put(None)

                    threading.Thread(target=worker, daemon=True).start()
                    logs: list[str] = []
                    last_mtime = None
                    last_emit = 0.0
                    import re as _re
                    tile_re = _re.compile(r"tuile (\d+)/(\d+)")
                    progress(0.03, desc="Preparing…")
                    while True:
                        try:
                            line = q.get(timeout=0.3)
                        except queue.Empty:
                            line = ""
                        if line is None:
                            break
                        if line:
                            logs.append(line)
                            mt = tile_re.search(line)
                            if mt:
                                cur, tot = int(mt.group(1)), max(1, int(mt.group(2)))
                                progress(0.05 + 0.9 * cur / tot,
                                         desc=f"Tuile {cur}/{tot}")
                        prev = gr.update()
                        new_prev = False
                        if preview_path.exists():
                            try:
                                mt = preview_path.stat().st_mtime
                                if mt != last_mtime:
                                    with _PILImage.open(preview_path) as _p:
                                        prev = _p.copy()
                                    last_mtime = mt
                                    new_prev = True
                            except (OSError, ValueError):
                                pass
                        now = time.time()
                        if new_prev or (line and now - last_emit >= 0.5):
                            last_emit = now
                            yield prev, "\n".join(logs[-400:])

                    if "err" in state:
                        logs.append(f"\n[ERROR] {state['err']}")
                        yield gr.update(), "\n".join(logs)
                        return
                    progress(1.0, desc="Done")
                    out = state.get("out")
                    if out:
                        try:
                            im = _PILImage.open(out)
                            logs.append(f"\n✅ Full-resolution image "
                                        f"({im.width}x{im.height}): {out}")
                        except Exception:  # noqa: BLE001
                            pass
                    # On affiche le FICHIER pleine résolution (téléchargement =
                    # image réelle, pas une preview réduite).
                    yield (str(out) if out else gr.update()), "\n".join(logs)

                c_evt = c_run.click(
                    do_creative,
                    inputs=[c_image, c_prompt, c_negative, c_scale, c_denoise,
                            c_steps, c_cfg, c_tile, c_controlnet, c_cnscale,
                            c_model, c_vae, c_esrgan],
                    outputs=[c_result, c_log])
                widgets.stop_into_log(c_stop, tools.cancel, c_log, [c_evt])

        # --- Réception d'une image envoyée depuis un onglet de génération ---
        if pending_toolkit is not None and tabs is not None:
            _keys = ["depth", "bg", "sam", "esrgan", "seedvr2", "creative"]

            def _consume(pend):
                # +2 sorties fixes : le groupe parent et le sélecteur d'outil.
                if not pend:
                    return tuple([gr.update()] * (len(_keys) + 2) + [None])
                path, dest = pend
                sub = gr.Tabs(selected=dest) if dest in _keys else gr.update()
                # Remonter le groupe « Outils » : sans ça l'outil est bien
                # sélectionné, mais dans un onglet que personne n'affiche.
                top = (gr.Tabs(selected=tab_id) if parent_tabs is not None
                       else gr.update())
                img_upd = [gr.update(value=path) if k == dest else gr.update()
                           for k in _keys]
                return tuple([top, sub] + img_upd + [None])

            _outs = [parent_tabs if parent_tabs is not None else sub_tabs,
                     sub_tabs, d_image, b_image, s_image, u_image,
                     seed_image, c_image, pending_toolkit]
            tabs.select(_consume, inputs=[pending_toolkit], outputs=_outs)
