"""Onglet « 🧹 Gestion & aide » : inventaire de ce qui occupe le disque
(moteurs, modèles, add-ons, données), suppression sélective, et documentation
de toutes les options de l'application.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from .. import inventory, settings, storage
from ..i18n import t
from . import widgets

CATEGORIES = ["Engines", "Models", "Toolkit add-ons", "Your data"]


def _choices_and_summary():
    """Cases à cocher (installés seulement) + tableau récapitulatif."""
    its = inventory.items()
    choices: list[tuple[str, str]] = []
    rows: list[str] = []
    total = 0
    for cat in CATEGORIES:
        cat_items = [i for i in its if i.category == cat]
        if not cat_items:
            continue
        rows.append(f"\n**{cat}**\n")
        rows.append("| | Item | Size |")
        rows.append("|---|---|---|")
        for i in cat_items:
            size = i.size
            total += size
            mark = "✅" if i.installed else "—"
            warn = " ⚠️" if (i.protected and i.installed) else ""
            rows.append(f"| {mark} | {i.label}{warn} | "
                        f"{inventory.human(size)} |")
            if i.installed:
                label = f"{i.label} — {inventory.human(size)}"
                if i.protected:
                    label = "⚠️ " + label
                choices.append((label, i.key))
    rows.append(f"\n**Total in use: {inventory.human(total)}**")
    return choices, "\n".join(rows)


def _move_choices() -> list[tuple[str, str]]:
    """Éléments (dossiers) déplaçables individuellement, avec leur état."""
    out: list[tuple[str, str]] = []
    for i in inventory.items():
        dirs = [p for p in i.paths if p.is_dir()]
        if not dirs:
            continue
        moved = [p for p in dirs if storage.is_link(p)]
        if moved:
            tgt = storage.link_target(moved[0])
            label = f"↗️ {i.label} — moved to {tgt}"
        else:
            label = f"{i.label} — {inventory.human(i.size)}"
        out.append((label, i.key))
    return out


def build_manage_tab():
    with gr.Tab("🧹 Manage & help"):
        # ------------------------------------------------------------------ #
        #  Inventaire & suppression
        # ------------------------------------------------------------------ #
        gr.Markdown(
            "### Disk space\nEverything the application has downloaded, with "
            "its size. Tick what you want to delete, then confirm. Items "
            "marked **⚠️** are **your data** (LoRAs, custom models, "
            "creations) — think twice. Everything else can be **downloaded "
            "again** from inside the app.")

        _choices, _summary = _choices_and_summary()
        summary_md = gr.Markdown(_summary)
        picks = gr.CheckboxGroup(choices=_choices, value=[],
                                 label="Items to delete")
        with gr.Row():
            confirm = gr.Checkbox(
                value=False,
                label="I confirm I want to delete the ticked items")
            refresh = gr.Button("↻ Refresh sizes", size="sm")
        delete_btn = gr.Button("🗑️ Delete the selection", variant="stop")
        result = gr.Markdown("")

        def _refresh():
            ch, summ = _choices_and_summary()
            return (gr.update(choices=ch, value=[]), gr.update(value=summ),
                    gr.update(value=False), "")

        refresh.click(_refresh,
                      outputs=[picks, summary_md, confirm, result])

        def _delete(keys, ok):
            if not keys:
                return (gr.update(), gr.update(), gr.update(),
                        t("Nothing ticked."))
            if not ok:
                return (gr.update(), gr.update(), gr.update(),
                        t("⚠️ Tick **“I confirm”** to delete."))
            msgs, freed = inventory.delete(list(keys))
            ch, summ = _choices_and_summary()
            report = "\n".join(msgs) + \
                f"\n\n**{inventory.human(freed)} freed.**"
            return (gr.update(choices=ch, value=[]), gr.update(value=summ),
                    gr.update(value=False), report)

        delete_btn.click(_delete, inputs=[picks, confirm],
                         outputs=[picks, summary_md, confirm, result])

        gr.Markdown(
            f"*Locations: models `{settings.MODELS_DIR.name}/`, engines "
            f"`{settings.BIN_DIR.name}/`, add-ons `tools_repo/`, LoRAs "
            f"`{settings.LORA_DIR.name}/`, outputs "
            f"`{settings.OUTPUT_DIR.name}/`.*")

        # ------------------------------------------------------------------ #
        #  Emplacement des modèles (dossier externe)
        # ------------------------------------------------------------------ #
        gr.Markdown("---\n### 📁 Where the models live")
        with gr.Accordion("Move the models to another drive (e.g. a fast NVMe)", open=False):
            gr.Markdown(
                "Models can live **outside the project folder**: useful to "
                "put them on an **NVMe** (much faster loads) or on a larger "
                "drive. Repeated reads **do not wear** an SSD — only writes "
                "count.\n\n⚠️ The change takes effect **when the application "
                "restarts**.")
            loc_now = gr.Markdown(
                f"**Current folder:** `{storage.current()}`"
                + ("  *(project default)*" if storage.is_default() else ""))
            dest_box = gr.Textbox(
                value=storage.configured(),
                label="New folder (absolute path)",
                placeholder="e.g. D:\\AI\\models  or  /mnt/nvme/models")
            with gr.Row():
                move_btn = gr.Button("📦 Move the models here",
                                     variant="primary")
                point_btn = gr.Button("🔗 Point here without moving")
                reset_btn = gr.Button("↩️ Back to the project folder")
            loc_log = gr.Textbox(label="Log", lines=10, autoscroll=True,
                                 elem_classes="log-box")

            def _do_move(raw):
                dest, err = storage.validate(raw)
                if err:
                    yield gr.update(), f"❌ {err}"
                    return
                lines: list[str] = []
                yield gr.update(), t("⏳ Moving… (slow across different drives "
                                     "— do not close)")
                for msg in storage.move(dest):
                    lines.append(msg)
                    yield gr.update(), "\n".join(lines)
                yield (gr.update(value=f"**Current folder:** `{storage.current()}` "
                                       "— *restart to apply*"),
                       "\n".join(lines))

            move_btn.click(_do_move, inputs=[dest_box],
                           outputs=[loc_now, loc_log])

            def _do_point(raw):
                dest, err = storage.validate(raw)
                if err:
                    return gr.update(), f"❌ {err}"
                msg = storage.save(dest)
                return (gr.update(value=f"**Current folder:** `{storage.current()}` "
                                        "— *restart to apply*"),
                        msg + "\n\n*(No file was moved: the folder you gave "
                              "must already contain your models, otherwise "
                              "they will be downloaded again.)*")

            point_btn.click(_do_point, inputs=[dest_box],
                            outputs=[loc_now, loc_log])

            def _do_reset():
                return (gr.update(value=f"**Current folder:** "
                                        f"`{settings.DEFAULT_MODELS_DIR}` "
                                        "— *restart to apply*"),
                        storage.save(None), gr.update(value=""))

            reset_btn.click(_do_reset, outputs=[loc_now, loc_log, dest_box])

        with gr.Accordion("Move ONLY some items (e.g. the 16 GB of 3D models)", open=False):
            gr.Markdown(
                "Moves **the ticked items** to another drive and leaves a "
                "**link** in their place: the application still finds them, "
                "**with no restart and no setting to change**. Ideal for "
                "moving the large 3D models out while keeping the image "
                "models on the NVMe.\n\n⚠️ The destination drive must stay "
                "**connected** — otherwise the moved items become "
                "unreachable.")
            sel_picks = gr.CheckboxGroup(choices=_move_choices(), value=[],
                                         label="Items to move out / bring back")
            sel_dest = gr.Textbox(
                label="Destination folder (absolute path)",
                placeholder="e.g. E:\\AI-big-models  or  /mnt/hdd/ai")
            with gr.Row():
                sel_move = gr.Button("📦 Move + create the link",
                                     variant="primary")
                sel_back = gr.Button("↩️ Bring back into the project")
                sel_refresh = gr.Button("↻ Refresh", size="sm")
            sel_log = gr.Textbox(label="Log", lines=10, autoscroll=True,
                                 elem_classes="log-box")

            def _sel_paths(keys):
                out = []
                for k in keys or []:
                    it = inventory.by_key(k)
                    if it:
                        out += [p for p in it.paths if p.is_dir()]
                return out

            def _sel_move(keys, raw):
                paths = _sel_paths(keys)
                if not paths:
                    yield gr.update(), t("Nothing ticked.")
                    return
                raw = (raw or "").strip().strip('"')
                if not raw or not Path(raw).expanduser().is_absolute():
                    yield gr.update(), t("❌ Give a destination folder as an "
                                         "**absolute** path.")
                    return
                lines: list[str] = []
                yield gr.update(), t("⏳ Moving…")
                for msg in storage.relocate(paths,
                                            Path(raw).expanduser()):
                    lines.append(msg)
                    yield gr.update(), "\n".join(lines)
                yield gr.update(choices=_move_choices(), value=[]), \
                    "\n".join(lines)

            sel_move.click(_sel_move, inputs=[sel_picks, sel_dest],
                           outputs=[sel_picks, sel_log])

            def _sel_back(keys):
                paths = _sel_paths(keys)
                if not paths:
                    yield gr.update(), t("Nothing ticked.")
                    return
                lines: list[str] = []
                yield gr.update(), t("⏳ Retour en cours…")
                for msg in storage.restore(paths):
                    lines.append(msg)
                    yield gr.update(), "\n".join(lines)
                yield gr.update(choices=_move_choices(), value=[]), \
                    "\n".join(lines)

            sel_back.click(_sel_back, inputs=[sel_picks],
                           outputs=[sel_picks, sel_log])
            sel_refresh.click(lambda: (gr.update(choices=_move_choices(),
                                                 value=[]), ""),
                              outputs=[sel_picks, sel_log])

        # ------------------------------------------------------------------ #
        #  Diagnostic « image cassée »
        # ------------------------------------------------------------------ #
        # Ce symptôme ne se reproduit pas depuis le code : sur la machine de
        # développement, six noms de fichiers hostiles (accents, #, %,
        # cyrillique, 120 caractères) s'affichent tous. Le problème est donc
        # dans l'environnement, et demander d'ouvrir la console du navigateur
        # pour lire un code HTTP n'est pas une réponse acceptable ici. Ce
        # bouton refait le trajet exact d'une image importée et le raconte.
        gr.Markdown("---\n### 🩺 An imported image will not display?")
        with gr.Accordion("Diagnose image display", open=False):
            gr.Markdown(
                "If an image you import stays a **broken icon**, this test "
                "repeats its whole journey: writing to the interface cache, "
                "then serving it over the web server. **The test image must "
                "appear on the right** — if it does, the chain works and the "
                "problem is the imported file; if it does not, the report "
                "names the link that failed.",
                elem_classes="hint")
            diag_btn = gr.Button("🩺 Run the diagnostic", variant="primary")
            with gr.Row():
                diag_md = gr.Markdown("")
                with gr.Column():
                    diag_img = gr.Image(label="Image de test", height=200,
                                        interactive=False,
                                        buttons=widgets.IMAGE_VIEW_ONLY)
                    # Une tuile par format : celle qui casse DÉSIGNE la cause,
                    # là où un contrôle général ne peut que dire « tout va
                    # bien » — ce qu'il disait, pendant que les imports
                    # cassaient.
                    diag_tiles = gr.Gallery(label="One format per tile",
                                            height=200, columns=5,
                                            buttons=widgets.IMAGE_VIEW_ONLY)
                    diag_last = gr.Image(
                        label="Last imported file (yours)", height=220,
                        interactive=False, buttons=widgets.IMAGE_VIEW_ONLY)

            def _diagnose():
                from .. import imgcheck
                r = imgcheck.report()
                return r.markdown, r.test_image, r.tiles, r.last_upload

            diag_btn.click(_diagnose,
                           outputs=[diag_md, diag_img, diag_tiles, diag_last])

        # ------------------------------------------------------------------ #
        #  Documentation des options
        # ------------------------------------------------------------------ #
        gr.Markdown("---\n### 📖 Help — what does each option do?")

        with gr.Accordion("🎨 Generation tabs (Flux.2 / Krea 2)",
                          open=False):
            gr.Markdown(
                "**Prompt** — your description. On an **editing model** "
                "(Flux.2), describe the\n*change* to apply.\n**Negative "
                "prompt** — what you do not want. **Shown only when the model "
                "honours\nit** (CFG > 1): distilled models run at CFG 1.0 and "
                "ignore it.\n\n**🎨 Styles** — one collapsed section gathering "
                "three banks, all **stackable**:\n· **🎭 Custom preset** — a "
                "prefix you write and save, prepended to every\ngeneration; "
                "“— None —” removes it. Presets are **global**: saved once, "
                "they\nappear in all three tabs. Some **ship with the app** "
                "and survive updates; they\ncannot be deleted, but saving a "
                "style under the same name creates your own\nversion, which "
                "takes precedence — deleting yours restores the original.\n· "
                "**📷 Photo styles** (139) — quality, light, lens, film stock, "
                "mood. Your\nsubject is inserted *into* each ticked style. "
                "Bank © ghleg, MIT.\n· **🖍️ Artistic styles** (397) — anime, "
                "cartoon, comics, drawing, design,\npainting; appended "
                "*after* your subject. 🎲 picks one at random.\n\n**🎨 Generate "
                "/ ⏹️ Stop / 🗑️ Clear** — right under the prompt. *Clear* "
                "empties\nthe prompt, the negative and every trace of "
                "improvement.\n**↩️ Restore the original prompt** — appears "
                "after an improvement and puts back\nexactly what you had "
                "written, `--` parameters included.\n\nℹ️ **A preset "
                "translates nothing**: it is a prefix glued in front of your "
                "text.\nWrite in French and you get French with an English "
                "header. It is the **✨\nImprove** button that translates and "
                "shapes — and it now takes the active\npreset into account: "
                "it describes the subject without adding a camera, a lens,\na "
                "light or a treatment that would contradict it. Recommended "
                "order: **pick the\npreset, then improve**.\n\n**✨ Prompt "
                "improver** — a small local LLM rewrites your idea as an "
                "English\nprompt (add-on, installed on demand). **Intensity** "
                "sets how far the rewrite\ngoes; **Suggestions** produces "
                "several at once in a single model load — click\nthe one you "
                "prefer. The **grey line** under the menus summarises what "
                "the\nbutton will do before you click it.\n\n**Format / Width "
                "/ Height** — presets aligned on the model's "
                "**native\nresolutions** (it renders better "
                "there).\n**Steps** — denoising steps. Distilled models are "
                "calibrated for few steps\n(4–8); beyond that you gain "
                "little.\n**CFG** — prompt guidance strength. **1.0 = off** "
                "(normal for distilled\nmodels). >1 follows the prompt more "
                "closely (and enables the negative). 0 can\n**ignore** the "
                "prompt.\n**Preset** — the sampler/scheduler/steps "
                "combination recommended by the model's\nown documentation. "
                "When in doubt, keep it.\n**Sampler / Scheduler** — sampling "
                "algorithms. “Auto” lets the engine choose,\nwhich is the "
                "safe option.\n**Flow shift** — 0 = auto (recommended). Too "
                "low means grain at high\nresolution.\n**Seed** — -1 is "
                "random; a fixed value **replays the same "
                "image**.\n**Images** — how many to generate (consecutive "
                "seeds).\n\n**🖼️ Reference image** — *editing* (Flux.2: up to "
                "3 images, driven by the\nprompt) or *image-to-image* "
                "(**transformation strength**: low stays faithful to\nthe "
                "original, high reinvents it).\n**🧩 Centred outpaint** — "
                "enlarges the canvas symmetrically and lets the model\nfill "
                "the borders (experimental, editing models only). For a "
                "**directional**\noutpaint, see the “🖼️ Outpaint” tab.\n**🧩 "
                "LoRA** — extra styles or concepts, with a weight (≈ 0.6–1.0 "
                "usually). They\nmust match the **model's "
                "architecture**.\n**📂 Local files** — use a model you dropped "
                "in by hand instead of the catalog\none.")

        with gr.Accordion("⚙️ Settings (hardware & optimization)", open=False):
            gr.Markdown(
                "**One question, not twenty.** Everything the machine can "
                "work out from your\nhardware, it works out: diffusion "
                "quantization from **VRAM**, text-encoder\nquantization from "
                "**RAM**, flash-attention, CPU offload, VAE tiling. What "
                "is\nleft is a matter of taste, so Settings asks it once: "
                "**more memory headroom, or\nmore detail?** Three notches, "
                "and the page tells you what each one changed —\nin "
                "observable consequences, not flag names.\n\n**🔧 Expert** "
                "(collapsed, optional) exposes the raw sd.cpp options for "
                "when you\nwant to measure instead of trust: forced "
                "quantization, memory flags, step\ncache, direct convolution, "
                "compute budget (`--max-vram`), layer streaming, and\nthe "
                "**resident engine** (keeps the model loaded between images — "
                "no live\npreview in exchange).\n\n**🧮 Multi-GPU** (two or "
                "more cards) offers three mutually exclusive\nstrategies: "
                "*single card* (the reliable default), *text encoder on the "
                "second\ncard*, or *auto-fit* (sd.cpp spreads everything). It "
                "is memory-aware, not\ntopology-aware — the **benchmark "
                "button measures them on your machine** instead\nof asking "
                "you to bet. Note that a card without tensor cores (Pascal, "
                "GTX 16xx)\nis refused for the encoder automatically: it runs "
                "fp16 at a fraction of its\nfp32 rate, and prompt encoding is "
                "one big fp16 matmul.\n\n**⚡ Step cache** reuses computation "
                "between diffusion steps. It pays beyond\n~10 steps; our "
                "distilled models run 4–8, so leave it off unless you "
                "measured\notherwise.")

        with gr.Accordion("🖼️ Outpaint (extend an image)", open=False):
            gr.Markdown(
                "Extends an image **left, right, up, down or all around**, "
                "Midjourney style. It\nworks with **any model** in the "
                "catalog and **without a prompt** — no\ninpainting model is "
                "required.\n\n**⚠️ Use an EDITING model** (Flux.2 Klein). "
                "This is not a preference, it is\nwhat makes it work. An "
                "editing model receives the enlarged canvas as "
                "a\n**reference** — its image conditioning tells it what the "
                "scene contains — plus\nan **extension instruction written "
                "automatically**. An ordinary text-to-image\nmodel only "
                "receives a noised latent in image-to-image: it does not know "
                "what it\nis continuing, so it **reinvents**. No strength, "
                "feather or fill setting fixes\nthat; it is a limit of the "
                "method. The fallback exists so you are not blocked,\nnot "
                "because it gives a good result.\n\nThat is also what "
                "“without a prompt” means: you write nothing, but the "
                "model\nreceives a precise instruction naming the extended "
                "sides and asking it to carry\non the perspective, light, "
                "palette and style without touching the original "
                "or\nduplicating a subject.\n\n**How it works** — the canvas "
                "is enlarged, the new area is filled (neutral grey\nfor an "
                "editing model: a frankly empty area reads as “fill me”), the "
                "model\ngenerates, the **tone of the new pixels is matched** "
                "to the original, then the\n**original image is pasted back** "
                "on top.\n\n⚠️ Tone matching is not a detail: in "
                "image-to-image the model re-renders the\nWHOLE canvas with "
                "more contrast. Without it, the pasted-back original shows "
                "as\na duller rectangle in the middle — and feathering cannot "
                "help, the gap being\nglobal rather than "
                "local.\n\n**Direction** — one or more sides to "
                "extend.\n**Extension per side** — the share added on each "
                "chosen side (0.25 = +25%). The\nnew size is shown below; it "
                "is aligned to 16 px and capped at **2048 px** per\nside "
                "(beyond that the margins are reduced "
                "automatically).\n**Model** — any installed model. The "
                "**sampler, CFG and steps** come from its\nrecommended "
                "settings.\n**Prompt** — *optional*. Leave it empty for a "
                "neutral extension; filling it\nonly steers what appears in "
                "the new area.\n**Border fill** — *Neutral grey* (default on "
                "an editing model): the area to\nfill is unambiguous. "
                "*Blurred stretch* extends the edge pixels, which "
                "carries\ncolour but no shape. *Mirror* gives perfect "
                "continuity on a regular pattern\n**but reflects any subject "
                "near the edge** — and the model happily turns "
                "that\nreflection into a second, very real object. Keep it "
                "for uniform backgrounds.\n**Generation strength** — "
                "**fallback only**, greyed out on an editing model\n(which is "
                "driven by the instruction, not by a strength). High invents "
                "freely;\nlow stays close to the pre-fill.\n**Seam feather** "
                "— width of the gradient at the junction with the "
                "original.\n~24 px erases the seam. Careful: it blends a band "
                "roughly **twice that value**\n*inside* the original's edge — "
                "which is precisely what makes the seam vanish.\n**0 = hard "
                "paste**, original strictly intact everywhere.\n**Tone "
                "matching** — 0 to 1: how far the new pixels are pulled back "
                "onto the\noriginal's contrast and colour. Lower it only if "
                "the correction overshoots on\nan unusual image.\n**Seed** — "
                "-1 is random; a fixed value replays the same "
                "extension.\n**♻️ Re-extend the result** — reloads the result "
                "as the new input, to chain\nextensions (right, then up, and "
                "so on).\n\nEvery result is saved in `outputs/` with a `.txt` "
                "file beside it recording the\nmodel, seed and settings.")

        with gr.Accordion("🧊 Image → 3D (trellis)", open=False):
            gr.Markdown(
                "**Weights** — model variant: **f16** (~16.5 GB), **q8** "
                "(~9.9 GB) or **q4**\n(~6 GB). ⚠️ **f16 is the FASTEST** when "
                "it fits: in ggml, quantized weights are\ndequantized on the "
                "fly during computation, and that overhead is *not* "
                "amortized\non a compute-bound 3D workload — unlike LLMs. "
                "q8/q4 exist only to make a mode\n(1024/1536) fit that would "
                "otherwise overflow. Several variants can coexist;\nyou "
                "switch at generation time.\n\n**Pad to square** — TRELLIS "
                "pre-processes its input as a **square**: a 16:9\nimage sent "
                "as-is comes out **squashed**. Ticked (default), neutral bars "
                "are\nadded to keep the proportions; background removal takes "
                "them away again.\n\n**Geometry resolution** — **512** is the "
                "“light” path, the only one that fits a\ncard with ≤ 12 GB. "
                "**1024/1536** want ~16 GB or more: below that, "
                "geometry\noften comes out **corrupt (“blobs”)** rather than "
                "failing cleanly.\n\n**Seed** — -1 is random; a fixed value "
                "replays the same object.\n**Background removal** — "
                "*BiRefNet* (quality) or *Threshold* (fast).\n**⚡ Resident "
                "server** — keeps the 3D engine alive between generations "
                "(saves\n~30 s of reloading) **but holds the VRAM**: stop it "
                "before generating images.\n**Decimation** — target face "
                "count (lower = lighter mesh).\n**UV atlas** — texture "
                "resolution (1024 → 4096).\n**Geometry only** — no texture, "
                "faster.\n**Card used** — ⚠️ on a multi-GPU machine, letting "
                "the computation spill onto a\n**Pascal (GTX 10xx)** card can "
                "corrupt the geometry: **pin the newest card**.\n**Require "
                "the GPU** — prevents a silent CPU fallback (which would take "
                "hours).\nLeave it ticked.")

        with gr.Accordion("🔧 Convert to GGUF · 🧰 Toolkit", open=False):
            gr.Markdown(
                "**Convert to GGUF** — quantizes a model dropped into "
                "`models/custom/` to a\nlighter GGUF (100% CPU). sd.cpp "
                "accepts `q8_0 / q5_1 / q5_0 / q4_1 / q4_0 /\nf16` only "
                "(**no** k-quants here). `q5_1` is a good "
                "default.\n\n**Toolkit** — *Image → prompt* (read an image, "
                "get the prompt that would\nrecreate it), *Depth*, "
                "*Background removal*, *Cut out (SAM)*, *Layers (PSD)*,\nand "
                "the upscaling family, from the most faithful to the most "
                "inventive:\n\n- **Enlarge (ESRGAN)** — fast, deterministic, "
                "100% GPU. Pick a 🎨 **drawing /\n  anime** model for comics "
                "or illustration: a photo model puts halos on line\n  art and "
                "invents grain in flat areas. You can drop your own `.pth` "
                "files from\n  OpenModelDB into the upscalers folder.\n- "
                "**HD** — sd.cpp's native highres fix: your own generation "
                "model re-denoises\n  the whole image, no tiles, so no seams "
                "are possible.\n- **High resolution** — Flux.2 re-renders the "
                "image at its native resolution,\n  using it as both "
                "reference and starting latent. Best-looking, least "
                "faithful.\n- **Restore (SeedVR2)** — one-step diffusion "
                "restoration, 3B or 7B.\n- **Faces** — GFPGAN, "
                "RestoreFormer++ or CodeFormer on the detected faces only.\n  "
                "Run it last.\n- **Creative upscale (SDXL)** — tiled img2img "
                "that openly invents detail.\n\nEach tool installs in one "
                "click (PyTorch, on demand).")

        with gr.Accordion("🔄 Update the app",
                          open=False):
            gr.Markdown(
                "**`update.bat` updates the application itself.** It "
                "downloads the current code\nfrom GitHub and applies it in "
                "place: it writes only the files that actually\ndiffer, "
                "deletes what disappeared from the project (and only files it "
                "installed\nitself), backs up everything it replaces "
                "(`update.bat --rollback` undoes the\nupdate), and restores "
                "the previous version by itself if the new code does "
                "not\ncompile. Close the app first — Windows cannot replace a "
                "file that is open.\n\n**What is never touched:** `models/`, "
                "`loras/`, `outputs/`, `userdata/`,\n`tools_repo/`, `bin/`, "
                "`python/`. Your models, images, settings and engines "
                "are\nout of its reach by "
                "construction.\n\n**`maintenance.bat`** checks the "
                "installation and cleans up what an old\ncopy-paste update "
                "left behind: code of removed features, orphaned "
                "add-on\nfolders, orphaned models, stale `__pycache__`. It "
                "also reports any file from\nthe last update that has gone "
                "missing. It downloads nothing and never touches\nyour "
                "data.\n\n⚠️ **One trap worth knowing.** A Toolkit add-on is "
                "not just a folder of\nweights: its installer also pins "
                "Python package versions shared with the other\nadd-ons. When "
                "an update changes *how an add-on installs*, updating the "
                "app\nleaves that add-on frozen in its old state — and it "
                "only shows at the next use.\nIn that case, **click its "
                "“Install” button again**; already-downloaded weights\nare "
                "not fetched twice. `maintenance.bat` names precisely what is "
                "stale, so you\ndo not have to guess.\n\n**Summary**: "
                "`update.bat` for the app, `update-engine.bat` for "
                "sd.cpp,\n`update-trellis.bat` for the 3D engine, "
                "`maintenance.bat` when something feels\noff.")

        with gr.Accordion("🌐 Network, sharing & maintenance", open=False):
            gr.Markdown(
                "**Hugging Face endpoint** — an alternative mirror when HF is "
                "blocked or slow on\nyour network.\n**Civitai token** — "
                "needed to import some gated LoRAs.\n**LAN sharing** — "
                "`run-lan.bat` exposes the app to other machines on the "
                "local\nnetwork (mind your firewall).\n**`update.bat`** — "
                "updates the application itself (the code), with no "
                "manual\nre-download.\n**`maintenance.bat`** — checks the "
                "installation, removes obsolete files, purges\ncaches, "
                "verifies everything compiles. Never touches your models or "
                "outputs.\n**`update-engine.bat`** — updates the sd.cpp "
                "engine (official binary).\n**`update-trellis.bat`** — "
                "updates the **3D** engine (trellis.cpp). The ~16 GB\nof 3D "
                "models are not downloaded again.\n**📁 Model location** "
                "(above) — moves the models to another drive (NVMe, "
                "or\nsimply a larger one). *Move* transfers the files; *point "
                "without moving* reuses\na folder that already contains them. "
                "Takes effect **on restart**.")
