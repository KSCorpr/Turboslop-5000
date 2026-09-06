"""Onglet « 🖼️ Outpaint » : étendre une image à gauche / à droite / en haut /
en bas — ou tout autour — façon Midjourney.

Deux chemins, selon le modèle choisi (voir engine/outpaint.py) :

* modèle d'ÉDITION (Flux.2 Klein) — la toile agrandie part en
  RÉFÉRENCE (-r) avec une consigne d'extension générée automatiquement. Le
  modèle voit la scène et la prolonge. C'est le seul mode qui donne un résultat
  cohérent ;
* modèle sans édition — repli img2img (-i), conservé pour ne pas bloquer, mais
  le modèle ne voit rien et réinvente au lieu de prolonger.

Dans les deux cas la tonalité est recalée puis l'original recollé.
"""
from __future__ import annotations

import queue
import random
import threading
import time

import gradio as gr

from .. import registry, settings
from ..engine import generate as gen_engine
from ..engine import outpaint as op
from ..engine import sdcpp
from ..i18n import t
from . import widgets

PRESETS = [
    ("← Gauche", ["left"]),
    ("→ Droite", ["right"]),
    ("↑ Haut", ["top"]),
    ("↓ Bas", ["bottom"]),
    ("↔ Horizontal", ["left", "right"]),
    ("↕ Vertical", ["top", "bottom"]),
    ("⤢ All around", ["left", "right", "top", "bottom"]),
]


def _models() -> list:
    return registry.load_base_models(settings.load_prefs())


def _is_edit(m) -> bool:
    """Modèle d'ÉDITION natif (Flux.2 Klein) : il « voit » l'image qu'on
    lui passe en référence grâce à son encodeur vision. C'est la seule famille
    capable de prolonger une scène de façon sensée."""
    return (m.defaults.get("edit") if m else None) in (True, "full")


def _model_choices() -> tuple[list[tuple[str, str]], str | None]:
    """(choix du menu, modèle à présélectionner).

    On préfère un modèle d'ÉDITION installé : sur les autres, l'outpaint n'a
    pratiquement aucune chance de donner un résultat cohérent (cf. do_outpaint).
    """
    choices, first_edit, first_any = [], None, None
    for m in _models():
        ready = registry.model_is_ready(m)
        edit = _is_edit(m)
        if ready and edit and first_edit is None:
            first_edit = m.id
        if ready and first_any is None:
            first_any = m.id
        label = m.name if edit else f"{m.name} — no editing support, not recommended"
        choices.append((label if ready else f"{label} (not installed)", m.id))
    return choices, (first_edit or first_any)


def _defaults(model_id: str | None) -> dict:
    """Réglages d'échantillonnage propres au modèle choisi (catalogue)."""
    m = registry.get_base_model(model_id, settings.load_prefs()) if model_id \
        else None
    d = dict(m.defaults) if m else {}
    return {"steps": int(d.get("steps", 8) or 8),
            "cfg_scale": float(d.get("cfg_scale", 1.0) or 1.0),
            "sampler": d.get("sampler") or "euler",
            "schedule": d.get("scheduler") or "auto",
            "flow_shift": float(d.get("flow_shift", 0.0) or 0.0),
            "edit": _is_edit(m)}


def build_outpaint_tab(tab_id="outpaint", pending_outpaint=None, tabs=None,
                       parent_tabs=None):
    """`parent_tabs` : le groupe « 🧰 Outils » qui contient cet onglet (voir
    build_toolkit_tab — l'image doit atterrir dans un onglet VISIBLE)."""
    with gr.Tab("🖼️ Outpaint", id=tab_id):
        gr.Markdown(
            "### Extend an image (outpaint)\nEnlarges the canvas in the "
            "directions you pick and lets the model **continue the scene**, "
            "Midjourney-style.\n\n**Use it with an EDITING model** (Flux.2 "
            "Klein). Only such a model actually *looks* at the image, through "
            "its vision encoder: it knows what it is continuing. The enlarged "
            "canvas is handed to it as a **reference** along with an "
            "**extension instruction written automatically** — that is what "
            "“no prompt” means here: you type nothing, but the model receives "
            "a precise instruction.\n\nOn a model **without** editing "
            "support, the tab falls back to img2img: the model does not see "
            "the image, it only gets a noised latent, and it **reinvents "
            "instead of continuing**. The fallback is kept, but the result is "
            "incoherent by construction — that is a limit of the method, not "
            "a setting to tune.")

        _choices, _first = _model_choices()
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image to extend", type="pil",
                                 buttons=widgets.IMAGE_VIEW_ONLY)
                direction = gr.Radio(
                    [(lbl, "|".join(d)) for lbl, d in PRESETS],
                    value="|".join(PRESETS[-1][1]), label="Direction")
                amount = gr.Slider(
                    0.05, 1.0, value=0.25, step=0.05,
                    label="Extension per side (as a fraction of the image)",
                    info="0.25 = +25% on each side you pick. The canvas is "
                         "aligned to 16 px and capped at 2048 px.")
                plan_md = gr.Markdown("")
                model = gr.Dropdown(_choices, value=_first,
                                    label="Model used")
                mode_md = gr.Markdown("")
                prompt = gr.Textbox(
                    label="Prompt (facultatif)", lines=2,
                    placeholder="Leave empty: the extension instruction is "
                                "written automatically…",
                    info="Only useful to say what should appear in the new "
                         "area. The extension instruction itself is always "
                         "sent to the model.")
                with gr.Accordion("Advanced settings", open=False):
                    fill = gr.Radio(
                        op.FILLS, value="neutral",
                        label="Border fill",
                        info="What the model sees where the void is. “Neutral "
                             "grey” with an editing model: the area to fill "
                             "is unambiguous.")
                    strength = gr.Slider(
                        0.3, 1.0, value=0.85, step=0.05,
                        label="Generation strength (models WITHOUT editing)",
                        interactive=False,
                        info="No effect on an editing model: that one is "
                             "driven by the instruction, not by a strength.")
                    feather = gr.Slider(
                        0, 96, value=24, step=4,
                        label="Join feather (px)",
                        info="Softens the join with the original image. 0 = a "
                             "hard paste.")
                    tone = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.1,
                        label="Tone matching",
                        info="Pulls the new area's contrast and colour back "
                             "onto the original's. At 0, the centre can look "
                             "duller than the generated surroundings.")
                    steps = gr.Slider(
                        1, 40, value=_defaults(_first)["steps"], step=1,
                        label="Steps",
                        info="Adjusted automatically to the model you pick.")
                    seed = gr.Number(value=-1, precision=0,
                                     label="Seed (-1 = random)")
                with gr.Row():
                    run = gr.Button("🖼️ Extend the image", variant="primary",
                                    scale=3)
                    stop = gr.Button("⏹️ Cancel", variant="stop", scale=1)
                status = gr.Markdown("")
            with gr.Column(scale=4):
                result = gr.Image(label="Result", type="filepath",
                                  buttons=widgets.IMAGE_BUTTONS)
                again = gr.Button("♻️ Extend the result again", size="sm")
                log = gr.Textbox(label="Log", lines=12, autoscroll=True,
                                 elem_classes="log-box")

        # --- Aperçu du plan (dimensions) ---
        def _preview(img, dirs, amt):
            if img is None:
                return ""
            p = op.plan(img.size, (dirs or "").split("|"), float(amt))
            return f"**New size:** {op.describe(p)}"

        for comp in (image, direction, amount):
            comp.change(_preview, inputs=[image, direction, amount],
                        outputs=[plan_md])

        # Le choix du modèle change tout : nombre d'étapes, chemin utilisé, et
        # donc quels réglages ont encore un sens.
        def _on_model(mid):
            dd = _defaults(mid)
            if dd["edit"]:
                note = ("✅ **Editing model** — the canvas is handed to it as "
                        "a reference along with an extension instruction: it "
                        "*sees* the image and continues the scene.")
            else:
                note = ("⚠️ **Model without editing** — img2img fallback: it "
                        "does not see the image, it gets a noised latent and "
                        "**reinvents** instead of continuing. The result is "
                        "often incoherent. Prefer Flux.2 Klein.")
            return (gr.update(value=dd["steps"]),
                    gr.update(interactive=not dd["edit"]),
                    gr.update(value="neutral" if dd["edit"] else "edge"),
                    gr.update(value=note))

        model.change(_on_model, inputs=[model],
                     outputs=[steps, strength, fill, mode_md])

        def do_outpaint(img, dirs, amt, model_id, prompt_txt, fill_v, strength_v,
                        feather_v, tone_v, steps_v, seed_v):
            if img is None:
                raise gr.Error(t("Load an image to extend."))
            if not model_id:
                raise gr.Error(t("Pick a model."))
            p = op.plan(img.size, (dirs or "").split("|"), float(amt))
            if not any(p[d] for d in op.DIRECTIONS):
                raise gr.Error(t("No direction selected."))

            settings.ensure_dirs()
            d = _defaults(model_id)
            canvas = op.build_canvas(img, p, fill=fill_v or "edge")
            canvas_path = settings.TMP_DIR / "outpaint_init.png"
            canvas.save(canvas_path)

            try:
                s = int(seed_v)
            except (TypeError, ValueError):
                s = -1
            if s < 0:
                s = random.randint(0, 2**31 - 1)

            # ------------------------------------------------------------------
            # DEUX CHEMINS RADICALEMENT DIFFÉRENTS.
            #
            # Modèle d'ÉDITION (Flux.2 Klein) -> la toile part en
            # RÉFÉRENCE (-r) avec une consigne d'extension explicite. Le modèle
            # REGARDE l'image via son encodeur vision : il sait ce qu'il prolonge.
            # C'est la seule façon d'obtenir une extension qui ait du sens.
            #
            # Modèle SANS édition -> img2img (-i) : le modèle ne voit rien, il ne
            # reçoit qu'un latent bruité. Il réinvente au lieu de prolonger. On le
            # garde en repli, mais le résultat est médiocre par construction.
            # ------------------------------------------------------------------
            instr = op.instruction(p, prompt_txt)
            mask_path = None
            mf = None
            if not d["edit"]:
                # Masque : seulement si CE binaire sd-cli connaît l'option.
                mf = sdcpp.mask_flag(settings.find_sd_cli())
                if mf:
                    mask_path = settings.TMP_DIR / "outpaint_mask.png"
                    op.build_mask(p, feather=int(feather_v)).save(mask_path)

            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    kw = dict(
                        model_id=model_id, negative="", steps=int(steps_v),
                        cfg_scale=d["cfg_scale"],
                        width=p["width"], height=p["height"], seed=s,
                        batch_count=1, sampler=d["sampler"],
                        schedule=d["schedule"], flow_shift=d["flow_shift"],
                        log=q.put, save_prompt=False)
                    if d["edit"]:
                        # Édition : pilotée par le prompt, sans force ni init.
                        kw.update(prompt=instr, ref_image=[canvas_path])
                    else:
                        kw.update(prompt=prompt_txt or "",
                                  init_image=canvas_path,
                                  strength=float(strength_v),
                                  mask_image=mask_path)
                    outs = gen_engine.generate(**kw)
                    state["outs"] = [str(x) for x in outs]
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs = [f"Plan: {op.describe(p)}", f"Seed: {s}",
                    f"Border fill: {fill_v or 'edge'}",
                    (f"Mode: EDITING (-r + instruction) — the model sees "
                     f"the image"
                     if d["edit"] else
                     "Mode: img2img (-i) — model without editing, uncertain "
                     "result" + (f" · mask {mf}" if mf else "")),
                    f"Model: {model_id} · {d['sampler']} · "
                    f"cfg {d['cfg_scale']} · {int(steps_v)} steps"]
            if d["edit"]:
                logs.append(f"Instruction: {instr[:160]}…")
            yield t("⏳ Extending…"), gr.update(), "\n".join(logs)
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), gr.update(), "\n".join(logs[-400:])

            if "err" in state or not state.get("outs"):
                logs.append(f"\n[ERROR] {state.get('err', 'no output')}")
                yield (t("❌ Failed — see the log."), gr.update(),
                       "\n".join(logs))
                return

            # 1) recalage de tonalité : le modèle re-rend toute la toile plus
            #    contrastée ; sans ça l'original recollé fait un rectangle terne.
            # 2) recollage de l'original : la zone d'origine reste intacte.
            from PIL import Image as _PI
            gen = _PI.open(state["outs"][0])
            gen = op.match_tone(gen, img, p, amount=float(tone_v))
            final = op.composite_back(gen, img, p, feather=int(feather_v))
            out_path = settings.OUTPUT_DIR / \
                f"outpaint-{time.strftime('%Y%m%d-%H%M%S')}.png"
            final.save(out_path)
            _sidecar(out_path, p, model_id, prompt_txt, s, strength_v,
                     feather_v, steps_v, d, fill_v, tone_v, mf)
            logs.append(f"\n✅ Tone matched ({float(tone_v):.1f}), then the "
                        f"original pasted back (feather "
                        f"{int(feather_v)} px) → "
                        f"{out_path.name}")
            yield (t("✅ Extended: {n}").format(n=out_path.name),
                   gr.update(value=str(out_path)), "\n".join(logs))

        evt = run.click(
            do_outpaint,
            inputs=[image, direction, amount, model, prompt, fill, strength,
                    feather, tone, steps, seed],
            outputs=[status, result, log])
        widgets.stop_into_status(stop, gen_engine.cancel, status, [evt])

        # Enchaîner : le résultat redevient l'image d'entrée (extensions
        # successives, comme dans Midjourney).
        def _again(path):
            if not path:
                return gr.update(), t("Nothing to extend again.")
            from PIL import Image as _PI
            return (gr.update(value=_PI.open(path)),
                    t("Result reloaded — pick a direction."))

        again.click(_again, inputs=[result], outputs=[image, status])

        # Réception d'une image envoyée depuis un onglet de génération.
        if pending_outpaint is not None and tabs is not None:
            def _consume(pend):
                if not pend:
                    return gr.update(), None, gr.update(), gr.update()
                from PIL import Image as _PI
                im = _PI.open(pend)
                top = (gr.Tabs(selected=tab_id) if parent_tabs is not None
                       else gr.update())
                return gr.update(value=im), None, gr.update(value=""), top

            tabs.select(_consume, inputs=[pending_outpaint],
                        outputs=[image, pending_outpaint, status,
                                 parent_tabs if parent_tabs is not None
                                 else status])


def _sidecar(out_path, p, model_id, prompt_txt, seed, strength_v, feather_v,
             steps_v, d, fill_v=None, tone_v=None, mask_used=None) -> None:
    """Journal .txt à côté du PNG, comme pour les images et les GLB."""
    lines = [
        f"File: {out_path.name}",
        f"Model: {model_id}",
        f"Prompt: {prompt_txt or '(none)'}",
        f"Extension: {op.describe(p)}",
        f"Seed: {seed}",
        f"Remplissage: {fill_v or 'edge'}",
        f"Engine mask: {mask_used or 'non'}",
        f"Force: {strength_v}",
        f"Fondu: {int(feather_v)} px",
        f"Tone matching: {tone_v}",
        f"Steps: {int(steps_v)}  ·  Sampler: {d['sampler']}  ·  "
        f"CFG: {d['cfg_scale']}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    try:
        out_path.with_suffix(".txt").write_text("\n".join(lines),
                                                encoding="utf-8")
    except OSError:
        pass
