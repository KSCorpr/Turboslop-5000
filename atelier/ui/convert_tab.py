"""Onglet « Convertir en GGUF » : quantifie un modèle (checkpoint / safetensors
/ diffusion) vers un GGUF plus léger, via sd.cpp (--mode convert).

100% CPU, aucune diffusion : c'est une transformation des poids. But : faire
tenir un modèle communautaire sur une carte modeste (quant selon la VRAM).
"""
from __future__ import annotations

import queue
import threading

import gradio as gr

from .. import settings
from ..engine import generate as gen_engine
from ..engine import sdcpp
from ..i18n import t
from . import widgets

# (valeur sd.cpp, libellé). Du plus lourd/fidèle au plus léger/agressif.
# Les k-quants (q*_k) demandent un moteur récent ; en cas de refus, le journal
# le dira et on retombe sur q5_1 / q8_0. Défaut : q4_k (bon compromis VRAM).
# (libellé affiché, valeur sd.cpp). sd.cpp « --mode convert » n'accepte QUE ces
# formats (doc quantization_and_gguf.md) : PAS de k-quants (q4_k, q5_k, q6_k…),
# qui sont refusés par le moteur (« invalid weight format »). Du plus fidèle au
# plus léger.
QTYPES = [
    ("q5_1 — 5 bits · recommended (a good size/quality compromise)", "q5_1"),
    ("q8_0 — 8 bits · near lossless (larger)", "q8_0"),
    ("q5_0 — 5 bits", "q5_0"),
    ("q4_1 — 4 bits · light", "q4_1"),
    ("q4_0 — 4 bits · the lightest", "q4_0"),
    ("f16 — 16 bits (no loss, no saving either)", "f16"),
]
_DEFAULT_QTYPE = "q5_1"


def _suggest_name(model_name: str | None, qtype: str) -> str:
    if not model_name:
        return ""
    stem = model_name.rsplit(".", 1)[0]
    return f"{stem}-{qtype or _DEFAULT_QTYPE}.gguf"


def build_convert_tab():
    with gr.Tab("🔧 Convert to GGUF"):
        gr.Markdown(
            "### Quantize a model to GGUF\n"
            "Turns a **checkpoint / safetensors / diffusion** model into a "
            "lighter **GGUF**, so it fits on your card. This is **100% CPU** "
            "(no diffusion): a few minutes depending on the size and the "
            "disk. Once only — after that you reuse the GGUF.\n\n"
            f"1. Drop your model into **`{settings.CUSTOM_DIR}`** then "
            "**↻ Refresh**.  \n"
            "2. Pick the quantization: `q8_0` ≈ lossless → `q5_1` a good "
            "compromise → `q4_0` the lightest. *(sd.cpp only handles "
            "q8_0/q5_1/q5_0/q4_1/q4_0/f16 when converting — no k-quants.)*  \n"
            "3. **Convert**: the GGUF is written into that same "
            "`models/custom/` folder and becomes usable as a **local model** "
            "in the generation tabs (the “Refresh local files "
            "locaux »).")

        with gr.Row():
            src = gr.Dropdown(
                gen_engine.list_custom_models(), value=None, scale=3,
                label="Model to convert (in models/custom/)",
                allow_custom_value=False)
            refresh = gr.Button("↻ Refresh", size="sm", scale=1)
        with gr.Row():
            qtype = gr.Dropdown(QTYPES, value=_DEFAULT_QTYPE, scale=2,
                                label="Quantification cible")
            out_name = gr.Textbox(label="Name of the output GGUF file", scale=3,
                                  placeholder="e.g. my-model-q4_k.gguf")

        with gr.Row():
            run = gr.Button("🔧 Convertir", variant="primary", scale=3)
            stop = gr.Button("⏹️ Cancel", variant="stop", scale=1)
        status = gr.Markdown("")
        log = gr.Textbox(label="Log", lines=14, autoscroll=True,
                         elem_classes="log-box")

        # --- Comportements ---
        def _refresh():
            return gr.update(choices=gen_engine.list_custom_models())

        refresh.click(_refresh, outputs=[src])

        def _on_pick(name, qt):
            return gr.update(value=_suggest_name(name, qt))

        src.change(_on_pick, inputs=[src, qtype], outputs=[out_name])
        qtype.change(_on_pick, inputs=[src, qtype], outputs=[out_name])

        def do_convert(src_name, qt, out):
            sd_cli = settings.find_sd_cli()
            if sd_cli is None:
                raise gr.Error(t("Binaire sd-cli introuvable (install.bat / "
                                 "get_sdcpp.py)."))
            in_path = gen_engine.custom_path(src_name)
            if in_path is None:
                raise gr.Error(t("Pick a model to convert (dropped into "
                                 "models/custom/)."))
            out = (out or "").strip() or _suggest_name(src_name, qt)
            if not out.lower().endswith(".gguf"):
                out += ".gguf"
            out_path = settings.CUSTOM_DIR / out
            if out_path.resolve() == in_path.resolve():
                raise gr.Error(t("The output file must differ from the input."))

            cmd = sdcpp.build_convert_cmd(sd_cli, in_path, out_path, qt)
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    sdcpp.run(cmd, log=q.put)   # convert = CPU, pas de GPU épinglé
                    state["ok"] = out_path.is_file()
                except Exception as exc:  # noqa: BLE001
                    state["err"] = str(exc)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            yield t("⏳ Converting… (CPU, a few minutes)"), ""
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), "\n".join(logs[-500:])

            if "err" in state:
                logs.append(f"\n[ERROR] {state['err']}")
                yield (t("❌ Conversion failed — see the log."),
                       "\n".join(logs))
                return
            if not state.get("ok"):
                yield (t("⚠️ Finished but the output file was not found — see "
                         "the log (quantization refused by the engine?)."),
                       "\n".join(logs))
                return
            size_mb = out_path.stat().st_size / 1e6
            yield (t("✅ Converted: **{name}** ({size:.0f} MB) into "
                     "`models/custom/`. Usable through “Refresh local files” "
                     "in a generation tab.").format(
                        name=out, size=size_mb),
                   "\n".join(logs))

        evt = run.click(do_convert, inputs=[src, qtype, out_name],
                        outputs=[status, log])
        widgets.stop_into_status(stop, gen_engine.cancel, status, [evt])
