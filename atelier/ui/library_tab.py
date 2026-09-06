"""Onglet Catalogue de modèles : catalogue des modèles, recommandations selon le
matériel, téléchargement à la demande."""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry, settings
from ..i18n import t


def _card_md(model: registry.BaseModel, recos: dict[str, list[str]]) -> str:
    ready = registry.model_is_ready(model)
    status = (f"<span class='status-ok'>{t('● installed')}</span>" if ready
              else f"<span class='status-missing'>{t('○ not installed')}</span>")
    tags = " ".join(f"<span class='tag'>{t}</span>" for t in model.tags)
    reco = " · ".join(recos.get(model.id, []))
    return (f"<div class='model-card'><h3>{model.name} &nbsp; {status}</h3>"
            f"{tags}<p>{model.description}</p>"
            f"<small>{reco}</small></div>")


def build_library_tab():
    with gr.Tab("📚 Model Catalog"):
        gr.Markdown("### Base models\nOn-demand download. Quantization is "
                    "chosen automatically from your VRAM/RAM (changeable in "
                    "Settings).")
        gr.Markdown(
            "> ℹ️ The quantization shown (Settings) is a **target**. If the "
            "repo doesn't offer it, the closest available quant **below** it "
            "is downloaded (to fit your VRAM) — shown in the log and flagged "
            "after the download.")

        prefs = settings.load_prefs()
        models = registry.load_base_models(prefs)
        recos = registry.recommend(prefs)

        cards: list[gr.Markdown] = []
        log = gr.Textbox(label="Download log", lines=8,
                         autoscroll=True, elem_classes="log-box")

        for m in models:
            with gr.Row():
                with gr.Column(scale=5):
                    card = gr.Markdown(_card_md(m, recos))
                with gr.Column(scale=1, min_width=170):
                    btn = gr.Button("⬇️ Download", variant="primary")
                    del_btn = gr.Button("🗑️ Delete", size="sm")
            cards.append(card)

            def make_handler(model_id):
                def handler(progress=gr.Progress()):
                    p = settings.load_prefs()
                    model = registry.get_base_model(model_id, p)
                    lines: list[str] = []
                    for msg in downloader.download_model(model, log=lines.append):
                        lines.append(msg)
                        yield "\n".join(lines)
                    # Avertit visiblement si un quant a été remplacé par un repli.
                    if any("⚠️" in line and "indisponible" in line for line in lines):
                        gr.Warning(t("Quantization adjusted: the repo doesn't "
                                     "offer the target quant, fell back to "
                                     "the closest available (see the log)."))
                return handler

            def make_deleter(model_id):
                def deleter():
                    p = settings.load_prefs()
                    model = registry.get_base_model(model_id, p)
                    deleted = registry.delete_model(model, p)
                    msg = (t("🗑️ “{name}” deleted: {n} file(s) removed.").format(name=model.name, n=len(deleted))
                           if deleted else
                           t("Nothing to delete for “{name}” (not installed "
                             "or shared files).").format(name=model.name))
                    return _card_md(model, registry.recommend(p)), msg
                return deleter

            btn.click(make_handler(m.id), outputs=[log])
            del_btn.click(make_deleter(m.id), outputs=[card, log])

        refresh = gr.Button("↻ Refresh status")

        def refresh_cards():
            p = settings.load_prefs()
            r = registry.recommend(p)
            ups = [gr.update(value=_card_md(m, r))
                   for m in registry.load_base_models(p)]
            # Avec une seule carte, Gradio attend une valeur unique (pas une
            # liste), sinon la liste est passée telle quelle au Markdown.
            return ups[0] if len(ups) == 1 else ups

        refresh.click(refresh_cards, outputs=cards)

        gr.Markdown(
            "---\n*The tools (depth, background removal, SAM, prompt "
            "improver, enlargement) live in the Toolkit tab.*")
