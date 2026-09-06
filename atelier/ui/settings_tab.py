"""Onglet Réglages.

CE QUE CET ONGLET REFUSE DE FAIRE : poser des questions auxquelles personne
ne peut répondre. « Flash attention ? » « Convolution directe ? » « Budget
VRAM du graphe ? » — une case qu'on ne sait pas cocher n'est pas un réglage,
c'est une source de doute. Il y en avait une vingtaine, réparties sur quatre
onglets ; on ne savait ni par où commencer ni ce qu'on risquait.

Trois principes, dans cet ordre :

1. **La machine décide ce que la machine sait.** Quantification, offload,
   tiling, flash-attention : tout se déduit du matériel détecté. On l'affiche
   en clair — non pour le faire régler, mais pour dire ce qui a été décidé, et
   en CONSÉQUENCES observables plutôt qu'en noms d'options.
2. **Il reste UNE question, et c'est un goût** : plus de marge mémoire, ou plus
   de détail ? Le matériel ne peut pas y répondre à votre place. Un curseur à
   trois crans, et c'est tout.
3. **Quand on ne sait pas, on mesure.** Le placement multi-GPU ne se devine pas
   (il dépend du lien PCIe autant que de la mémoire), donc un bouton le mesure
   au lieu de demander de parier.

Le reste — les options brutes de sd.cpp — vit sous un seul repli « Expert »,
annoncé comme facultatif.

Il n'y a PAS de bouton « Enregistrer » : chaque changement s'applique aussitôt
et le dit. Un bouton d'enregistrement est une occasion de plus de se demander
si ça a été pris en compte — et il n'existait déjà pas pour le thème, ce
qui rendait le reste ambigu.
"""
from __future__ import annotations

import json
import queue
import threading

import gradio as gr

from .. import benchmark, diagnostics, hardware, settings
from ..engine import engine_build_source, resident_engine
from ..i18n import t
from . import widgets

QUANTS = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
          "Q6_K", "Q8_0"]

# Préfixe des confirmations. Elles sont au passé et concrètes — « appliqué »,
# pas « enregistré » : ce qu'on veut savoir, c'est que c'est FAIT.
_OK = "✅ "


def _resident_reason() -> str:
    """Pourquoi le moteur résident n'est pas proposable — ou "" s'il l'est.

    Première version : la case disparaissait purement et simplement. Résultat,
    on cherche dans les Réglages une case dont on vient de lire la description,
    sans jamais savoir ce qui manque. Une option absente doit dire ce qui
    l'empêche, et le geste qui la débloque.
    """
    server = resident_engine()
    if server is None:
        return t("⚠️ **Resident engine unavailable**: the file "
                 "`atelier/engine/sdserver.py` is missing. Your copy of the "
                 "application is incomplete — download the archive again, "
                 "**close the application**, then re-extract it.")
    if server.available():
        return ""
    if engine_build_source() == "custom-ci":
        # Le build maison a été retiré du projet : il n'empaquetait qu'un
        # binaire sur les deux, et entretenir une deuxième chaîne de
        # compilation pour ça ne valait pas son prix.
        return t("⚠️ **Resident engine unavailable**: `sd-server` is not in "
                 "`bin/`. Your engine comes from the project's former "
                 "in-house build, which only packaged `sd.exe` and no longer "
                 "exists. Run `update-engine.bat` to switch to the official "
                 "binary, which contains both.")
    return t("⚠️ **Resident engine unavailable**: `sd-server` is not in "
             "`bin/`. Run `update-engine.bat` to reinstall the complete "
             "engine.")


def _said(msg: str):
    """Affiche une confirmation — et fait EXISTER la ligne à ce moment-là.

    Un Markdown vide n'est pas invisible dans Gradio : son conteneur garde son
    cadre. Une bande vide en permanence au-dessus des réglages, c'est du bruit
    qui ressemble à un message qu'on n'arrive pas à lire. La ligne apparaît donc
    au premier changement, et pas avant.
    """
    return gr.update(value=msg, visible=True)

_STRATEGY_SAID = {
    "single": "Everything will run on a single card.",
    "encoder": "The 2nd card will read your text; the image stays on the first.",
    "autofit": "Automatic memory placement enabled. Recent engines compute on one GPU.",
}


def _gpu_choices() -> list[tuple[str, int]]:
    return [(f"#{g.index} — {g.label()}", g.index)
            for g in hardware.detect_gpus()]


def _save(**changes) -> dict:
    """Applique des changements aux préférences et renvoie le tout."""
    p = settings.load_prefs()
    # Nettoyage des anciens réglages moteur (serveur/ComfyUI, retirés).
    for stale in ("engine", "use_sd_server", "sd_server_port", "comfyui_port"):
        p.pop(stale, None)
    p.update(changes)
    settings.save_prefs(p)
    return p


def _strategy_of(prefs: dict) -> str:
    if prefs.get("auto_fit"):
        return "autofit"
    eg = prefs.get("encoder_gpu_index")
    if eg is not None and eg != prefs.get("gpu_index"):
        return "encoder"
    return "single"


def _apply_strategy_prefs(choice: str, gpu_idx) -> dict:
    """Traduit le choix en préférences (les trois stratégies s'excluent)."""
    gpus = hardware.detect_gpus()
    sel = gpu_idx if gpu_idx is not None else (
        max(gpus, key=lambda x: x.vram_gb).index if gpus else None)
    other = next((g.index for g in gpus if g.index != sel), None)
    split = (choice == "encoder") and sel is not None and other is not None
    return _save(
        auto_fit=(choice == "autofit"),
        encoder_gpu_index=other if split else None,
        # Résidence explicite : le calcul ET les poids de l'encodeur vont sur
        # la 2e carte. Sans cela, un encodeur « sur la 2e carte » relirait ses
        # poids depuis la RAM à chaque image, à travers le PCIe.
        params_backend=(f"diffusion=cuda{sel},vae=cuda{sel},te=cuda{other}"
                        if split else ""))


# --------------------------------------------------------------------------- #
#  Ce que la machine a décidé — en français, pas en noms d'options
# --------------------------------------------------------------------------- #
def _headline(prefs: dict | None = None) -> str:
    """Carte lisible : le matériel, puis ce qui en découle.

    On ne recopie surtout pas la liste des flags : `vae_tiling`, `clip_on_cpu`
    ne veulent rien dire pour qui ne connaît pas sd.cpp. Chacun est traduit en
    conséquence observable.
    """
    prefs = prefs if prefs is not None else settings.load_prefs()
    bias = hardware.bias_from_prefs(prefs)
    prof = hardware.biased_profile(bias, prefs.get("gpu_index"))
    gpus = hardware.detect_gpus()

    if prof.gpu is None:
        return t("### ⚠️ No NVIDIA card detected\nThe app will run on the "
                 "processor: that is **very slow** (minutes per image). Check "
                 "your drivers, or type `nvidia-smi` in a terminal.")

    lines = [t("### Your hardware"),
             t("**{name}** — {vram} GB of video memory · {ram} GB of RAM"
               ).format(name=prof.gpu.name, vram=f"{prof.gpu.vram_gb:.0f}",
                        ram=f"{prof.ram_gb:.0f}")]
    if len(gpus) > 1:
        others = ", ".join(g.name for g in gpus if g.index != prof.gpu.index)
        lines.append(t("Second card available: {other}.").format(
            other=others))

    lines += ["", t("**What the app does with it, without asking you anything:**")]
    lines.append(t("- the image model is loaded as `{quant}` — the best "
                   "trade-off that fits in {vram} GB;").format(
                       quant=prof.quant, vram=f"{prof.gpu.vram_gb:.0f}"))
    lines.append(t("- your text is analysed as `{enc}`, **kept in RAM**: it "
                   "takes no room on the card;").format(
                       enc=prof.enc_quant))
    lines.append(
        t("- “flash attention” acceleration is on (your card supports it);") if prof.diffusion_fa else
        t("- “flash attention” stays off: your card lacks the units for it, "
          "turning it on would slow things down;"))
    lines.append(
        t("- the final image is assembled in pieces, so the card is not "
          "saturated at the last moment.") if prof.vae_tiling else
        t("- the final image is assembled in one piece: you have the room, so "
          "no seams."))
    return "\n".join(lines)


def _bias_note(bias: str, prefs: dict | None = None) -> str:
    """Ce que le cran choisi change — une raison, puis un chiffre."""
    prefs = prefs if prefs is not None else settings.load_prefs()
    spec = hardware.BIASES.get(bias) or hardware.BIASES["balanced"]
    idx = prefs.get("gpu_index")
    prof = hardware.biased_profile(bias, idx)
    ref = hardware.biased_profile("balanced", idx)
    detail = t(spec["why"])
    if prof.quant != ref.quant:
        detail += t("  \n→ model loaded as `{quant}` instead of `{ref}`."
                    ).format(quant=prof.quant, ref=ref.quant)
    else:
        detail += t("  \n→ model loaded as `{quant}`.").format(quant=prof.quant)
    return detail


def build_settings_tab():
    with gr.Tab("⚙️ Settings"):
        prefs = settings.load_prefs()
        gpus = hardware.detect_gpus()
        multi_gpu = len(gpus) > 1

        headline = gr.Markdown(_headline(prefs))
        status = gr.Markdown("", elem_classes="feedback", visible=False)

        # ------------------------------------------------------------------ #
        #  LE réglage : marge mémoire ou détail
        # ------------------------------------------------------------------ #
        gr.Markdown(t(
            "---\n### The one question we ask you\nEverything else is "
            "computed from your card. This one cannot be, because it is a "
            "preference: do you want **headroom** (it always fits) or "
            "**detail** (finer, but tighter on memory)?"))
        _bias = hardware.bias_from_prefs(prefs)
        bias = gr.Radio(
            [(t(spec["label"]), key) for key, spec in hardware.BIASES.items()],
            value=_bias, label="Priority", show_label=False)
        bias_note = gr.Markdown(_bias_note(_bias, prefs))
        # Le vrai mode d'emploi : partir du SYMPTÔME. C'est ainsi qu'on arrive
        # sur cette page — pas en se demandant « quelle quantification ? ».
        # Deux des trois réponses renvoient ailleurs, et c'est volontaire :
        # laisser croire que tout se règle ici serait la même impasse.
        gr.Markdown(t(
            "> **Something specific going wrong?**  \n> *“Out of memory / "
            "generation stops”* → pick **🪶 More memory headroom** above.  \n> "
            "*“It is too slow”* → not settled here but in the generation tab: "
            "lower the **step count** and the **image size**, which weigh far "
            "more.  \n> *“My images look dull”* → not here either: that is "
            "the **prompt** and the **styles**, not a hardware setting."))

        # ------------------------------------------------------------------ #
        #  Deux cartes : on ne fait pas parier, on propose de mesurer
        # ------------------------------------------------------------------ #
        gpu = gr.Dropdown(
            label="Card used for generating", choices=_gpu_choices(),
            value=prefs.get("gpu_index"), visible=multi_gpu,
            info=t("The most powerful one is used by default."))
        if multi_gpu:
            gr.Markdown(t(
                "---\n### You have two cards\nThere is no universally right "
                "answer: it depends as much on the second card's **PCIe "
                "slot** as on its memory. Rather than make you guess, the app "
                "can **measure**."))
            strategy = gr.Radio(
                [(t("Everything on one card — the most reliable"), "single"),
                 (t("The 2nd card handles the text — frees memory for the "
                    "image"), "encoder"),
                 (t("Place weights automatically — GPU, RAM, then disk"),
                  "autofit")],
                value=_strategy_of(prefs), label="Split",
                show_label=False)
            tools_gpu = gr.Dropdown(
                label="Card for the prompt enhancer",
                choices=[(t("The same one as for the image"), None)] + _gpu_choices(),
                value=prefs.get("text_gpu_index"))
        else:
            strategy = gr.State(_strategy_of(prefs))
            tools_gpu = gr.State(prefs.get("text_gpu_index"))

        # ------------------------------------------------------------------ #
        #  Mesurer plutôt que deviner
        # ------------------------------------------------------------------ #
        gr.Markdown(t(
            "---\n### 🧪 When in doubt, measure\nThe app generates the **same "
            "image** {n} times per configuration (plus a first one thrown "
            "away, the time for everything to load) and keeps the median. It "
            "changes **no setting**: it tells you which is fastest, then you "
            "decide."
        ).format(n=benchmark.MEASURED_RUNS))
        with gr.Row():
            bench_btn = gr.Button(t("⏱️ Measure on my machine"),
                                  variant="primary")
            apply_bench = gr.Button(t("Apply the fastest"))
            bench_stop = gr.Button(t("⏹️ Stop"), variant="stop")
        bench_status = gr.Markdown("")
        with gr.Accordion(t("Measurement details (log and report)"),
                          open=False):
            bench_file = gr.File(label="JSON report", interactive=False)
            bench_log = gr.Textbox(label="Test log", lines=10,
                                   autoscroll=True, elem_classes="log-box")
            system_md = gr.Markdown(diagnostics.summary_markdown())
            report_btn = gr.Button(t("📋 Export the system report"),
                                   size="sm")

        # ------------------------------------------------------------------ #
        #  Expert : les options brutes de sd.cpp, sous UN seul repli
        # ------------------------------------------------------------------ #
        with gr.Accordion(
                t("🔧 Expert — raw sd.cpp options (optional)"),
                open=False):
            gr.Markdown(t(
                "⚠️ **Nothing here is required.** These options exist because "
                "sd.cpp exposes them, not because you should touch them. They "
                "are set by measuring, not by guessing — and the slider above "
                "already covers the usual cases. Touching this section "
                "**turns off automatic tuning**."))

            gr.Markdown(t("**Forced quantization** — “auto” = let the app "
                          "decide from the card."))
            with gr.Row():
                quant = gr.Dropdown(label="Image model",
                                    choices=["auto"] + QUANTS,
                                    value=prefs.get("quant") or "auto")
                enc_quant = gr.Dropdown(label="Text analysis",
                                        choices=["auto"] + QUANTS,
                                        value=prefs.get("enc_quant") or "auto")
            f = prefs.get("flags", {})
            gr.Markdown(t("**Engine memory options.**"))
            with gr.Row():
                fa = gr.Checkbox(value=f.get("diffusion_fa", True),
                                 label="Flash attention")
                offload = gr.Checkbox(value=f.get("offload_to_cpu", True),
                                      label="Model kept in RAM")
                tiling = gr.Checkbox(value=f.get("vae_tiling", True),
                                     label="Image assembled in pieces")
            with gr.Row():
                clip_cpu = gr.Checkbox(value=f.get("clip_on_cpu", False),
                                       label="Text on the processor")
                vae_cpu = gr.Checkbox(value=f.get("vae_on_cpu", False),
                                      label="Assembly on the processor")

            gr.Markdown(t(
                "---\n**Cache between steps** — reuses computations from one "
                "diffusion step to the next. Only pays off above ~10 steps; "
                "our models run 4 to 8, so **leave it off** unless a "
                "measurement says otherwise."))
            with gr.Row():
                cache_mode = gr.Dropdown(
                    [(t("Disabled (recommended)"), ""),
                     ("easycache", "easycache"), ("dbcache", "dbcache"),
                     ("taylorseer", "taylorseer"), ("cache-dit", "cache-dit"),
                     ("spectrum", "spectrum")],
                    value=prefs.get("cache_mode", ""), label="Cache mode")
                cache_opt = gr.Textbox(
                    value=prefs.get("cache_option", ""),
                    label="Option (blank = defaults)",
                    placeholder="ex. threshold=0.2")

            gr.Markdown(t(
                "---\n**Direct convolution** — removes a large intermediate "
                "buffer. The memory gain is certain; the speed effect is "
                "**unpredictable** (sometimes better, sometimes worse). To be "
                "timed, not ticked blindly."))
            with gr.Row():
                conv_diff = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_diffusion")),
                    label="Direct convolution — image model")
                conv_vae = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_vae")),
                    label="Direct convolution — assembly")

            gr.Markdown(t(
                "---\n**Splitting the computation** — lets the engine cut its "
                "graph to fit the available memory. Recent engines handle "
                "segmentation and weight prefetch automatically. An explicit "
                "budget leaves room for other applications. The HD tab "
                "already sets its own budget."))
            with gr.Row():
                max_vram = gr.Dropdown(
                    [(t("Engine default"), ""),
                     (t("Auto — free memory minus 1 GB"), "auto"),
                     (t("Managed memory budget: 6 GB"), "6"),
                     (t("Managed memory budget: 8 GB"), "8"),
                     (t("Managed memory budget: 10 GB"), "10")],
                    value=prefs.get("max_vram", ""), allow_custom_value=True,
                    label="Memory budget for the computation")
                stream_layers = gr.Checkbox(
                    value=bool(prefs.get("stream_layers")),
                    label="Layer streaming from RAM (older engines)",
                    info=t("Requires weights in RAM. Recent engines stream "
                           "automatically; this switch only affects older builds."))

            gr.Markdown(t(
                "---\n**Resident engine** — today the engine starts, reads "
                "the model, makes the image and exits: the loading is paid "
                "again for **every** image. Ticked, the model stays loaded "
                "between generations. That is pure gain when you generate one "
                "image at a time to refine a prompt.\n\nIn exchange: **no "
                "preview while it computes** (the image arrives all at once), "
                "and the model occupies the card permanently — Toolkit tools "
                "unload it by themselves when they need the GPU. Reference "
                "editing also reuses the model when the server supports it. "
                "LoRAs and the HD pass use the command-line engine. Cancelling "
                "stops the resident process; the next image reloads it."))
            _no_resident = _resident_reason()
            gr.Markdown(_no_resident, visible=bool(_no_resident))
            resident = gr.Checkbox(
                value=bool(prefs.get("resident_engine")),
                label="Keep the model loaded between images",
                visible=not _no_resident)

            # Confirmation LOCALE : la ligne d'état du haut est hors de l'écran
            # quand on coche quelque chose ici. Un réglage qui s'applique sans
            # rien dire de visible, c'est un réglage dont on doute.
            expert_status = gr.Markdown("", elem_classes="feedback", visible=False)

            combo = hardware.rtx3060_1080ti_combo() if multi_gpu else None
            combo_btn = gr.Button(t("⚡ RTX 3060 + GTX 1080 Ti profile"),
                                  visible=bool(combo))

        # ------------------------------------------------------------------ #
        #  Ce qui n'a rien à voir avec la génération
        # ------------------------------------------------------------------ #
        with gr.Accordion(t("🌍 Theme and accounts"), open=False):
            with gr.Row():
                theme_dd = gr.Dropdown(
                    [(t("Light"), "light"), (t("Dark"), "dark")],
                    value=prefs.get("theme", "light"),
                    label="🎨 Theme (restart required)")
            hf_ep = gr.Textbox(
                value=prefs.get("hf_endpoint", "https://huggingface.co"),
                label="Hugging Face endpoint (optional mirror)")
            civitai_tok = gr.Textbox(
                value=prefs.get("civitai_token", ""),
                label="Civitai token (optional — gated LoRAs)",
                type="password")
            account_status = gr.Markdown("", elem_classes="feedback", visible=False)

        # ================================================================== #
        #  Câblage — tout s'applique à la volée
        # ================================================================== #
        def _apply_bias(choice, gpu_idx):
            prof = hardware.biased_profile(choice, gpu_idx)
            # « Équilibré » RESTE l'automatique : c'est exactement ce qu'il
            # calcule. Les deux autres crans sont un choix explicite, donc ils
            # figent quant et flags — sinon le profil automatique les
            # réécrirait au prochain démarrage et le réglage aurait l'air de
            # « ne pas tenir ».
            p = _save(auto_optimize=(choice == "balanced"),
                      quant=None if choice == "balanced" else prof.quant,
                      enc_quant=None if choice == "balanced" else prof.enc_quant,
                      flags=prof.flags())
            return (_headline(p), _bias_note(choice, p),
                    _said(_OK + t("Priority applied: **{label}**.").format(
                        label=t(hardware.BIASES[choice]["label"]))))

        bias.change(_apply_bias, inputs=[bias, gpu],
                    outputs=[headline, bias_note, status])

        def _apply_gpu(idx, choice):
            _save(gpu_index=idx)
            p = _apply_strategy_prefs(choice, idx)
            return _headline(p), _said(_OK + t(
                "Generation card: #{idx}.").format(idx=idx))

        gpu.change(_apply_gpu, inputs=[gpu, strategy],
                   outputs=[headline, status])

        if multi_gpu:
            def _apply_strategy(choice, idx):
                p = _apply_strategy_prefs(choice, idx)
                return _headline(p), _said(_OK + t(_STRATEGY_SAID[choice]))

            strategy.change(_apply_strategy, inputs=[strategy, gpu],
                            outputs=[headline, status])

            def _apply_tools_gpu(v):
                _save(text_gpu_index=v)
                return _said(_OK + t("Enhancer card saved."))

            tools_gpu.change(_apply_tools_gpu, inputs=[tools_gpu],
                             outputs=[status])

        # ---- Expert : chaque contrôle s'applique seul --------------------- #
        def _apply_expert(q, eq, fa_, off, til, clip, vae, cm, co,
                          cd, cv, mv, sl):
            p = _save(auto_optimize=False,
                      quant=None if q == "auto" else q,
                      enc_quant=None if eq == "auto" else eq,
                      flags={"diffusion_fa": bool(fa_),
                             "offload_to_cpu": bool(off),
                             "vae_tiling": bool(til),
                             "clip_on_cpu": bool(clip),
                             "vae_on_cpu": bool(vae)},
                      cache_mode=cm or "", cache_option=(co or "").strip(),
                      conv_direct_diffusion=bool(cd),
                      conv_direct_vae=bool(cv),
                      max_vram=(mv or "").strip(),
                      stream_layers=bool(sl))
            return (_headline(p),
                    _bias_note(hardware.bias_from_prefs(p), p),
                    _said(_OK + t("Expert setting applied (automatic tuning "
                                  "off).")))

        _expert = [quant, enc_quant, fa, offload, tiling, clip_cpu, vae_cpu,
                   cache_mode, cache_opt, conv_diff, conv_vae, max_vram,
                   stream_layers]
        for comp in _expert:
            comp.change(_apply_expert, inputs=_expert,
                        outputs=[headline, bias_note, expert_status])

        # Le moteur résident n'est PAS un réglage de sd.cpp : il ne doit donc
        # pas basculer l'application en mode manuel comme le fait `_apply_expert`.
        def _apply_resident(on):
            _save(resident_engine=bool(on))
            if on:
                return _said(_OK + t("The model will stay loaded between "
                                     "images. The first load will take as "
                                     "long as it always has."))
            server = resident_engine()
            if server is not None:
                server.stop()
            return _said(_OK + t("Resident engine off, the card's memory is "
                                 "released."))

        resident.change(_apply_resident, inputs=[resident],
                        outputs=[expert_status])

        # ---- Theme, accounts ---------------------------------------------- #
        def _apply_theme(th):
            _save(theme="dark" if th == "dark" else "light")
            return _said(_OK + t("Theme saved. **Restart the app** to apply "
                                 "it."))

        def _apply_endpoint(v):
            _save(hf_endpoint=v or "https://huggingface.co")
            return _said(_OK + t("Endpoint saved."))

        def _apply_token(v):
            _save(civitai_token=(v or "").strip())
            return _said(_OK + t("Civitai token saved."))

        theme_dd.change(_apply_theme, inputs=[theme_dd], outputs=[account_status])
        hf_ep.change(_apply_endpoint, inputs=[hf_ep], outputs=[account_status])
        civitai_tok.change(_apply_token, inputs=[civitai_tok],
                           outputs=[account_status])

        # ---- Mesure -------------------------------------------------------- #
        _bench_stop = threading.Event()

        def _verdict(data: dict) -> str:
            """Dire ce qui a été mesuré, pas seulement qui gagne."""
            rows = [r for r in data.get("results", []) if r.get("ok")]
            if not rows:
                return t("❌ No configuration could be measured (see the log).")
            lines = [f"- **{r['label']}** — {r['seconds']:.2f} s "
                     f"(± {r.get('spread_seconds', 0):.2f} s)" for r in rows]
            best = data.get("recommended_mode") or data.get("fastest_model")
            head = t("✅ Median over {n} runs · fastest: **{best}**"
                     ).format(n=data.get("measured_runs", "?"), best=best)
            return head + "\n" + "\n".join(lines)

        def _do_bench():
            """Diffuse le journal au fil de l'eau.

            Rendre le résultat d'un bloc à la fin laisserait l'interface muette
            plusieurs minutes, sans dire si ça avance ni comment l'arrêter.
            """
            _bench_stop.clear()
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    state["path"] = benchmark.run_hardware_benchmark(
                        log=q.put, cancel=_bench_stop.is_set)
                except Exception as exc:  # noqa: BLE001
                    state["err"] = exc
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            yield t("⏳ Measuring…"), gr.update(), ""
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), gr.update(), "\n".join(logs[-500:])
            tail = "\n".join(logs[-500:])
            if isinstance(state.get("err"), benchmark.Cancelled):
                yield (t("⏹️ Measurement interrupted — no setting changed."),
                       gr.update(), tail)
                return
            if "err" in state:
                yield f"❌ {state['err']}", gr.update(), tail
                return
            path = state["path"]
            yield (_verdict(json.loads(path.read_text(encoding="utf-8"))),
                   str(path), tail)

        _bench_evt = bench_btn.click(
            _do_bench, outputs=[bench_status, bench_file, bench_log])

        def _cancel_bench() -> str:
            # On POSE le drapeau au lieu de tuer le fil : la génération en
            # cours va au bout et la mesure s'arrête proprement ensuite.
            # Couper au milieu laisserait un tir à moitié chronométré.
            _bench_stop.set()
            return t("⏹️ Stop requested — the run in progress will finish first.")

        widgets.stop_into_log(bench_stop, _cancel_bench, bench_log,
                              [_bench_evt])

        def _apply_report(raw):
            if not raw:
                return gr.update(), _said(t("❌ Run the measurement first."))
            try:
                mode = benchmark.apply_recommendation(getattr(raw, "name", raw))
            except Exception as exc:  # noqa: BLE001
                return gr.update(), _said(f"❌ {exc}")
            return _headline(), _said(_OK + t(
                "Measured configuration applied: **{mode}**.").format(
                    mode=mode))

        apply_bench.click(_apply_report, inputs=[bench_file],
                          outputs=[headline, status])

        def _system_report():
            return (diagnostics.summary_markdown(),
                    str(diagnostics.write_system_report()))

        report_btn.click(_system_report, outputs=[system_md, bench_file])

        def _apply_combo():
            preset = hardware.rtx3060_1080ti_prefs()
            p = _save(**preset)
            return (_headline(p), gr.update(value=preset["gpu_index"]),
                    gr.update(value="encoder"),
                    _said(_OK + t("Two-card profile applied: the RTX 3060 "
                                  "draws, the 1080 Ti reads your text.")))

        combo_btn.click(_apply_combo,
                        outputs=[headline, gpu, strategy, status])
