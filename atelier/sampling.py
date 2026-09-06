"""Échantillonneurs et schedulers : la liste ET ce qu'il faut en penser.

Pourquoi une documentation par MODÈLE plutôt qu'un glossaire neutre : sur les
modèles DISTILLÉS, la moitié du menu est inutile, et pas un peu —
structurellement. Trois propriétés vérifiées dans le moteur l'expliquent, et
elles suffisent à classer tout le reste :

1. **Ce sont des modèles de FLOW MATCHING.** sd.cpp les fait tourner en
   `FLUX_FLOW_PRED` / `FluxFlowDenoiser` (Krea 2 avec un flow shift de 1,15).
   Les schedulers Karras et Exponential ont été conçus pour la diffusion EDM à
   prédiction d'epsilon : leur répartition de sigmas n'a pas le même sens ici.

2. **Ils sont DISTILLÉS à CFG 1.0.** Il n'y a donc aucun guidage à corriger, ce
   qui vide de leur objet toute la famille « CFG++ ». Et le prompt négatif est
   ignoré, quel que soit l'échantillonneur.

3. **Ils tournent en TRÈS PEU DE PAS** — 4 pour Flux.2 Klein, 8 pour Krea 2
   Turbo. Deux conséquences : les méthodes ANCESTRALES réinjectent du bruit à
   chaque pas et n'ont pas le temps de reconverger ; les méthodes MULTI-PAS
   doivent d'abord accumuler un historique d'évaluations, qui n'existe presque
   pas sur un budget aussi court.

Ces verdicts sont donc RAISONNÉS à partir des propriétés des modèles, pas tirés
d'un banc d'essai. Ils disent où porter ses essais, pas ce que vous allez
préférer : sur du rendu, l'œil tranche mieux qu'un principe.
"""
from __future__ import annotations

from . import i18n

# Niveaux de recommandation, du meilleur au pire.
BEST, OK, MEH, BAD = "best", "ok", "meh", "bad"

_MARK = {BEST: "⭐", OK: "", MEH: "△", BAD: "⚠️"}
_VERDICT = {
    BEST: "**Recommended** for this model.",
    OK: "Usable, with no clear advantage here.",
    MEH: "Poorly suited to this model.",
    BAD: "**Discouraged** with this model.",
}

# clé -> (libellé, résumé, avantage, inconvénient, {famille: niveau})
#
# `flux2` = Flux.2 Klein (4 pas) · `krea2` = Krea 2 Turbo (8 pas).
SAMPLERS: dict[str, tuple] = {
    "euler": (
        "Euler",
        "The baseline method: one step, one evaluation, no added noise.",
        "Predictable, reproducible, and the only one with nothing to “catch "
        "up” when steps are scarce. It is the default on every one of our "
        "models.",
        "No refinement: over MANY steps other methods beat it — but that is "
        "not the regime we are in.",
        {"flux2": BEST, "krea2": BEST}),
    "euler_a": (
        "Euler Ancestral",
        "Euler plus a fresh injection of noise at every step.",
        "On non-distilled models with many steps, it adds variety and "
        "micro-detail.",
        "The injected noise then has to be reconverged, which takes a "
        "comfortable step budget. Too short: a soft or noisy render. And no "
        "two renders are ever alike.",
        {"flux2": BAD, "krea2": BAD}),
    "heun": (
        "Heun",
        "Euler with a correction: two evaluations per step.",
        "A more accurate trajectory per step.",
        "**Twice as slow** for the same step count. When steps are scarce, "
        "that budget is better spent on extra Euler steps.",
        {"flux2": MEH, "krea2": OK}),
    "dpm2": (
        "DPM2",
        "A second-order method, two evaluations per step.",
        "Good per-step accuracy on classic models.",
        "The same doubled cost as Heun; it takes enough steps for the gain to "
        "show.",
        {"flux2": MEH, "krea2": MEH}),
    "dpm++2s_a": (
        "DPM++ 2S Ancestral",
        "Second order, single-step memory, with ancestral noise.",
        "Well regarded on SD1.5/SDXL at 20-30 steps.",
        "It stacks the two flaws that matter here: doubled cost AND "
        "unreconverged ancestral noise.",
        {"flux2": BAD, "krea2": BAD}),
    "dpm++2m": (
        "DPM++ 2M",
        "Multistep: it reuses the previous evaluation instead of computing a new one.",
        "The best quality/time ratio of the lot… from about fifteen steps up.",
        "Its history only exists after the 2nd step: on a very short run, a "
        "good part of it happens without one.",
        {"flux2": MEH, "krea2": OK}),
    "dpm++2mv2": (
        "DPM++ 2M v2",
        "A DPM++ 2M variant with a revised step computation.",
        "Fixes some v1 artefacts on the first steps.",
        "Same limit: multistep needs steps.",
        {"flux2": MEH, "krea2": OK}),
    "dpm++2m_sde": (
        "DPM++ 2M SDE",
        "DPM++ 2M in stochastic form (noise at every step).",
        "Richer texture on long sampling runs.",
        "Stochastic: the same step requirement as the ancestral ones, and a "
        "non-reproducible render.",
        {"flux2": BAD, "krea2": MEH}),
    "dpm++2m_sde_bt": (
        "DPM++ 2M SDE (Brownian)",
        "A Brownian-tree variant: the noise becomes reproducible.",
        "Recovers the reproducibility the plain SDE version loses.",
        "Still stochastic in principle: it needs steps.",
        {"flux2": BAD, "krea2": MEH}),
    "ipndm": (
        "iPNDM",
        "Improved pseudo-multistep, with no added noise.",
        "Sober and deterministic; quality climbs from about ten steps up.",
        "A history to build, like every multistep method.",
        {"flux2": MEH, "krea2": OK}),
    "ipndm_v": (
        "iPNDM v",
        "iPNDM with variable coefficients.",
        "Slightly more stable than iPNDM on irregular schedules.",
        "Same reservation about the step count.",
        {"flux2": MEH, "krea2": OK}),
    "lcm": (
        "LCM",
        "The sampler for models distilled **by Latent Consistency**.",
        "Excellent — on an LCM model.",
        "Neither Flux.2 Klein nor Krea 2 Turbo is LCM-distilled. Applying its "
        "trajectory to them gives a washed-out render.",
        {"flux2": BAD, "krea2": BAD}),
    "ddim_trailing": (
        "DDIM Trailing",
        "DDIM with “trailing” timestep alignment.",
        "Useful on models whose end of trajectory is poorly sampled.",
        "Designed for classic diffusion; moot on flow matching.",
        {"flux2": MEH, "krea2": MEH}),
    "tcd": (
        "TCD",
        "Like LCM: reserved for models distilled **in TCD**.",
        "Very few steps — on a TCD model.",
        "Ours are not distilled that way; it washes the render out.",
        {"flux2": BAD, "krea2": BAD}),
    "res_multistep": (
        "Res Multistep",
        "An exponential multistep integrator.",
        "Very good accuracy on flow models, at medium step counts.",
        "Multistep: hobbled when steps are missing. The most credible "
        "candidate for trying something other than Euler as soon as there are "
        "some.",
        {"flux2": MEH, "krea2": OK}),
    "res_2s": (
        "Res 2S",
        "A single-step, second-order exponential integrator.",
        "Accurate from the very first steps, with no history to build — which "
        "makes this one compatible with a tight budget.",
        "Two evaluations per step: for the same wall time, Euler does twice as many.",
        {"flux2": OK, "krea2": OK}),
    "er_sde": (
        "ER SDE",
        "An exactly reversible SDE solver.",
        "The most rigorous of the stochastic methods.",
        "Stochastic: it needs steps to show what it can do.",
        {"flux2": BAD, "krea2": MEH}),
    "euler_cfg_pp": (
        "Euler CFG++",
        "Euler with the “CFG++” guidance correction.",
        "Removes the over-saturation caused by a high CFG.",
        "**Both of our models run at CFG 1.0**: there is no guidance to "
        "correct. This variant has no business here.",
        {"flux2": BAD, "krea2": BAD}),
    "euler_a_cfg_pp": (
        "Euler Ancestral CFG++",
        "The ancestral version of the above: CFG++ correction plus a noise "
        "injection at every step.",
        "None here: the CFG++ correction is a no-op at CFG 1.0, leaving only "
        "the ancestral noise, which Euler Ancestral already provides.",
        "It stacks the uselessness of CFG++ at CFG 1.0 with ancestral noise, "
        "which needs a comfortable step budget to settle.",
        {"flux2": BAD, "krea2": BAD}),
    "euler_ge": (
        "Euler GE",
        "Euler with gradient extrapolation (the `gamma` parameter).",
        "Can tighten the result when steps are very scarce — the only one of "
        "the lot explicitly aimed at that regime.",
        "Not exposed here: `gamma` is set through `--extra-sample-args`, and "
        "without it the effect is marginal.",
        {"flux2": OK, "krea2": OK}),
    "lms": (
        "LMS (linear multi-step)",
        "Classic linear multistep (`lms_divisions`, default 1000).",
        "A recent sd.cpp addition; a method proven on long runs.",
        "Multistep: without a real step budget, the history never exists.",
        {"flux2": MEH, "krea2": MEH}),
}

# clé -> (libellé, résumé, avantage, inconvénient, {famille: niveau})
SCHEDULES: dict[str, tuple] = {
    "auto": (
        "Auto (model)",
        "Lets the engine choose according to the loaded model.",
        "Always consistent with the model: `flux2` for Flux.2 Klein, "
        "`discrete` for Krea 2. This is the setting sd.cpp documents.",
        "None — unless you want to experiment knowingly.",
        {"flux2": BEST, "krea2": BEST}),
    "discrete": (
        "Discrete",
        "A uniform spread over the model's sigmas.",
        "Neutral and unsurprising. This is what “Auto” picks on Krea 2.",
        "Nothing in particular; simply not tuned for any one model.",
        {"flux2": OK, "krea2": BEST}),
    "karras": (
        "Karras",
        "A spread that concentrates the steps towards the low sigmas.",
        "The reference on SD1.5 / SDXL, where it gains a lot.",
        "Designed for **EDM epsilon-prediction diffusion**. Our models are "
        "flow matching: the curve does not match the trajectory.",
        {"flux2": BAD, "krea2": MEH}),
    "exponential": (
        "Exponential",
        "Exponential decay of the sigmas.",
        "Simple, occasionally useful on v-prediction models.",
        "The same mismatch as Karras with respect to flow matching.",
        {"flux2": BAD, "krea2": MEH}),
    "ays": (
        "AYS (Align Your Steps)",
        "A spread optimised by NVIDIA for **small step budgets**.",
        "Designed for exactly the 8-12 step regime — the idea is sound here.",
        "Its tables are calibrated on SD1.5/SDXL, not on our models: the "
        "transfer is plausible but not guaranteed. Worth trying on Krea 2.",
        {"flux2": MEH, "krea2": OK}),
    "gits": (
        "GITS",
        "A spread derived from a graph search.",
        "Good published results at low step counts.",
        "Same reservation as AYS: calibrated elsewhere.",
        {"flux2": MEH, "krea2": OK}),
    "smoothstep": (
        "Smoothstep",
        "A curve smoothed at both ends.",
        "Soft transitions, few jolts at the start of the run.",
        "A subtle effect; nothing that makes up for a model-appropriate scheduler.",
        {"flux2": OK, "krea2": OK}),
    "sgm_uniform": (
        "SGM Uniform",
        "Uniform, in the style of the SGM implementations.",
        "Close to Discrete, predictable behaviour.",
        "No identified advantage on our models.",
        {"flux2": OK, "krea2": OK}),
    "simple": (
        "Simple",
        "An elementary linear spread.",
        "Robust, parameter-free. The default for DDIM Trailing.",
        "Coarse when the steps are few.",
        {"flux2": OK, "krea2": OK}),
    "kl_optimal": (
        "KL Optimal",
        "A spread minimising a KL divergence along the trajectory.",
        "Theoretically well founded, correct at medium step counts.",
        "No demonstrated gain when steps are scarce.",
        {"flux2": MEH, "krea2": OK}),
    "lcm": (
        "LCM",
        "The spread for Latent Consistency models.",
        "Indispensable — with the LCM sampler.",
        "Outside that pairing it crushes the trajectory and washes the render out.",
        {"flux2": BAD, "krea2": BAD}),
    "bong_tangent": (
        "Bong Tangent",
        "A tangent curve, very pronounced.",
        "An occasionally interesting stylistic effect.",
        "Empirical, with no grounding for our models.",
        {"flux2": MEH, "krea2": MEH}),
    "flux2": (
        "Flux.2",
        "A spread **cut for Flux.2**.",
        "What “Auto” selects on Flux.2 Klein: the right choice, made explicit.",
        "On Krea 2, nothing says it transfers.",
        {"flux2": BEST, "krea2": MEH}),
    "flux": (
        "Flux",
        "A sigma spread cut for the **Flux.1** models, with the shift "
        "specific to that generation.",
        "It remains a coherent flow-matching curve: it breaks nothing, and "
        "gives a slightly more contrasted render on close-ups.",
        "Flux.2 has its own; using the Flux.1 one amounts to picking the "
        "previous version of a bespoke setting.",
        {"flux2": MEH, "krea2": MEH}),
    "beta": (
        "Beta",
        "A spread following a Beta law (`alpha`, `beta` parameters).",
        "Highly tunable — through `--extra-sample-args`.",
        "Without tuning its parameters, no benefit over Discrete.",
        {"flux2": MEH, "krea2": MEH}),
    "logit_normal": (
        "Logit Normal",
        "A logit-normal spread, the one used to train many flow models.",
        "Consistent with how these models were trained — the most defensible "
        "avenue after “Auto”.",
        "Its parameters (`mu`, `std`) are not exposed here.",
        {"flux2": OK, "krea2": OK}),
}


# Familles documentées. Le repli sur « flux2 » vaut pour un modèle inconnu :
# mieux vaut les verdicts d'un distillé à peu de pas — les plus restrictifs —
# que pas de verdict du tout.
FAMILIES = ("flux2", "krea2")


def _family(model_family: str) -> str:
    return model_family if model_family in FAMILIES else "flux2"


def level(kind: str, key: str, model_family: str) -> str:
    table = SAMPLERS if kind == "sampler" else SCHEDULES
    entry = table.get(key)
    if not entry:
        return OK
    return entry[4].get(_family(model_family), OK)


def choices(kind: str, model_family: str) -> list[tuple[str, str]]:
    """Menu ANNOTÉ : le libellé porte déjà le verdict.

    Marquer les options dans la liste évite d'avoir à ouvrir une aide pour
    savoir laquelle prendre — l'information est là où se fait le choix.
    """
    table = SAMPLERS if kind == "sampler" else SCHEDULES
    out = []
    for key, entry in table.items():
        mark = _MARK[level(kind, key, model_family)]
        out.append((f"{i18n.t(entry[0])} {mark}".strip(), key))
    return out


_RATIONALE = """On **{model}**, three properties of the model decide almost everything — and
they rule out entire families of options, not one or two case by case.

**1. It is a *flow matching* model.** sd.cpp runs it in “Flux FLOW” mode. The
**Karras** and **Exponential** schedulers, which gain a lot on SD 1.5 and SDXL,
were designed for another mechanism (EDM epsilon-prediction diffusion): their
sigma spread does not match the trajectory followed here.

**2. It is distilled at CFG 1.0.** There is therefore **no guidance to
correct**: the whole **CFG++** family (`Euler CFG++`, `Euler Ancestral CFG++`)
has literally nothing to do. This is also why the **negative prompt is
ignored**, whichever sampler you pick.

**3. It runs in {steps} steps.** That is very few, and it disqualifies two
families:

- the **ancestral** and **stochastic** methods (`Euler Ancestral`,
  `DPM++ 2S Ancestral`, the `SDE` ones, `ER SDE`) inject noise at every step.
  That noise then has to be reconverged — there is no budget for it, and the
  render comes out soft or noisy;
- the **multistep** methods (`DPM++ 2M`, `iPNDM`, `Res Multistep`, `LMS`) must
  first accumulate a history of evaluations. Over {steps} steps, a good part of
  the run happens before that history exists.

Finally, **LCM** and **TCD** are not general-purpose options: they are the
samplers of models distilled *by those very methods*. This model is not;
applying them washes the render out.

---

**What is left, in practice:** `Euler` + `Auto`. {advice}

*These verdicts are reasoned from the model's properties, not drawn from a
benchmark — they say where to aim your experiments, not what your eye will
prefer.*"""

_ADVICE = {
    "flux2": "Over 4 steps there is virtually nothing to gain elsewhere; if "
             "you want to experiment, `Res 2S` is the only other one that is "
             "accurate without a history to build.",
    "krea2": "Over 8 steps the margin is a little wider: `Res Multistep`, "
             "`DPM++ 2M` and the `AYS` scheduler (designed for small step "
             "budgets) are worth a side-by-side try at a fixed seed.",
}


# (nom affiché, pas, CFG) par famille documentée.
_MODEL = {
    "flux2": ("Flux.2 Klein", "4", "1.0"),
    "krea2": ("Krea 2 Turbo", "8", "1.0"),
}


def rationale(model_family: str) -> str:
    fam = _family(model_family)
    name, steps, cfg = _MODEL[fam]
    # Traduire AVANT de formater : les placeholders survivent (garanti par
    # tests/test_i18n.py), et le texte inséré est traduit séparément.
    return i18n.t(_RATIONALE).format(model=name, steps=steps, cfg=cfg,
                                     advice=i18n.t(_ADVICE[fam]))


def describe(kind: str, key: str, model_family: str) -> str:
    """Fiche Markdown de l'option choisie : résumé, pour, contre, verdict."""
    table = SAMPLERS if kind == "sampler" else SCHEDULES
    entry = table.get(key)
    if not entry:
        return ""
    name, summary, pro, con, _lv = entry
    lv = level(kind, key, model_family)
    return (f"**{i18n.t(name)}** — {i18n.t(summary)}\n\n"
            f"✅ {i18n.t(pro)}\n\n"
            f"❌ {i18n.t(con)}\n\n"
            f"{_MARK[lv]} {i18n.t(_VERDICT[lv])}".replace("  ", " "))
