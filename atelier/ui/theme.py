"""Thème et habillage CSS.

Trois principes, dans cet ordre :

1. **L'accent se mérite.** Le cyan ne sert qu'à l'action principale et à l'état
   actif. Quand chaque libellé de champ porte une pastille colorée — ce que
   faisait la version précédente — plus rien ne ressort, et l'ensemble a l'air
   d'un jouet. La hiérarchie se fait à la graisse et à l'espacement.

2. **Aucune ressource externe.** Une application locale ne doit pas attendre
   Google Fonts pour s'afficher : hors ligne, la page reste sur « Loading… » le
   temps du timeout. On utilise donc la pile de polices SYSTÈME, qui est déjà
   sur la machine, s'affiche instantanément et ne dépend de rien.

3. **Les sélecteurs doivent exister.** Gradio 5.x a renommé le conteneur
   d'onglets : le CSS visait `.tab-nav`, qui n'existe plus, donc rien ne
   s'appliquait. On cible désormais `button[role=tab]`, stable parce que c'est
   la sémantique ARIA et non une classe interne.
"""
from __future__ import annotations

import gradio as gr

ACCENT = "#00b8e6"
ACCENT_HOVER = "#00a1cc"
ACCENT_DARK = "#00647f"

# Rampe de teintes centrée sur l'accent, pour les composants Gradio.
_ACCENT_RAMP = gr.themes.Color(
    name="cyan-accent",
    c50="#ecfbff", c100="#cff4ff", c200="#9de8ff", c300="#5fd8fb",
    c400="#22c6ef", c500="#00b8e6", c600="#0092b8", c700="#00708f",
    c800="#00566e", c900="#083a4a", c950="#04222e",
)

# Pile SYSTÈME : présente partout, aucun téléchargement, aucun décalage de mise
# en page au chargement. `-apple-system` couvre macOS, `Segoe UI Variable`
# Windows 11, `Segoe UI` Windows 10, `Inter`/`Roboto` les Linux qui les ont.
_FONTS = ["-apple-system", "BlinkMacSystemFont", "Segoe UI Variable Text",
          "Segoe UI", "Inter", "Roboto", "Helvetica Neue", "system-ui",
          "sans-serif"]
_MONO = ["ui-monospace", "SFMono-Regular", "SF Mono", "Cascadia Mono",
         "JetBrains Mono", "Consolas", "monospace"]


def theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=_ACCENT_RAMP,
        secondary_hue=_ACCENT_RAMP,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=_FONTS,
        font_mono=_MONO,
    ).set(
        body_background_fill="#f4f6fa",
        body_background_fill_dark="#0a0f14",
        block_background_fill="#ffffff",
        block_background_fill_dark="#131a21",
        block_border_width="1px",
        block_border_color="#e6e9ef",
        block_border_color_dark="#1f2a35",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color="#5b6675",
        block_label_text_color_dark="#8d9aa9",
        block_label_text_weight="600",
        block_label_text_size="0.8rem",
        block_title_text_color="#5b6675",
        block_title_text_color_dark="#8d9aa9",
        block_title_text_weight="600",
        block_shadow="0 1px 2px rgba(15,23,42,.05)",
        input_background_fill="#ffffff",
        input_background_fill_dark="#0f161d",
        input_border_color="#dde2ea",
        input_border_color_dark="#243240",
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_hover=ACCENT_HOVER,
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_large_radius="10px",
        button_small_radius="8px",
        slider_color=ACCENT,
    )


CSS = f"""
/* ------------------------------------------------------------------ *
 *  Tokens: a single source for the colours we repeat.
 * ------------------------------------------------------------------ */
:root {{
  --tsg-accent: {ACCENT};
  --tsg-accent-dark: {ACCENT_DARK};
  --tsg-ink: #0f172a;
  --tsg-muted: #64748b;
  --tsg-line: #e6e9ef;
  --tsg-surface: #ffffff;
  --tsg-raise: 0 1px 2px rgba(15,23,42,.05);
}}
.dark {{
  --tsg-ink: #e6edf3;
  --tsg-muted: #8d9aa9;
  --tsg-line: #1f2a35;
  --tsg-surface: #131a21;
  --tsg-raise: none;
}}

.gradio-container {{ max-width: 1750px !important; margin: auto; }}

/* ------------------------------------------------------------------ *
 *  Header: compact. It must not cost a third of the first screen.
 * ------------------------------------------------------------------ */
#atelier-header {{ display:flex; align-items:baseline; gap:12px;
                   flex-wrap:wrap; padding:2px 0 8px 0; }}
#atelier-header h1 {{ font-size:1.35rem; margin:0; letter-spacing:-.01em;
                      font-weight:700; color:var(--tsg-ink); }}
#atelier-header .sub {{ color:var(--tsg-muted); font-size:.8rem; }}
#atelier-header .dot {{ color:var(--tsg-line); }}
/* Hardware status chip: readable at a glance, never garish. */
#atelier-header .chip {{ font-size:.74rem; font-weight:600; padding:2px 9px;
    border-radius:999px; border:1px solid var(--tsg-line);
    color:var(--tsg-muted); white-space:nowrap; }}
#atelier-header .chip.ok {{ color:#15803d; border-color:#bbf7d0;
    background:#f0fdf4; }}
#atelier-header .chip.warn {{ color:#b45309; border-color:#fde68a;
    background:#fffbeb; }}
.dark #atelier-header .chip.ok {{ background:#052e16; border-color:#14532d;
    color:#4ade80; }}
.dark #atelier-header .chip.warn {{ background:#2a1c05; border-color:#57400d;
    color:#fbbf24; }}

/* ------------------------------------------------------------------ *
 *  Tabs. `button[role=tab]`: ARIA semantics, so it stays stable from one
 *  Gradio version to the next — unlike the internal class names.
 * ------------------------------------------------------------------ */
.tab-container {{ border-bottom:1px solid var(--tsg-line) !important;
                  gap:2px !important; margin-bottom:14px !important; }}
button[role=tab] {{
    font-size:.9rem !important; font-weight:600 !important;
    padding:8px 14px !important; color:var(--tsg-muted) !important;
    border:none !important; background:transparent !important;
    border-radius:8px 8px 0 0 !important;
    border-bottom:2px solid transparent !important;
    transition:color .12s ease, background .12s ease; }}
button[role=tab]:hover {{ color:var(--tsg-ink) !important;
    background:rgba(100,116,139,.08) !important; }}
button[role=tab][aria-selected=true] {{
    color:var(--tsg-accent-dark) !important;
    background:transparent !important;
    border-bottom:2px solid var(--tsg-accent) !important; }}
.dark button[role=tab][aria-selected=true] {{ color:var(--tsg-accent) !important; }}
/* Sub-tabs: quieter than the root tabs, so the hierarchy reads without
   having to think about it. */
.tab-container .tab-container button[role=tab] {{
    font-size:.85rem !important; padding:6px 11px !important; }}

/* ------------------------------------------------------------------ *
 *  Labels: typographic, NOT coloured pills.
 *
 *  Gradio's "Soft" theme puts a tinted background behind every block
 *  label. Multiplied by the twenty fields of a tab, the accent ends up
 *  everywhere — and therefore nowhere: nothing stands out, and the whole
 *  thing looks like a mock-up. The theme tokens do not cover every
 *  component (gallery, accordion, image), hence these explicit rules.
 * ------------------------------------------------------------------ */
.block-label, .block-title, label > span:first-child,
.gradio-container .block > .label-wrap > span {{
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
    color:var(--tsg-muted) !important;
    font-weight:600 !important;
    font-size:.8rem !important;
    letter-spacing:.005em;
    padding-left:0 !important; }}
.block-label {{ backdrop-filter:none !important; }}
.gradio-container .block > .label-wrap {{ margin-bottom:4px; }}
/* The accordion keeps its weight: it is a section title, not a label. */
.gradio-container .label-wrap > span {{ font-size:.88rem !important;
    color:var(--tsg-ink) !important; }}

/* ------------------------------------------------------------------ *
 *  Primary action: the only place where the accent is solid.
 * ------------------------------------------------------------------ */
.go-row {{ gap:8px !important; }}
.go-row button {{ font-weight:650 !important; }}
.go-row button.primary {{ font-size:1rem !important; letter-spacing:.01em;
    box-shadow:0 1px 2px rgba(0,184,230,.35), 0 4px 14px rgba(0,184,230,.22); }}
.go-row button.primary:hover {{
    box-shadow:0 1px 2px rgba(0,184,230,.4), 0 6px 18px rgba(0,184,230,.3); }}

/* ------------------------------------------------------------------ *
 *  Cards, tags, states.
 * ------------------------------------------------------------------ */
.model-card {{ border:1px solid var(--tsg-line); border-radius:12px;
               padding:13px 15px; background:var(--tsg-surface);
               margin-bottom:9px; box-shadow:var(--tsg-raise); }}
.model-card h3 {{ margin:0 0 4px 0; font-size:1rem; }}
.tag {{ display:inline-block; background:rgba(0,184,230,.12);
        color:var(--tsg-accent-dark); border-radius:999px; padding:2px 9px;
        font-size:.7rem; font-weight:600; margin-right:5px; }}
.dark .tag {{ color:#5fd8fb; }}
.status-ok {{ color:#15803d; font-weight:600; }}
.status-missing {{ color:#b45309; font-weight:600; }}
.log-box textarea {{ font-family:{", ".join(_MONO)}; font-size:.78rem;
                     line-height:1.5; resize:vertical; }}

/* .hint = what IS GOING to happen · .feedback = what HAS happened. */
.hint p {{ margin:.2rem 0 !important; font-size:.8rem;
           color:var(--tsg-muted); line-height:1.5; }}
.feedback:not(:empty) {{ border-left:3px solid var(--tsg-accent);
    background:rgba(0,184,230,.06); border-radius:0 8px 8px 0;
    padding:7px 11px; margin:6px 0; }}
/* The `:not(:empty)` above NEVER bites: Gradio puts the class on the
   container, which keeps a child even when the Markdown is empty. The result
   is a permanent blue band that looks like an unreadable message. So we test
   for a RENDERED paragraph instead, which is the real criterion. */
.feedback:not(:has(p)) {{ display:none; }}
.feedback p {{ margin:.15rem 0 !important; font-size:.83rem; line-height:1.5; }}

/* Start-up alert banner: compact, folded onto one line. */
#atelier-alerts:not(:empty) {{ border:1px solid #fde68a; background:#fffbeb;
    color:#92400e; border-radius:10px; padding:8px 12px; margin-bottom:10px;
    font-size:.84rem; }}
#atelier-alerts p {{ margin:.12rem 0 !important; }}
.dark #atelier-alerts:not(:empty) {{ background:#2a1c05; border-color:#57400d;
    color:#fcd34d; }}

/* ------------------------------------------------------------------ *
 *  Dimension stability (avoids jumps when resizing).
 * ------------------------------------------------------------------ */
[data-testid="image"] img, .image-frame img, .image-container img {{
    object-fit:contain !important; width:100% !important; max-height:70vh; }}
[data-testid="image"], .image-container {{ overflow:hidden; }}
textarea {{ resize:vertical !important; max-width:100% !important; }}
.gr-image, .gr-gallery {{ min-height:0; }}
footer {{ display:none !important; }}

/* Accessibility: a visible focus ring, in the accent colour. */
:where(button, input, textarea, select, [tabindex]):focus-visible {{
    outline:2px solid var(--tsg-accent) !important; outline-offset:2px; }}
"""
