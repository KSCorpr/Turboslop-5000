# 🟢 Turboslop 5000

> **5000 edition:** based on [KSCorpr/Turbo-Slop-Generator-3000](https://github.com/KSCorpr/Turbo-Slop-Generator-3000)
> at `2e83c762e439853d93b9932969c249163ee64c4d`. Keeps the original local,
> Gradio + stable-diffusion.cpp architecture and its existing tools.
> See [changes, research and verification](docs/TURBOSLOP_5000.md).
> Updates target [KSCorpr/Turboslop-5000](https://github.com/KSCorpr/Turboslop-5000).
> The original repository is unchanged.


> **It adapts to your hardware.** By default the app uses the **best NVIDIA card
> it detects** (or the **Apple Silicon GPU** on a Mac) and tunes itself to it:
> diffusion quantization from VRAM, text encoder offloaded to system RAM,
> flash-attention, CPU offload, VAE tiling. Two advanced options live in
> **Settings**: the **per-generation card profiles** (GTX 10xx → RTX 50xx, one
> click) and **multi-GPU** (which card generates, whether the text encoder moves
> to a second card, or automatic memory placement).

> **The interface is in English**, and only in English. It used to be written in
> French and translated through a dictionary; that layer is gone and the source
> strings are English. See [Interface language & theme](#interface-language--theme).

A **local**, modern, lightweight image-generation studio for artists, built on
**[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** (native
CUDA, GGUF). Generate with **Flux.2 Klein 9B** and **Krea 2 Turbo**, with an
on-demand model catalog, automatic optimization for your RTX card, LoRA, native
resolution presets, saved styles, an AI prompt enhancer, multi-reference image
editing, three upscalers, and a utility toolkit.

No ComfyUI, no node spaghetti — just a clean web UI.

> **Credits & honesty.** All the heavy lifting — the inference engine, GGUF
> support, CUDA kernels — comes from **Leejet’s
> [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)**. This
> project is just a friendly local UI on top of it; full credit and thanks to
> Leejet and the sd.cpp contributors.
>
> This GUI was **vibe-coded with [Claude](https://claude.ai/code)** (Anthropic) —
> built iteratively in plain language rather than hand-written line by line. Treat
> it accordingly: it’s a hobby tool, not battle-tested production software. Read
> the code, test before relying on it, and report anything that breaks.

**Six root tabs**, arranged by what they do rather than by what they are: what
*produces* an image stays at the root, what *retouches* one lives under **Tools**,
what administers the machine lives under **System**. (Eleven root tabs used to
overflow into a `…` menu, which made Manage and Settings invisible at a glance —
so the grouping is not decoration.)

| Tab | What it does |
|---|---|
| 🟣 **Flux.2 Klein** | fast (4 steps) · text-to-image & **multi-reference image editing** · presets, styles, LoRA |
| ⚡ **Krea 2 Turbo** | fast photorealism (8 steps, GGUF, Qwen3-VL encoder, WAN 2.1 VAE) |
| 💊 **Xanax** | one sentence → **one photo** · style **hard-wired**, nothing to configure · model picker for either engine |
| 📚 **Model Catalog** | hardware-aware recommendations, on-demand download / delete |
| 🧰 **Tools** | **Toolkit** (**image → prompt** · depth · background removal · click-to-cutout (SAM) · layers → PSD · ESRGAN · **HD**, the native sd.cpp highres fix with no tiles · **high resolution** (Flux.2 as its own upscaler) · SeedVR2 · **face restoration** · creative SDXL upscale) · **Outpaint** · **Image → 3D** (textured GLB via **trellis.cpp**, native CUDA, no PyTorch) |
| ⚙️ **System** | **Settings** (detected hardware, quantization, optimizations) · **Manage & help** (disk inventory with sizes, selective uninstall, models location, image-display diagnostic, in-app documentation of every option) · **Convert to GGUF** |

The exact tab tree, since two of the six are containers:

```
🟣 Flux.2 Klein 9B   ⚡ Krea 2 Turbo   💊 Xanax   📚 Model Catalog
🧰 Tools    → 🧰 Toolkit  ·  🖼️ Outpaint  ·  🧊 Image → 3D
⚙️ System   → ⚙️ Settings ·  🧹 Manage & help  ·  🔧 Convert to GGUF

🧰 Toolkit  → 📝 Image → prompt · 🌐 Depth · ✂️ Background removal
              🪄 Cut out (SAM)  · 🧩 Layers · 🔼 Upscale · 🚀 HD
              🔍 High resolution · 🌱 Restore · 🙂 Faces · ✨ SDXL upscale
```

---

## Table of contents

- [Install](#install)
  - [Updating the app itself](#updating-the-app-itself)
  - [Maintenance](#maintenance)
- [Quick start](#quick-start)
- [Model catalog](#model-catalog)
- [Generation options](#generation-options)
- [Xanax tab](#xanax-tab)
- [Hardware & optimization](#hardware--optimization)
  - [Samplers & schedulers](#samplers--schedulers)
  - [Multi-GPU](#multi-gpu)
  - [One-shot CLI by default, resident engine on demand](#one-shot-cli-by-default-resident-engine-on-demand)
  - [Updating the engines](#updating-the-engines)
  - [Interface language & theme](#interface-language--theme)
- [Upscaling](#upscaling)
  - [🔼 Simple (ESRGAN)](#-simple-esrgan-native-sdcpp)
  - [🚀 HD (native highres fix)](#-hd-native-sdcpp-highres-fix)
  - [🌱 Restore (SeedVR2)](#-restore-seedvr2-3b--7b)
  - [🔍 High resolution (Flux.2)](#-high-resolution-flux2-as-its-own-upscaler)
  - [🙂 Faces](#-faces-gfpgan--restoreformer--codeformer)
  - [✨ Creative (SDXL)](#-creative-sdxl-ultimate-sd-upscale)
- [Outpaint](#outpaint)
- [Image → 3D](#image--3d)
- [Toolkit](#toolkit)
  - [Image → prompt](#-image--prompt)
  - [Layers → PSD](#layers--psd)
  - [Prompt enhancer](#prompt-enhancer-ai)
- [Convert to GGUF](#convert-to-gguf)
- [Managing disk space & uninstalling](#managing-disk-space--uninstalling)
- [Sharing on your LAN](#sharing-on-your-lan)
- [Distributing a portable package](#distributing-a-portable-package)
- [Models & sources](#models--sources)
- [Project layout](#project-layout)
- [Gradio version](#gradio-version)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Install

### Windows (RTX cards)
```bat
install.bat      ::  portable Python + dependencies + GGUF engine (CUDA)
run.bat          ::  launch the UI at http://127.0.0.1:7860
update.bat       ::  update the app itself (code only, never your data)
```
Windows and Linux use the **CUDA** build; macOS uses the **Metal** one. The
engine variant is derived from the platform, so no flag to remember.

### Linux (NVIDIA)
```bash
./install.sh
./run.sh
```

### macOS (Apple Silicon)
```bash
./install.sh
./run.sh
```
The same script; it detects the platform and fetches the **Metal** build of
stable-diffusion.cpp instead of the CUDA one. What differs on a Mac:

- **Unified memory, not VRAM.** The GPU addresses the same memory as the CPU, so
  the app reports "~75% of RAM addressable by the GPU" rather than inventing a
  VRAM figure. **RAM offload is switched off** — offloading to a memory the GPU
  is already using saves nothing and only adds copies.
- **The encoder budget is tighter than on PC.** On a PC the text encoder is
  offloaded to system RAM and competes with nothing; here it shares one pool with
  the diffusion model, so the encoder quantization is picked from what's left
  rather than from total RAM.
- **Toolkit add-ons run on MPS** (Metal) via PyTorch, with
  `PYTORCH_ENABLE_MPS_FALLBACK` set so an operator MPS lacks drops to CPU instead
  of killing the run.
- **🧊 Image → 3D is unavailable.** trellis.cpp ships a Windows-CUDA binary only —
  no Apple Silicon build, no Metal path. The tab says so instead of offering an
  installer that would find nothing. Everything else works.
- **16 GB is the realistic floor**, 24 GB+ comfortable. Intel Macs are not
  supported: upstream publishes no Intel build.

> The install does **not** download any models. You fetch them on demand from the
> **Model Catalog** tab (like a media library). Everything stays inside the
> project folder.

Generation runs through stable-diffusion.cpp (no heavy PyTorch for image
generation). **PyTorch is only installed on demand** for the optional Toolkit
tools (depth, background removal, SAM, prompt enhancer, creative SDXL upscale),
each via its own one-click installer.

### Updating the app itself

**`update.bat` is the way** (`./update.sh` on Linux/Mac). It downloads the
current code from GitHub and applies it in place — no manual re-download, and
nothing of yours is touched: `models/`, `loras/`, `outputs/`, `userdata/`,
`tools_repo/`, `bin/` and `python/` are off limits by construction.

```bat
update.bat              ::  fetch and apply the current code
update.bat --check      ::  show what would change, write nothing
update.bat --rollback   ::  undo the last update
```

What it does that dropping a ZIP over the folder cannot:

- it writes **only files that actually differ**, and lists them;
- it **deletes what disappeared** from the project — but only files it installed
  itself, tracked in `userdata/app-update.json`. A file it never wrote is not
  its business;
- it **backs up everything it replaces**, so `--rollback` undoes the update;
- it **refuses a suspicious download** (a proxy's HTML error page is a perfectly
  readable "zip") and any archive that does not contain the app;
- if the updated code **does not compile**, it restores the previous version by
  itself;
- it purges `__pycache__`, because a `.pyc` of a deleted module stays importable.

Close the app first: Windows cannot replace a file that is open.

**`update.bat` updates the code only.** The engines are separate — see
[Updating the engines](#updating-the-engines).

### Maintenance
If you instead update by extracting the repo ZIP over your existing folder
(keeping `python/`, `bin/`, `models/`…), copy-paste **adds and overwrites files
but never deletes** the ones removed upstream — they linger as orphans, and
stale `__pycache__` can confuse Python. After each copy-paste update, run:
```bat
maintenance.bat      ::  Windows   (./maintenance.sh on Linux/Mac)
```
It deletes the **code** of removed features, purges `__pycache__` and `tmp/`,
then verifies that everything compiles, the model catalog is valid, and the
dependencies + `sd-cli` engine are present. It never touches `models/custom/`,
`loras/`, `outputs/`, `userdata/`, `python/` or `bin/`.

**The engine updates separately from the code, and that is the trap.** Copying
the repo over the old one does not touch `bin/`, so the app can start asking for
a `sd-cli` option the installed binary has never heard of — and you find out when
a tab fails. Maintenance therefore checks the engine's *capabilities*, not just
its presence: it parses `sd-cli -h` against the options the current code actually
needs, and names the feature rather than the flag ("`--hires` missing" tells
nobody anything; "the HD tab will not work" does). To fix everything in one go:

```bat
maintenance.bat --all    ::  purge + engine update  (./maintenance.sh --all)
```

`--update-engine` alone does just the engine. Updates are transactional: the
archive is unpacked into a staging directory, checked with `sd-cli -h`, and only
then swapped into `bin/`. The last known-good engine stays in
`.engine-previous/`; use `rollback-engine.bat` (or `./rollback-engine.sh`) to
restore it immediately. A failed download, unsafe archive or failed smoke test
leaves the installed engine untouched.

**Orphan modules are found generically.** Beyond the hand-declared
`REMOVED_FEATURES`, maintenance walks the import graph from `app.py` and
`scripts/`, and reports any module under `atelier/` that nothing reaches. That
catches leftovers from versions nobody remembered to declare. It reports rather
than deletes: a dynamically loaded module would show up here wrongly.

**Data left behind by removed features is measured, not deleted.** When a feature
goes away it leaves gigabytes on disk — downloaded weights, cloned repos, model
folders no longer in the catalog. Erasing those silently is not the script's call,
so it reports each one with its size and a single recoverable total:

```bat
maintenance.bat --purge      ::  actually deletes them (./maintenance.sh --purge)
```

Three things are checked, and none of them relies on a hand-kept list of files:
orphan **add-ons** in `tools_repo/` are whatever no longer matches an add-on in
the code, orphan **models** are whatever the catalog no longer references, and
removed features declare their own leftovers in `REMOVED_FEATURES` at the top of
`scripts/maintenance.py` — adding an entry there is the only step needed when
something is dropped.

**What a copy-paste update does and does not refresh**

| | In the ZIP? | Refreshed by copy-paste |
| --- | --- | --- |
| App code, `config/models.yaml`, docs | yes | **yes** |
| Engine binary (`bin/`) | no | no — run `update-engine.bat` only when a new sd.cpp feature is needed |
| Models, LoRAs, outputs, prefs (`models/`, `loras/`, `outputs/`, `userdata/`) | no | no — kept, which is the point |
| Toolkit add-ons (`tools_repo/`) | no | **no — and this one bites** |

That last row matters. An add-on is not just a folder of weights: its installer
also pins Python package versions **shared with every other add-on**. When a
release changes *how an add-on installs*, updating the app leaves the installed
add-on frozen in its old state, and you only find out at the next use. So:
**after updating, if an add-on misbehaves, re-run its one-click installer** —
weights already on disk are not re-downloaded.

`maintenance` reports leftovers from removed features and tells you exactly how
much disk they hold, so you don't have to guess.

It also **reports any file from the update manifest that has gone missing** —
the diagnostic that was absent the day a module vanished and the app stopped
starting with an `ImportError` naming the module but not the cause.

---

## Quick start

1. Run `install.bat` / `./install.sh`, then `run.bat` / `./run.sh`.
2. Open the **📚 Model Catalog** tab and download **Flux.2 Klein 9B** (or **Krea
   2 Turbo**). Quantization is picked automatically for your VRAM/RAM.
3. Go to the model's **generation tab**, type a prompt, click **🎨 Generate**.
4. (Optional) Install the **prompt enhancer** and click **✨ Enhance prompt** to
   turn a rough idea into a detailed English prompt.

---

## Model catalog

**📚 Model Catalog** is the media library: nothing is downloaded at install
time, everything is fetched on demand from here.

Each entry shows what the model is, what it weighs, and **whether it is ready**
(● installed · ○ to download). The **quantization is chosen for your machine**
before the download starts — diffusion from VRAM, text encoder from RAM — and
if the exact rung does not exist in the source repository the downloader takes
the closest one **and says so**, rather than silently handing you a smaller
model (see [Models & sources](#models--sources) for why that footnote exists).

- **⬇️ Download** fetches every component the model needs (diffusion, VAE, text
  encoder, and the vision projector for edit models) into `models/`, with a
  live log.
- **🗑️ Delete** removes a model's own files and **keeps shared ones**: an
  encoder or VAE used by another installed model is never taken out from under
  it.
- The catalog itself is `config/models.yaml` — the single source of truth for
  sources, defaults and presets.

The tools that live under 🧰 Toolkit are **not** here: each one carries its own
one-click installer in its own tab, because each pulls a different Python
dependency set.

---

## Generation options

Every generation tab exposes the same controls.

### Prompt & system style
- **Prompt** — your description. For **edit models** (Flux.2 Klein) describe the
  *modification* to apply to the reference image.
- **✨ Enhance prompt (AI)** — see [Prompt enhancer](#prompt-enhancer-ai). Note
  that a **style preset translates nothing** — it is a prefix glued in front of
  your text, so writing in French leaves you with French plus an English header.
  The enhancer is what translates and shapes the prompt, and it is **given the
  active preset as a constraint**: it describes the subject without adding
  camera, lens, lighting or processing wording that would contradict the style.
  Pick the preset first, then enhance.
- **Negative prompt** — shown only for models that support it (CFG > 1). Both
  our models are distilled at CFG 1.0 and ignore it, so the field stays hidden.
- **System / style prefix** (🎨 Styles → *Custom preset*) — a prefix prepended
  to every prompt. Save
  reusable styles to a dropdown (persisted in `userdata/`). Styles are **global**:
  saved once, available in every generation tab. A few are **bundled** with the
  app in `config/style_presets.json` — they survive updates and cannot be
  deleted, but saving a style under the same name creates your own version, which
  takes precedence; deleting that restores the original. Two separate buttons,
  so nothing is lost by accident: **✖️ Stop applying** only detaches the style
  from the next generations and keeps the preset, while **🗑️ Delete this preset
  (permanent)** erases it and asks for a second click to confirm. Bundled today:
  **📷 Provincial France 1995-2005 (amateur)**, a transcription of a
  "mundane amateur snapshot, provincial France, always overcast, no
  post-processing" brief.
- **📷 Krea 2 photo styles** (🎨 Styles → *Photo*) — a bundled bank of **139 stackable
  photographic styles** (quality, lighting, lens, film stock, mood…), grouped by
  category in a **multi-select** dropdown. Your prompt subject is inserted into
  each selected style (`{prompt}` template), and multiple styles chain their
  descriptions after the subject so you can combine axes. Style negatives are
  merged into the negative field **only when the model uses them** (CFG > 1);
  on distilled CFG 1.0 models they are dropped. Bank © *ghleg* — MIT
  ([aoleg/Photographic-styles-and-wildcards-for-Krea-2](https://github.com/aoleg/Photographic-styles-and-wildcards-for-Krea-2)),
  shipped as `config/krea2_styles.csv`.
- **🎨 Krea artistic styles** (🎨 Styles → *Artistic*) — a bundled bank of **397 stackable
  artistic styles** (anime, cartoon, comics, drawing, photography, design,
  digital painting, painting), grouped by category in a **multi-select**
  dropdown. These have no `{prompt}` and no negatives: the style description is
  appended after your subject. They stack with each other **and** with the photo
  styles above. A **🎲 wildcard** button rolls one at random. Shipped as
  `config/krea2_art_styles.csv` — a Krea style collection supplied by the user;
  **provenance/license to be confirmed** (no license was attached to the source).

### Reference image / image-to-image
The accordion adapts to the model family:
- **Flux.2 Klein (edit model)** — **Multi-reference editing**: load the image to
  edit plus up to **2 extra reference images**, and describe the change or the
  combination in the prompt (e.g. *“put the character from image 1 into the scene
  of image 2”*). Each image is passed to the engine as a separate `-r` flag. No
  strength slider — editing is prompt-driven. Output aspect follows your image.
  An **🧩 Outpaint** slider (experimental) extends the canvas and lets the model
  fill the new borders — describe the extension in the prompt.
- **Krea 2 Turbo — ✏️ Edit mode (Ostris Edit)** — tick **✏️ Edit mode** to pass
  the image as a **context reference** (style transfer, subject reference, edits)
  instead of an img2img starting point. This requires a **Krea 2 edit LoRA**
  (e.g. HF repo [`ostris/krea2_turbo_style_reference`](https://huggingface.co/ostris/krea2_turbo_style_reference) —
  add it via the LoRA panel) and a **recent sd.cpp engine** (`update-engine.bat`;
  needs the `Krea2OstrisEdit` + `--llm_vision` support from July 2026). The
  Qwen3-VL **vision projector (mmproj)** downloads automatically with the model
  and is only loaded in edit mode.
- **Other models (img2img)** — load a **reference / starting image** and set the
  **transformation strength** — low (0.2–0.4) keeps the reference's structure,
  high (0.7–1.0) reinvents it.

### LoRA
Drop `.safetensors` / `.gguf` files into **`loras/`**, then pick up to **2** with
their weights. The `<lora:name:weight>` syntax is forwarded to the engine. Use
**↻ Refresh** after adding files, **✖ Clear** to reset. You can also **import a
LoRA from Civitai** in one click — paste the model URL (or version ID) into the
LoRA accordion; gated models need a Civitai token (Settings).

### Local / custom files
To use a model downloaded elsewhere, drop the file(s) into **`models/custom/`**
and select them as *Diffusion / VAE / Encoder (local)*. Empty = use the catalog
model. **✖ Clear custom fields** resets the selection.

### Resolution presets (native per model)
Each model offers **formats aligned with its training resolutions** (it renders
best on these):
- **Flux.2 Klein** — ~1 MP, 32-px grid: 1024², 1248×832, 1184×880, 1392×752,
  1568×672… (+ a 2K option).
- **Krea 2** — 1024 family (multiples of 64): 1024², 1216×832, 1152×896,
  1344×768… (+ a 2K option).

Pick a ratio from the dropdown, or choose **Custom (sliders)** for free width /
height (256–2048, step 16). Loading a reference image auto-fits width/height to
its aspect.

### Sampler / scheduler / steps
- **Preset** — vetted combos per model (e.g. Flux.2 Klein → 4 steps / CFG 1.0 /
  euler + simple). Selecting one fills sampler, scheduler, steps and CFG.
- **Sampler** — all samplers supported by sd.cpp (euler, dpm++2m, res_multistep…).
  Each entry is **annotated for the model of the current tab** — ⭐ recommended,
  no mark = usable, △ poorly suited, ⚠️ discouraged — and the card below the menu
  spells out what the selected one does, its ✅ upside and its ❌ downside.
- **Scheduler (sigmas)** — auto (model default), karras, simple, exponential…
  Annotated and documented the same way.
- **📖 Why half of this menu is useless here** — a fold-out that explains the
  verdicts from the model's own properties. See
  [Samplers & schedulers](#samplers--schedulers) for the full reasoning.
- **Steps** — diffusion steps. Distilled models need few (4–8).
- **CFG** — guidance. **1.0 = no guidance** (normal for distilled Flux). Values
  other than 1.0 are experimental on distilled models.
- **Flow shift** — leave at **0 (auto)**: the model picks the right value for the
  resolution. Too low (1–2) leaves grain/noise at high resolution; ~3–4 reinforces
  structure.

### Seed & batch
- **Seed** — `-1` = random. The used seed is shown under each result and written
  to the sidecar file.
- **Images** — batch count (1–8).

### Output
- **Merged preview & results** — the live preview shows in the gallery during
  generation, then the final images replace it (one view).
- **Seed** — the selected image's seed shows in a copy-button box; **Reuse this
  seed** drops it back into the seed field. Clearing the seed field resets it to -1.
- **Send to Toolkit** — push the selected image straight into a Toolkit tool
  (depth, background removal, SAM, ESRGAN or creative upscale).
- **Saved prompts** — every image gets an A1111-style `.txt` sidecar in
  `outputs/` with the prompt, negative, model, sampler/scheduler, seed and size.

## Xanax tab

**💊 Xanax** takes **one sentence about your day and returns one photo**. Pick
the model with a radio button (Krea 2 Turbo or Flux.2 Klein); everything else is
compiled in. No style dropdown, no system-prompt box, no preset menu — **the
style cannot be changed**. That is the point of the tab; the normal generation
tabs are there when you want to tune something.

The fixed style: amateur snapshot, provincial France, 1995–2005, cheap
point-and-shoot, ordinary people, unstaged, always overcast, no grain, no filter,
no post-processing, **4:3** on each model's native grid (1184×880 for Flux.2,
1152×896 for Krea 2).

### Write a diary line, not an image description
This is the part that decides whether the result works:

> **not** *"a man waiting for the bus outside a supermarket"*
> **but** *"had lunch at the motorway cafeteria with Gran"*, *"rubbish day but
> at least I got my cigarettes"*

A description is already framed — it says what to show, so the model centres the
subject and composes it. A diary line does not say what to show: the place, the
hour and the bystanders have to be worked out from it, and what comes back looks
like a photo taken in passing. Which is the whole aesthetic.

The **🎲 A random day** button fills the box from a bank of ready-made banal
sentences — as much to show the register expected as to unblock you. Edit the
line it gives you and generate.

### Why the enhancer matters more here
The style is an English prefix glued in front of your text — on its own it
translates nothing, and the image model has never heard of Flunch. With the
enhancer installed, the checkbox turns your sentence into *what the photo would
show*: the self-service cafeteria with its plastic trays and fluorescent
ceiling, the tobacconist's red sign.

It runs on **its own system prompt** (`xanax`), separate from the two used by
the normal tabs. Those ask for a named lens, a lighting setup and a composition —
applied here they would produce a *good* photograph, which is exactly this tab's
failure mode. The Xanax one bans photographic craft outright, forbids
beautifying, and keeps the output to two or three flat sentences: a long prompt
makes the model compose.

Without the enhancer the tab says so, and your sentence is sent **as is** —
write in English then, and say what is visible rather than what you did.

The exact prompt that produced the image is shown under it, a fixed seed replays
the same photo, and the usual `.txt` sidecar lands next to it in `outputs/`.


## Hardware & optimization

### Automatic optimization
The app detects your GPU (via `nvidia-smi`) and RAM, then chooses on its own:
- **diffusion quantization** by VRAM
  (`<8 GB → Q4_K_S`, `8–12 → Q4_K_M`, `12–16 → Q5_K_M`, `16–24 → Q6_K`, `≥24 → Q8_0`);
- **encoder quantization** by RAM (the text encoder is offloaded to RAM, so it
  costs no VRAM);
- **flags**: flash-attention (Turing / RTX 20xx and newer), CPU offload, VAE
  tiling, CLIP/VAE on CPU — enabled progressively as VRAM gets tighter;
- Pascal cards (GTX 10xx) → flash-attention disabled automatically (it's slow there).

The engine log also reports **how fast your models are read**. sd-cli already
prints the read time and the buffer size, on two separate lines; the app does
the division and, below 300 MB/s, says plainly that the models sit on a
mechanical drive and what that costs per image. The case that prompted it read
8.2 GB at **105 MB/s** — 65 s lost on every single image.

Multi-GPU: the largest card is used by default, changeable in **Settings**
(see [Multi-GPU](#multi-gpu)).

These map to stable-diffusion.cpp flags: `--diffusion-fa` (CUDA: faster + less
VRAM), `--offload-to-cpu`, `--vae-tiling`, plus GGUF quantization. On a recent
engine the app also separates **where computation runs** (`--backend`) from
**where weights live** (`--params-backend`). This matters on a second GPU behind
a slow PCIe link: an encoder can keep its weights on that GPU instead of staging
them from RAM. The old CLIP/VAE-on-CPU flags remain compatibility fallbacks.

### One question, not twenty
The Settings tab asks you **exactly one thing**, because it is the only thing
your hardware cannot answer for you:

> **More memory headroom · Balanced (recommended) · More detail**

Everything else — quantization, offload, tiling, flash-attention — is derived
from the detected card and simply *reported*, in consequences rather than flag
names ("the final image is assembled in pieces, so the card is not saturated at
the last moment" rather than `vae_tiling=True`). The three-notch choice shifts
the diffusion quant one rung along `QUANT_LADDER` and tightens or relaxes the
memory options with it; the line under the radio states the actual change
("model loaded as `Q5_K_M` instead of `Q4_K_M`").

A short **"Something specific going wrong?"** block maps symptoms to actions,
and two of its three answers deliberately point *elsewhere*: "too slow" is the
step count and image size in the generation tab, "images look dull" is the
prompt and the styles. Pretending everything is solved in Settings is what sent
people hunting through checkboxes in the first place.

**There is no Save button.** Every control applies immediately and says so, next
to itself. A Save button is one more chance to wonder whether the change was
taken into account — and the theme already saved itself, which made the rest
ambiguous.

**When the answer needs measuring, the app measures.** The multi-GPU placement
depends on the second card's PCIe link as much as on its memory, so instead of
asking you to bet there is a button that runs the comparison (see
[Measured hardware profile](#measured-hardware-profile-and-krea-int8-probe)).

Everything sd.cpp exposes and nobody needs to touch lives under a single folded
**🔧 Expert** section, which says in its first line that nothing in it is
required and that touching it turns automatic tuning off. `tests/test_ui_shape.py`
keeps the shape honest: no tabs inside the tab, at most three folded sections,
exactly one control visible up front on a single-GPU machine, and no Save button.

### Per-generation presets
`hardware.GENERATIONS` still holds a curated profile per RTX generation (**GTX
10xx → RTX 50xx**): real VRAM of the selected GPU, diffusion quant with a small
speed/quality bias per generation, encoder quant from RAM, memory flags
(flash-attention off on Pascal, VAE tiling / CPU offload on tighter cards). It is
no longer a row of five buttons in the interface — the three-notch choice covers
the same ground with one decision instead of five — but `generation_profile()`
remains the reference used by the auto profile and the tests.

### Multi-GPU
When **two or more GPUs** are detected, a **Multi-GPU** accordion appears with a
single mutually-exclusive strategy:
- **Single card (recommended)** — everything on the generation GPU, text encoder
  offloaded to RAM. The most reliable.
- **Text encoder on the 2nd card** — diffusion + VAE stay on the main GPU, the
  text encoder runs **and keeps its weights** on the other card (`--backend` +
  `--params-backend …,te=cudaN`). Frees VRAM on the main card without repeatedly
  staging the encoder through system RAM.
  **Refused automatically when that second card has no tensor cores** (Pascal,
  GTX 16xx) while the generation card does. Measured cost of getting this wrong:
  **38 s of prompt encoding per image** on a GTX 1080 Ti, against ~1 s on the
  RTX 3060 next to it — consumer Pascal runs fp16 at 1/64 of its fp32 rate, and
  prompt encoding is one big fp16 matmul. The card is still excellent at
  *holding* weights; it is bad at computing on them. In that case the encoder's
  weights go to RAM and its computation to the generation card
  (`--params-backend` sets residency, not where work happens), which costs no
  VRAM at all. The benchmark can still measure the refused placement on purpose.
- **Text encoder computed on the 2nd card, weights in RAM** — a compatibility
  fallback kept as a measurable option. It can win on unusual topologies, but
  normally loses to resident weights because it crosses PCIe repeatedly.
- **Auto-fit** — the published build 841 and newer source builds use different
  command syntax. Turboslop detects the installed binary's help and sends either
  the legacy switch or `--auto-fit on|off`. Current source builds compute on one
  GPU and place weights in GPU memory, RAM, another GPU or disk according to the
  available budget. This is not simultaneous computation on every GPU.
  Explicit multi-GPU computation requires backend assignments; benchmark a
  mismatched pair or a PCIe x4 secondary slot before selecting it.


A separate **prompt-enhancer GPU** can also be chosen (the text LLM runs there;
image generation and SDXL upscale always stay on the generation GPU). The GPU
picker and these strategies live in **Settings**.

### Samplers & schedulers
The built-in **presets follow the official sd.cpp docs** (`docs/flux2.md`,
`docs/krea2.md`): **Euler** sampler with the **scheduler left to the engine
default** (the docs never force one), at each model's documented steps/CFG
(Flux.2 Klein 4 steps · CFG 1.0; Krea 2 Turbo 8 steps · CFG 1.0). The dropdowns
still expose the full sd.cpp list for manual experimentation, **annotated per
model** — with a description card and a fold-out rationale right in the tab.
New entries need a recent engine (`update-engine.bat`).

#### Why most of the menu does not apply here
Three properties of our two models — read off sd.cpp itself, not guessed —
decide almost every verdict, and they rule out whole families at once:

1. **They are flow-matching models.** sd.cpp runs both in `FLUX_FLOW_PRED` /
   `FluxFlowDenoiser` (Krea 2 with a flow shift of 1.15). **Karras** and
   **Exponential**, which gain a lot on SD 1.5 / SDXL, were designed for EDM
   epsilon-prediction diffusion: their sigma spread does not match this
   trajectory.
2. **They are distilled at CFG 1.0.** There is no guidance to correct, so the
   whole **CFG++** family has nothing to do — and the negative prompt is ignored
   whichever sampler you pick.
3. **They run in very few steps** (4 for Flux.2 Klein, 8 for Krea 2 Turbo).
   **Ancestral/stochastic** methods re-inject noise that never gets reconverged;
   **multistep** methods need a history of evaluations that barely has time to
   exist.

**LCM** and **TCD** are not general-purpose options either: they are the
samplers of models distilled *by those methods*, which neither of ours is.

#### The verdicts, at a glance
⭐ recommended · ✓ usable · △ poorly suited · ⚠️ discouraged
*(in the menu itself, "usable" simply carries no mark)*

| Sampler | Flux.2 Klein (4 steps) | Krea 2 Turbo (8 steps) |
| --- | :---: | :---: |
| `euler` | ⭐ | ⭐ |
| `res_2s` · `euler_ge` | ✓ | ✓ |
| `heun` · `dpm++2m` · `dpm++2mv2` · `ipndm` · `ipndm_v` · `res_multistep` | △ | ✓ |
| `dpm2` · `ddim_trailing` · `lms` | △ | △ |
| `dpm++2m_sde` · `dpm++2m_sde_bt` · `er_sde` | ⚠️ | △ |
| `euler_a` · `dpm++2s_a` · `lcm` · `tcd` · `euler_cfg_pp` · `euler_a_cfg_pp` | ⚠️ | ⚠️ |

| Scheduler | Flux.2 Klein | Krea 2 Turbo |
| --- | :---: | :---: |
| `auto` (engine default) | ⭐ | ⭐ |
| `flux2` | ⭐ | △ |
| `discrete` | ✓ | ⭐ |
| `smoothstep` · `sgm_uniform` · `simple` · `logit_normal` | ✓ | ✓ |
| `ays` · `gits` · `kl_optimal` | △ | ✓ |
| `flux` · `beta` · `bong_tangent` | △ | △ |
| `karras` · `exponential` | ⚠️ | △ |
| `lcm` | ⚠️ | ⚠️ |

`auto` means *don't pass `--scheduler`*, so the engine picks: `flux2` for
Flux.2 Klein, `discrete` for Krea 2 — which is why those two are also marked ⭐
on their own model.

**What is worth actually trying:** on Flux.2 Klein, almost nothing beyond
`euler` — 4 steps leave no room, and `res_2s` is the only other one accurate
without a history to build. On Krea 2 Turbo the margin is wider: `res_multistep`,
`dpm++2m` and the `ays` scheduler (designed for small step budgets) deserve a
side-by-side run at a fixed seed.

These verdicts are **reasoned from the models' properties, not measured on a
benchmark**. They say where to aim your experiments, not what your eye will
prefer. The source of truth is [`atelier/sampling.py`](atelier/sampling.py);
`tests/test_sampling_docs.py` checks every key against the engine's own list so
a typo can't reach the menu.

### Cache acceleration (experimental)
**Settings → 🗃️ Cache acceleration** exposes sd.cpp's step-caching
([`caching.md`](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/caching.md)):
`easycache`, `dbcache`, `taylorseer`, `cache-dit` or `spectrum`, plus a free-form
option (e.g. `threshold=0.2`). It reuses near-identical computations across
diffusion steps. Honest note: it pays off mostly above ~10 steps — on 4–8-step
distilled models the gain is small and artifacts are possible, hence **off by
default**. Requires a recent engine (`update-engine.bat`).

### Measured hardware profile and Krea INT8 probe
**Settings → 🧪 Measure this machine** runs a fixed 512×512 / seed 424242 generation
at the model’s native step count (Flux.2: 4; Krea 2: 8) through every sensible placement: main GPU with RAM staging,
encoder resident on the second GPU, and the staged dual-GPU path.

**How it measures matters more than what it measures.** Each placement gets one
**discarded warm-up run** followed by **three timed runs**, and the reported
figure is the **median**, not a single sample. Without the warm-up, the first
placement tested pays the cold-disk cost of reading ~9 GB of GGUF while the
later ones hit the OS page cache — the ranking would then tell you the order the
tests ran in, not which placement is faster. Peak VRAM is reported **over the
baseline measured just before each run**, after unloading any resident engine.
Benchmarks disable resident generation and step caches for consistent comparisons.

A winner is only declared when it beats the runners-up by **more than the
measured spread** (max − min across the timed runs). When two placements sit
inside the noise, the report keeps the **simpler** one rather than pretending to
split them. The log streams while the test runs and **⏹️ Stop** ends it cleanly
after the run in progress. Running the test never alters your preferences —
applying the profile is a separate, explicit click.

The same block compares the normal **Krea 2 Turbo GGUF** with the optional
**INT8 ConvRot** checkpoint from `Comfy-Org/Krea-2` (sd.cpp gained INT8 ConvRot
support in build 817, August 2026). The RTX 3060 executes the INT8 kernels while
the oversized checkpoint is streamed from RAM. GGUF remains the default because
the two outputs still require a visual quality decision — and because none of
this is measured on your machine until you press the button.

### Direct convolution (memory)

**⚡ Acceleration (advanced)** also exposes `--diffusion-conv-direct` and
`--vae-conv-direct`. They swap the convolution algorithm: instead of **im2col**
— which unfolds the image into a large matrix before multiplying — the
convolution is computed directly. im2col is fast, but that intermediate buffer
is big; computing directly removes it.

What can honestly be promised: **less memory**. Speed depends on tensor shapes —
sometimes better, sometimes worse. **Measure it**, don't tick it on principle.
The clear case for turning it on is when you are close to running out of memory.

Both are recent sd.cpp options: the app **checks the installed binary** and
simply omits them if it doesn't know them, so an older engine cannot break on an
unknown argument. `update-engine.bat` to get them.

> **What about SageAttention or Triton?** They cannot be added, and it is not a
> matter of build flags. Both live in the **PyTorch** ecosystem: Triton is a
> kernel compiler driven from Python and JIT-compiled at runtime, and
> SageAttention is a pip package whose CUDA/Triton kernels hook into PyTorch's
> attention. stable-diffusion.cpp is C/C++ on GGML — no Python, no PyTorch, no
> runtime that could host a Triton kernel, and therefore no hook point. Getting
> quantized attention here would mean **reimplementing it as a ggml CUDA
> kernel** upstream, not flipping an option. What sd.cpp already has on that
> front is flash attention (`--diffusion-fa`, compiled in by default via
> `GGML_CUDA_FA`), plus GGUF quantization and the step caches above — which is
> where the actual speedups live.

### One-shot CLI by default, resident engine on demand
Generation runs through **one-shot `sd-cli`** by default: it gives the **live
step preview**, and it is the path every feature is verified against.

This section used to claim that reload cost was "handled by the OS disk cache".
A measured log says otherwise — 250 s for one Krea 2 Turbo image, of which
**80 s re-reading the model** and **38 s encoding the prompt**, against 117 s of
actual sampling. Paid again for every image. That claim was wrong and is gone.

So the **resident engine** is back as an opt-in, in **Settings → Expert**. It
runs `sd-server` — the same sd.cpp, shipped in the same archive, already sitting
in `bin/` — which keeps the model loaded and answers local HTTP requests
(`/sdcpp/v1/img_gen`, then polling `/sdcpp/v1/jobs/{id}`). The second image
avoids reloading the weights. New prompts still need encoding; there is no
guaranteed per-image speedup until measured on your hardware.

`sd-server` ships in the same archive as `sd-cli` — verified by listing the
contents of `sd-master-6b3edaa-bin-win-cuda12-x64.zip`, which contains
`sd-cli.exe` **and** `sd-server.exe` — but **only from the official upstream
release**, and only from a release recent enough to have the server example.
When the option cannot be offered, Settings says which of the two pieces is
missing and what to do about it, instead of hiding the checkbox.

It is deliberately **never mandatory**:
- LoRAs, the HD pass, step caches and auto-fit are
  **not** served — they fall back to `sd-cli` silently (the API ignores
  `<lora:…>` prompt tags by design, so serving them would quietly produce an
  image *without* the LoRA);
- reference editing can reuse the resident model when its API advertises
  `ref_images`; the vision projector is included when required;
- startup/protocol failures fall back to `sd-cli` with the reason in the log;
- user cancellation terminates the process, including during model loading;
  native running jobs cannot currently be cancelled through the job endpoint;
- a two-hour generation deadline stops a hung job without restarting it;
- **no live preview**: the image arrives at the end, and the log says so;
- the model holds VRAM, so `sd-cli` runs, Toolkit tools and trellis 3D all
  **stop the server first** and let it reload on the next image;
- changing model, quantization or residency restarts it — serving a different
  model than the one requested would be far worse than being slow.

(The earlier ComfyUI backend stays removed: too fragile.)

### Engine binary: official or self-built (CI)
By default `update-engine.bat` downloads the **official** prebuilt binary from
[`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
releases — the simplest, always-works path.

The engine binary comes from the **official upstream release**, and only from
there. The project used to publish its own CI build (arch-tuned for this
project's cards, `update-engine-ci.bat`); it was removed. It packaged a single
executable and therefore silently lacked `sd-server`, and maintaining a second
build chain to save a few percent was not worth its cost — nor the class of bug
where the two engines differ.

### Updating the engines

**The code and the engines update separately, and that is the trap.** Three
scripts, three targets:

| Script | Updates |
|---|---|
| `update.bat` | **the application** — code, from GitHub (see [Updating the app itself](#updating-the-app-itself)) |
| `update-engine.bat` | **sd.cpp** — latest official prebuilt binary (image generation) |
| `update-trellis.bat` | **trellis.cpp** — latest official Windows CUDA build (Image → 3D) |

The two engine scripts replace only the **engine binary**. sd.cpp updates are
first validated outside `bin/`, then swapped atomically; the previous working
build is retained for one-command rollback (`rollback-engine.bat` /
`./rollback-engine.sh`) and an `engine-manifest.json` records its exact commit,
archive checksum and supported options. DLLs from two releases therefore never
mix, while a broken download can never destroy the working install. **Models
are never re-downloaded** — including the ~10 GB trellis 3D set; reinstall
those from the **🧊 Image → 3D** tab if ever needed.

**When do you need to update the engine?** When a tab tells you to. The app
parses `sd-cli -h` and checks the options the current code actually needs, so a
missing capability is reported as a *feature* ("the HD tab will not work"), not
as a flag name. `maintenance.bat --all` does the purge and the engine update in
one go.

`update-trellis.bat` picks its archive from your card — see
[Image → 3D](#image--3d).

### Keeping the interface shallow
Two rules, enforced by `tests/test_ui_shape.py` rather than by good intentions:

- **No accordion lives inside another accordion** — the threshold is zero, not
  "reasonable". A fold inside a fold costs two clicks to reveal one option, and
  the second click is never announced. Where several sections must coexist, they
  are **tabs**: all of them visible without opening anything. That is why the
  536-style bank is now 🎨 Styles → three tabs (*Custom preset* · *Photo* ·
  *Artistic*), and why **Settings** is four tabs rather than a stack of six
  folds.
- **One-shot blocks disappear once they are done** — the seven "⚙️ Install …"
  panels are hidden on a machine where the tool is already installed, leaving a
  small **Reinstall / repair** button in their place. The state that matters is
  the one you live in, not the first launch.

Measured on the real rendered tree of a machine where every tool is already
installed: **62 visible accordions before, 33 after**, and accordion nesting
from **1 level to 0**. Part of that drop is simply the removal of a generation
tab; the rest is the two rules above.

### Interface language & theme
**The interface is English, and there is no language setting.** There used to be
one: the source strings were French and `atelier/i18n.py` held a French → English
dictionary that `t()` looked them up in. That layer had two costs. A string added
to a new tab was French until someone remembered to add a row to the table, so
the English half was only ever as good as the last person to update it. And the
table had to be kept in step with the code by hand, which is exactly the kind of
bookkeeping that quietly stops happening.

So the strings themselves are English now, and the table is gone.
`atelier/i18n.py` survives as a seam — `t()`, `to_source()` and
`translate_blocks()` are identity functions — so the ~40 call sites read the
same and a real language layer could come back at that one point. Engine logs,
progress lines and error messages are English too; there is nothing left that
speaks French to the user.

`tests/test_i18n.py` enforces it rather than trusting anyone's diligence: it
fails on any French string literal reaching the interface, detecting them by
accents, by a list of French words chosen to collide with no ordinary English
word, and by the metric units (`Go` / `Mo` / `Ko` / `To`) matched
case-sensitively so `To use` and `Go high resolution` stay legal. A short list
of accented English loanwords (`café`, `fête`, `naïve`…) is excepted rather than
weakening the accent rule.

**Code comments and docstrings are still French** and deliberately out of scope:
they are the source's own language, not the interface's, and nobody reads them
from the app.

**Theme** — **Settings → 🌍 Theme and accounts** switches between **Light** and
**Dark**. It is saved to `userdata/` and applied on **restart** (`run.bat` /
`run.sh`), because Gradio builds the interface once at launch.

---

## Upscaling

Four complementary tools live under **Toolkit**, in increasing order of
invention: ESRGAN enlarges, **HD** re-denoises with your own model, SeedVR2
restores, SDXL hallucinates.

### 🔼 Simple (ESRGAN, native sd.cpp)
Deterministic ESRGAN upscale via sd.cpp `--mode upscale`: **100% GPU, no PyTorch,
no prompt**. One-click downloads **all** models from
[`wbruna/upscalers-sdcpp-gguf`](https://huggingface.co/wbruna/upscalers-sdcpp-gguf)
(2x-ESRGAN, RealESRGAN_x4plus, 4xUltrasharpV10, 4x_foolhardy_Remacri…). Pick a
model (×2/×4 depending on its name); **Repeat ×2** chains two passes (a ×2 model
twice = ×4). Best for a clean, faithful enlargement.

**Tile size — where ESRGAN artifacts actually come from.** sd.cpp runs the RRDB
network on **128 px tiles** by default (25% overlap, smootherstep blend). A RRDB
decides how hard to sharpen from what it can see, and on 128 px it sees almost
nothing: two neighbouring tiles treat the same line differently. The feather
softens the seam but cannot reconcile two contradictory decisions — that is the
micro-staircasing on diagonals and the grain that changes character from square
to square. The app therefore sizes the tile itself, from your VRAM (1024 px at
16 GB+, 832 at 11 GB, 640 at 8 GB, 512 otherwise) and, when the image fits under
that cap, asks for **a tile at least as large as the image** — sd.cpp then takes
its untiled path and there is no seam at all, by construction. If the wider tile
runs out of memory the run is retried once at sd.cpp's 128 px and the log says
so. On an older `sd-cli` that predates `--upscale-tile-size`, the option is
simply not sent and the log tells you to update the engine.

**Repeating is worse than it looks.** Chaining passes runs the network on its
*own output*: pass 2 mistakes the high frequencies pass 1 invented for real
detail and sharpens them again, turning mild ringing into hard stair-steps. A ×4
model always beats a ×2 model run twice.

**Line art, comics and illustration.** The model matters more than the settings.
Photo-trained models (Remacri, Nomos, UltraSharp…) learned natural texture: on a
flat colour area they hallucinate grain, and along a clean ink line they ring.
The dropdown therefore tags each model — 🎨 **drawing / anime** vs 📷 **photo** —
lists the drawing ones first and preselects one, instead of defaulting to
whatever sorted first alphabetically.

**Bring your own models.** sd.cpp loads most `.pth` files directly, so the picker
accepts `.gguf`, `.pth` and `.safetensors`: drop a file in the upscalers folder,
hit **↻ Refresh**, and it appears. That opens the whole
[OpenModelDB](https://openmodeldb.info) catalog — filter on *anime* / *manga* /
*cartoon* for line art. GGUF still loads faster and avoids executing a pickle, and
you can convert one yourself with
`sd-cli --mode convert --model x.pth --output x.gguf`. Note that sd.cpp only
implements the **ESRGAN (RRDBNet)** architecture for image upscaling, so newer
SPAN / DAT / Compact models will not load.

### 🚀 HD (native sd.cpp *highres fix*)

The one that has no seams, because it never cuts the image up.

sd.cpp gained a native highres fix, and `sd_img_gen_params_t` carries both an
init image and the hires block — so a single `sd-cli` command does the whole
job: a very light img2img at the source size, then the enlargement (latent,
Lanczos or one of your ESRGAN models), then a **second denoise pass over the
entire image** at the final size. No PyTorch, no SDXL, **no tiles**.

Two things follow from that, and they are the reason this tab exists:

- **there is no seam to hide.** The creative SDXL upscale refines 1024 px tiles
  and blends them; a feather can smooth a border but cannot make two tiles agree
  about what they are drawing. Here the second pass sees the whole scene at once,
  so the question does not arise;
- **your model does the redrawing.** Krea 2 or Flux.2 add detail in the style
  they already know, instead of an SDXL from 2023 reinterpreting it.

Controls: the **factor**, the **added detail** (the hires denoise — the only
setting that really matters: 0.2 stays very close to the source, 0.5+ frankly
reinvents the material), the **intermediate enlarger**, and an optional short
description. The first img2img pass is fixed at a deliberately negligible
strength — sd.cpp computes `t_enc = steps × strength`, so it amounts to a single
step at very low sigma; it cannot be removed (the hires pass hangs off a
generation) so it is made harmless instead.

Sizes are aligned **up** to 16 px — the common divisor of every family's grid,
which leaves the app's own resolutions (1184×880, 1152×896…) untouched where a
64 px grid would move them. sd.cpp then aligns further up to its real multiple
if it needs to, and the log reports the size that actually came out rather than
the one that was requested. The final side is capped at 3072 px: past that the
model is outside its training scale and starts repeating patterns. Hitting the
cap lowers the *factor*, never the framing, and says so.

**VRAM is the real limit here, not the side length.** Refusing to tile has a
price: the second pass allocates a compute buffer proportional to the pixel
count, *on top of* the model weights already resident on the card. A measured
example — Krea 2 at 2304×1792 (4.13 Mpx) asks for a 4.62 GiB buffer while 8.4 GB
of weights are loaded: 13 GB total, which no 11–12 GB card can serve. So the
factor is budgeted rather than capped by a magic number: usable VRAM minus the
diffusion file's size on disk, divided by ~1200 bytes per pixel (the constant
comes from that same failure). Concretely, with Krea 2 at **Q5_K_M** (8.4 GB) an
11 GB card tops out near **×1.25** and a 12 GB card near **×1.5**; dropping to
**Q4_K_M** restores a full **×2** on 12 GB. A lighter quantization buys HD
factor.

**Segmented execution and memory budgets** (`--max-vram`) depend on the
installed engine generation. Published build 841 uses the explicit budget and
`--stream-layers` controls. Recent source builds automatically segment graphs,
evict idle weight copies and prefetch upcoming segments when needed; they no
longer expose `--stream-layers`. Turboslop probes the binary and only sends that
legacy switch when it is advertised.

A negative budget such as `-1` reserves roughly 1 GiB from free memory measured
at startup. It budgets managed weights and workspaces, not all physical GPU
allocations made by the driver or other applications. The HD tab sets a budget
and removes its conservative pixel estimate when the engine supports the flag.

On older engines, HD can retry once at the same size with layer streaming when
weights reside in RAM. On new engines this behavior is automatic, so repeating
the same run with an absent flag would not improve it. The existing OOM size
fallback remains available after the engine has exhausted its memory policy.

Neither the budget nor the segmentation is the safety net. **An out-of-memory
failure is caught, the factor is stepped down 20% and the run is retried** (twice
at most), and the log states what it settled on. sd-cli failures are now typed:
only a genuine VRAM error is retried, because it is the only one where trying
something smaller can succeed — everything else would fail identically. When even
the smallest attempt fails, the message names the two ways out (lower the factor,
or use the tiled ESRGAN → SDXL path, which fits in far less VRAM) instead of the
old bare "sd-cli exited with code 1".

Requires a recent engine. On an `sd-cli` that predates `--hires` the tab says so
and points at `update-engine.bat` instead of silently producing a plain image.

### 🌱 Restore (SeedVR2 3B / 7B)
Diffusion restoration/upscale using the standalone
[`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
engine, pinned to a known commit and installed in an isolated Python 3.12
environment. It restores natural detail more convincingly than ESRGAN while
staying closer to the source than the creative SDXL mode. GGUF weights download
automatically on first use and are SHA-256 verified by the upstream CLI.

Four weights are offered: **3B Q8** (the default), **3B Q4** (memory fallback),
**7B Q4** and **7B Q4 “sharp”**. A 7B is 4.76 GB of weights — it fits an 11–12
GB card with block swapping — keeps fine textures better and takes roughly twice
as long. The 3B has 32 transformer blocks, the 7B has 36; the swap slider is
clamped to whichever model is selected.

The dedicated **RTX 3060 12 GB + GTX 1080 Ti** preset keeps computation on the
RTX 3060 and uses the GTX 1080 Ti as an offload device. This is deliberate on a
PCIe x4 secondary slot: it avoids continuously splitting matrix operations
between mismatched GPUs. Start with **Q8, 2048 px, 16 swapped blocks, 1024 px
VAE tiles**. If memory runs out, try 24 then 36 blocks, or switch to Q4.

**Attention kernel.** The upstream CLI accepts `sdpa`, `flash_attn_2/3` and
`sageattn_2/3`. The app probes SeedVR2's own venv and asks for the fast kernel
only when the package is actually installed **and** the compute GPU is Ampere or
newer — Turing (RTX 2080 Ti) and Pascal (GTX 1080 Ti) cannot run FlashAttention 2
at all, so they stay on `sdpa`. Nothing to tick: `sdpa` is the default and works
everywhere. Note that neither FlashAttention nor SageAttention currently ships a
Windows wheel built against **torch 2.7.1 + cu126**, which is what this venv
uses; the published Windows wheels are cu128, and PyTorch's cu128 builds dropped
Pascal (`sm_61`), which would cost the GTX 1080 Ti as an offload device. That
trade is not worth it here, so the venv stays on cu126.

For a folder of images, use **Batch folder** in the same Restore tab. The app
passes the directory to SeedVR2 once, keeps its DiT and VAE caches warm across
the whole queue, and writes new PNGs without touching the originals. This avoids
paying model startup cost again for every image.

### 🔍 High resolution (Flux.2 as its own upscaler)
Runs the image back through **Flux.2 at its native resolution**, using it as
**both the reference and the starting latent**. Three details carry the whole
method, and none of them is obvious:

- the prompt says **“high resolution”**, not “upscale”. In training captions
  *upscaled* labels images that really were upscaled — carrying exactly the
  artefacts we are trying to avoid. *High resolution* labels photographs that
  were sharp to begin with. Different distribution, different output;
- the pre-enlargement is **bilinear**, deliberately bland. Lanczos adds ringing
  the model reads as detail and then amplifies;
- the same image goes in as the **reference** (content, through the VAE) *and*
  as the **starting latent** (structure). The engine always supported passing
  both; the interface sent one or the other, never both.

Denoise sits at **0.8** by default (the method's range is 0.7–0.9), and the
pass runs **8 steps**: Klein is distilled for 4, which at 0.8 denoise leaves
only 3 effective steps — too short to rebuild anything.

**This is not restoration.** At that denoise the model *redraws*: what survives
is plausibility, not fidelity. For a face that must stay the same person, use
**🌱 Restore** (SeedVR2). The two tools answer different questions.

Output is bounded on both ends: never below **1 MP** (Flux.2's own regime —
below it the model is out of its element anyway) and never past **3.7 MP**,
where it starts losing global coherence and the VRAM cost explodes. If the card
refuses, the target shrinks and the log says to what.

High denoise also drags colour toward the model's own prior — the original post
noticed the desaturation and suggested prompting against it. Prompting is a
wish; this tab does the arithmetic instead: the **original's low frequencies**
(hue, exposure, cast) are put back underneath the **result's high frequencies**
(the detail just added). On by default, one checkbox to turn off.

Credit: the method comes from a r/StableDiffusion post; the colour-matching
step and the memory ladder are ours.

### 🙂 Faces (GFPGAN · RestoreFormer++ · CodeFormer)
Rebuilds **faces only** — the rest of the image is untouched. This is the step
that ESRGAN and SeedVR2 cannot do: once a face is small or blurry, neither can
put clean eyes and a clean mouth back. Run it **last**, after the upscale; the
Upscale and Restore tabs both carry a **→ 🙂 Fix the faces** button that hands
the result straight over, so there is no file to find and re-upload.

**Three restorers**, installed together (~1.5 GB with the shared detector and
face parser) because none of them wins on every image:

| Model | Licence | Character |
|---|---|---|
| **GFPGAN v1.4** (default) | **Apache-2.0** | Gentler, preserves identity best |
| **RestoreFormer++** | **Apache-2.0** | Better on genuinely damaged photos, more aggressive |
| **CodeFormer** | **S-Lab 1.0 — non-commercial** | The only one with a fidelity dial |

The licence is shown **next to each model**, not buried in a footnote: it
decides what you may do with the result. If you sell your images, CodeFormer is
the one to avoid — which is why the default is GFPGAN.

**Faithfulness to the original face** (CodeFormer's `w`) applies to CodeFormer
only; 0.5 is almost always right, lower it for a badly damaged face and the
model invents more, raise it if the person stops looking like themselves. The
other two have no such dial and the runner says so instead of pretending. A
checkbox limits the pass to the main face instead of every detected one.

Each architecture has its own input convention, and getting it wrong is silent:
CodeFormer's reference pipeline works in `[-1, 1]` and is called directly (it is
the only one taking `weight=`), while GFPGAN and RestoreFormer go through
spandrel's descriptor, which knows each architecture's normalisation and returns
`[0, 1]`.

The implementation deliberately avoids `basicsr`, which the original CodeFormer
repository depends on: `basicsr` imports `torchvision.transforms.functional_tensor`,
removed in torchvision 0.17, so it no longer installs on our torch 2.4.1 base.
Instead the network comes from [`spandrel`](https://github.com/chaiNNer-org/spandrel)
(pure torch) and the detection / FFHQ-512 alignment / segmentation-masked
paste-back from [`facexlib`](https://github.com/xinntao/facexlib) — which is
exactly what upstream CodeFormer uses for that half. Three weights (~570 MB:
restorer, detector, face parser) are downloaded from the **official releases**
and **SHA-256 verified**, and facexlib is installed with `--no-deps` because its
`numba`/`filterpy` requirements only serve its video face tracker and would fight
our NumPy pin.

⚠️ CodeFormer is **non-commercial** (S-Lab License 1.0).

### ✨ Creative (SDXL, *Ultimate SD Upscale*)
Creative, Magnific-style upscale: pre-enlarge, then **refine tile by tile** with
SDXL img2img at low denoise. The model stays **resident** on the GPU so tiles are
fast; overlapping tiles are blended with a cosine feather for seamless joins, with
a **real-time preview**. Invents fine detail. This is an A1111-free
re-implementation. PyTorch + diffusers (**~9.5 GB**: SDXL base + fp16-fix VAE +
ControlNet Tile), installed in one click.

Controls:
- **SDXL model** — use the bundled SDXL Base 1.0, or drop your own SDXL
  checkpoint (`.safetensors`) into `tools_repo/upscale/checkpoints/` and pick it.
  **VAE** choice: external fp16-fix (recommended, avoids black images) or the
  checkpoint's **built-in VAE**.
- **Pre-upscale** — base enlargement before the SDXL tile refine: **Lanczos**
  (default) or any installed **ESRGAN** model (sharper, real detail). It runs
  **exactly one pass, never two**. Chaining is what manufactures the artifacts
  and aliasing this pass is supposed to avoid, and a soft base is the cheaper
  mistake: SDXL's refine puts detail back onto a soft base, but it *freezes*
  stair-stepped edges instead of fixing them. So when the model's factor falls
  short of the target the runner finishes in Lanczos and the log says so — and
  when the factor **overshoots** (a ×4 model for a ×2 target) the downscale that
  follows is free supersampling, which is the cleanest case available. Picking a
  pre-upscaler whose factor is *above* your target is the single best setting
  here.
- **Presets** — a dropdown that sets **the whole recipe**, not just a prompt:
  prompt, negative prompt, creativity, CFG, steps, structure locking and
  pre-upscaler. A line under the menu states what it just applied. See
  [Upscaling illustrations](#upscaling-illustrations-without-interpolation)
  below.
- **Creativity (denoise)** — 0.15 faithful → 0.75 inventive.
- **Negative prompt** — what SDXL is forbidden to add. The default is
  photo-oriented; on drawings it is what keeps grain and photo texture off the
  flat color areas.
- **🔒 ControlNet Tile** (optional) — conditions each tile on the source so you can
  push creativity higher **without drifting** from the original structure (the
  Magnific trick). Toggle + a *ControlNet fidelity* slider appear once it's
  installed (`xinsir/controlnet-tile-sdxl-1.0`, ~2.5 GB, included in the
  installer). Without it, it's plain low-denoise img2img — already very good.
- **Scale** — ×1.5 to ×8 (up to ~8K, capped at 8192 px). High factors mean many
  tiles → slow, and ~1–2 GB system RAM for the final assembly; VRAM stays constant
  (tiled).
- **Steps / tile**, **CFG**, **tile size** (640–1280).
- On < 12 GB VRAM, the model is automatically CPU-offloaded to avoid OOM.

> Use the right tool: **ESRGAN** is fast and deterministic; **SeedVR2** restores
> plausible detail with limited drift; **High resolution** re-renders through
> Flux.2 (best-looking, least faithful); **SDXL creative** is slower and
> explicitly invents detail; **Faces** fixes what all of them leave broken, and
> runs last.

#### Upscaling illustrations without interpolation

A photo-oriented upscale does three specific things to a drawing, and all three
have to be fixed together — which is why this is a preset and not a prompt:

1. **the base interpolates.** Lanczos does not add information, it averages
   pixels: linework goes soft and flat fills go mushy. Only an ESRGAN *trained on
   drawings* actually enlarges line art;
2. **the default negative prompt does not defend flat areas**, so SDXL happily
   lays photo grain and material texture over them;
3. **denoise around 0.40 redraws the linework**, which then wobbles — lines stop
   being the same lines.

The **🖍️ Illustration / comics — crisp linework, no interpolation** preset sets:
a **drawing ESRGAN** as the base (auto-picked from what you have installed,
preferring `RealESRGAN_x4plus_anime_6B`), a negative prompt aimed at
photorealism/grain/halos, **denoise 0.18**, **CFG 4.0**, and **ControlNet Tile at
0.85** so the structure is locked. At that point SDXL is no longer redrawing
anything — it only cleans up what the ESRGAN produced.

If you have **no drawing upscaler installed**, the preset says so explicitly and
warns that the base will stay on Lanczos: download the upscaler pack from the
**🔼 Upscale** tab first, otherwise the preset cannot do its main job.

For painted or brushwork illustration, **🎨 Painted illustration / concept art**
is the looser variant (denoise 0.35, ControlNet 0.7) — it keeps some material,
which is the point there.

---

## Outpaint

The **🖼️ Outpaint** tab extends an image **left, right, up, down — or all
around**, Midjourney-style.

> **Use it with an *edit* model** — Flux.2 Klein. This is not a
> preference, it is what makes the feature work at all.

**Why the model has to be an edit model.** An edit model receives the enlarged
canvas as a **reference image** (`-r`): its image conditioning tells it what the
scene actually contains, and an **auto-generated extension instruction** tells it
what to do with it. A plain text-to-image model gets none of that — in img2img it
only sees a noised latent, so it does not know what it is continuing and
**reinvents instead of extending**. No amount of tuning strength, feather or edge
fill fixes that; it is a limitation of the method, not a setting. The img2img path
is kept as a fallback so nothing is blocked, not because it produces good output.

That auto-generated instruction is what "no prompt" means here: you write
nothing, but the model still receives a precise directive naming which sides were
extended and telling it to continue perspective, lighting, palette and style
without touching the original or duplicating subjects.

**The pipeline** (`atelier/engine/outpaint.py`):

1. the canvas is enlarged in the chosen directions (snapped to 16 px, capped at
   2048 px per side — margins shrink proportionally if the cap is hit);
2. the new area is pre-filled: **neutral grey** for an edit model (an obviously
   empty zone reads as "fill this"; a fake backdrop would mislead it), or an
   edge-stretch/mirror fill for the img2img fallback, which has nothing else to
   go on;
3. **edit path** — canvas as `-r` plus the instruction, no strength, no mask.
   **Fallback path** — canvas as `-i` with a strength, plus an inpainting mask
   *if* the installed `sd-cli` has a mask option. That flag is **discovered at
   runtime** by parsing `sd-cli -h` (`sdcpp.supported_options` /
   `sdcpp.mask_flag`) rather than hard-coded, since its spelling varies between
   versions and between official and self-built binaries;
4. the result's **tone is matched back to the original** (per-channel mean and
   standard deviation, measured on the overlap region);
5. the **original is composited back on top**, with a feather at the seam.

Step 4 is not cosmetic. The model re-renders the *whole* canvas with punchier
contrast and saturation; pasting the untouched original back on top then leaves a
visibly duller rectangle in the middle, which a feather cannot hide because the
mismatch is global, not local. The fix corrects the *new* area toward the
original, never the reverse.

**Controls**

| Control | What it does |
| --- | --- |
| **Direction** | left / right / top / bottom / horizontal / vertical / all around |
| **Extension per side** | fraction of the original added to each chosen side (0.25 = +25 %); the resulting size is previewed live |
| **Model** | any installed model; sampler, CFG and steps follow its catalog defaults |
| **Prompt** | *optional* — leave empty for a neutral extension, fill it only to steer what appears in the new area |
| **Edge fill** | **Neutral grey** (default on an edit model) — the empty zone is unambiguous. **Blurred stretch** replicates the border outward, carrying color but no shape. **Mirror** gives perfect continuity on regular patterns, but reflects any subject near the edge and the model turns that reflection into a second real object — uniform backgrounds only |
| **Generation strength** | fallback path only, and greyed out on an edit model (which is driven by the instruction, not by a strength). High = invents freely; low = stays close to the pre-fill |
| **Feather** | width of the blend at the seam. Note it blends a band of roughly **2× its value** *inside* the original's border — that is what makes the seam disappear. **0 = hard paste**, original strictly untouched everywhere |
| **Tone match** | 0–1, how strongly the new area is pulled onto the original's contrast and color. Lower it only if the correction over-corrects on an unusual image |
| **Seed** | -1 = random; a fixed value replays the same extension |

**♻️ Re-extend the result** reloads the output as the new input, so extensions
can be chained (right, then up, …). Results land in `outputs/` with a `.txt`
sidecar recording model, seed and settings. Generation tabs have a
**🖼️ Send selection to Outpaint** button.

> The old **🧩 Centered outpaint** slider inside the Flux.2 tab is a different,
> experimental thing: symmetric only, edit-capable models only, prompt-driven.
> The dedicated tab supersedes it.

---

## Image → 3D

**🧰 Tools → 🧊 Image → 3D** turns one image into a **textured 3D mesh** (GLB)
through **[trellis.cpp](https://github.com/pwilkin/trellis.cpp)** (TRELLIS.2) —
a native CUDA binary on GGML, **no PyTorch**. Give it a sharp image of a
**single object** on a simple background; background removal is automatic.

> ⛔ **Windows CUDA only.** There is no Apple Silicon build and no Metal path.
> The tab says so on a Mac instead of offering an installer that would find
> nothing.

**Installation is two downloads**, both from the tab: the **binary** (~700 MB)
into `bin/trellis/`, and a **set of GGUF weights**
([`ilintar/trellis2-gguf`](https://huggingface.co/ilintar/trellis2-gguf)) into
`models/trellis/`. Three weight variants, and the intuition here is backwards:

| Variant | Size | Speed |
|---|---|---|
| **f16** (default) | ~16.5 GB | **the fastest** when it fits |
| **q8** | ~9.9 GB | slower |
| **q4** | ~6 GB | slower still |

In ggml, quantized weights are **dequantized on the fly at every computation**,
and a 3D workload does not amortize that overhead the way a long diffusion run
does. So quantizing here buys **memory, not speed** — which is the point when it
is what makes 1024/1536 reachable at all. You can install several and switch at
generation time.

**Stay at 512 below 16 GB of VRAM.** The 1024/1536 cascade is documented for a
16 GB card. Below that it does not fail cleanly — it **degrades the computation**
and returns a mesh made of blobs. No setting works around it: trellis has neither
offload nor tiling. To gain quality *without* touching the resolution, raise the
**UV atlas** (2048/4096) and the **decimation**: a well-textured 512 mesh beats a
botched 1024 one, at almost no VRAM cost.

**How the server is driven.** The Windows release ships `trellis-server.exe` —
an HTTP server, not a one-shot CLI. By default the app **starts it, posts the
image, takes the GLB and stops it**, so all the VRAM is released afterwards
(the low-VRAM strategy). A **⚡ Resident server** checkbox keeps it alive
instead: the next objects skip the ~30 s model reload, but the VRAM stays
occupied — stop it before generating images. Changing a *launch* setting
(resolution, decimation, atlas, GPU, texture) restarts the server automatically,
because those are not renegotiable per request; only the seed and background
removal can change without a reload.

**Controls**

| Control | What it does |
|---|---|
| **Input image** | one object, simple background |
| **Pad to square** | TRELLIS processes its input as a square — without this a non-square image comes out **distorted** |
| **Geometry resolution** | 512 (recommended, ≤ 12 GB) · 1024 · 1536 |
| **Decimation — target faces** | lower = lighter mesh (0 = engine default) |
| **UV atlas size** | texture resolution — the cheapest quality gain |
| **Background removal** | BiRefNet (quality, recommended) or threshold |
| **Geometry only** | skip the texture, faster |
| **Card used for 3D** | passed to the engine as its own `--gpu N` |
| **Seed** | shown under the result, to replay the same object |

The result is previewed in the browser and written to `outputs/`. Generation
tabs carry a **🧊 Send selection to Image → 3D** button.

**When it fails, the message is the real cause.** If the server dies
mid-generation the HTTP connection is cut and `requests` raises a bare
`ConnectionResetError`, which says nothing useful. The app captures the server's
own last lines and reports those instead — see the `no kernel image` entry in
[Troubleshooting](#troubleshooting), which is the one failure that used to be a
dead end and no longer is.

---

## Toolkit

One-click installable utilities (models pulled from Hugging Face, run as
subprocesses so torch DLLs never lock the UI process):

| Sub-tab | What it does | Add-on |
|---|---|---|
| **📝 Image → prompt** | hand it an image, get the prompt back ([below](#-image--prompt)) | Qwen2.5-VL-3B, ~7.5 GB |
| **🌐 Depth** | depth map — *Depth Anything V2* | ~100 MB |
| **✂️ Background removal** | cutout → transparent PNG — *RMBG-1.4* (**non-commercial**) | ~176 MB |
| **🪄 Cut out (SAM)** | click an object, extract it — *Segment Anything* (`facebook/sam-vit-base`) | ~375 MB |
| **🧩 Layers** | decompose into layers, write a PSD ([below](#layers--psd)) | shares SAM; CLIP optional (~600 MB) |
| **🔼 Upscale** | ESRGAN, deterministic ([Upscaling](#-simple-esrgan-native-sdcpp)) | GGUF upscaler pack, ~1 GB |
| **🚀 HD** | native sd.cpp highres fix, no tiles ([Upscaling](#-hd-native-sdcpp-highres-fix)) | none — uses your model |
| **🔍 High resolution** | Flux.2 as its own upscaler ([Upscaling](#-high-resolution-flux2-as-its-own-upscaler)) | none — uses your model |
| **🌱 Restore** | SeedVR2 3B/7B ([Upscaling](#-restore-seedvr2-3b--7b)) | isolated venv + GGUF |
| **🙂 Faces** | GFPGAN · RestoreFormer++ · CodeFormer ([Upscaling](#-faces-gfpgan--restoreformer--codeformer)) | ~1.5 GB, five weights |
| **✨ SDXL upscale** | creative, tile-by-tile ([Upscaling](#-creative-sdxl-ultimate-sd-upscale)) | ~9.5 GB with ControlNet |

Every generation tab has a **Send to Toolkit** control that pushes the selected
image straight into one of these, so there is no file to find and re-upload.

### 📝 Image → prompt
Give it an image, get back the prompt that would recreate it — then send that
prompt straight into a generation tab with one button.

**This is not captioning, and the difference is the whole point.** A vision
model left to itself says *"a photo of a cat on a sofa"*. That is a sentence
*about* the image; pasted into the Prompt field it produces something flat.
A prompt is an *instruction*: it names the medium, the light, the lens, the
palette and the framing, because those are the words a diffusion model actually
responds to. The system prompt therefore bans caption formulas outright, and the
lead-ins the model emits anyway ("This image shows…", "Sure, here is a
prompt:…") are stripped from the answer — the instruction lowers their
frequency, it does not reach zero, and one is enough to pollute the field.

Output is always **English**, whatever the interface language: that is what the
models were trained on.

**Repetition is the normal failure of this format, and it is handled in three
places.** A comma-separated keyword list never tells the model it is finished —
nothing in the grammar signals an end — so it loops: a real run produced a good
opening and then repeated *"high heels, fashion, modern, wet, rain"* until the
token budget ran out, 130 fragments for 50 useful ones. The fixes, in order of
how much they can be relied on:

1. the token budget is **matched to the requested length** (200 tokens for
   110 words, not 320 — that 65% of slack is exactly what the model fills with
   restatements);
2. `repetition_penalty` and `no_repeat_ngram_size=6` are passed to the
   generator, and the system prompt forbids restating an idea already written;
3. **the answer is deduplicated afterwards.** This is the one that guarantees
   the result rather than hoping for it: the output *is* a comma-separated
   list, so identical fragments can be dropped without losing anything. Order
   is preserved (the opening carries the subject, the light, the lens), and
   `wet pavement` collapses with `the wet pavement`. Prose answers — the plain
   description mode — are never deduplicated, since a comma there is grammar,
   not a separator.

**A second failure showed up in use, and it is worse than looping.** On a
photograph, the model finished with *"oil painting style, brushstrokes visible,
canvas texture evident"* — three fragments contradicting everything before
them, and enough on their own to make the diffusion model paint instead of
photograph. Same lesson, same shape of fix:

- the medium is now asked for **first**, not last. The drift happens at the
  *end* of a generation, when there is nothing real left to say; naming the
  medium while the model is still looking at the image pins it down;
- and it is **enforced afterwards**: whichever medium is established first wins,
  and later fragments from an incompatible vocabulary are dropped. First wins
  because the model describes what it sees before it starts confabulating — in
  the real case `depth of field` came nine fragments before `oil painting
  style`. Markers are deliberately unambiguous (`canvas texture` is a marker,
  bare `texture` is not): the cost of a wrong match is deleting a legitimate
  fragment.

**And the sampling was wrong for the job.** Describing is not creating. The
runner sampled at `temperature=0.7`, which is literally asking the model to pick
a less likely token now and then — on a description, that means inventing. It
reported *"dark red shoes"* for black ones. A single proposal is now decoded
**greedily**, with no sampling at all: the image does not vary, so there is
nothing to gain from varying the answer. Sampling only comes back when several
proposals are requested, and then at a low temperature. This is a fix at the
cause; a wrong colour cannot be detected from the text afterwards, and the
README will not pretend otherwise.

A tail cut off by the token limit is dropped too, but only when the runner
*knows* it was cut — the model emitted no end token. It is never guessed from
the text: `shallow dep` and `shallow` are indistinguishable without a
dictionary, and trimming a legitimate fragment is worse than leaving a stub.

Three modes, which ask for genuinely different things:

| Mode | What it writes | What it is for |
|---|---|---|
| 📸 **Recreate this image** | subject, setting, light, composition, colours, medium — 60–110 words | reproducing or varying an image on your own models |
| 🎨 **Style only** | medium, technique, light, palette, contrast, grain — **and not one word about the subject** | applying that look to a completely different subject |
| 🔍 **Plain description** | two or three sentences, no prompt jargon | knowing what is in an image |

The style mode's rule is strict on purpose: if the image is a red car in Rome,
neither "car" nor "Rome" may appear. A style that names its subject is not
transposable, and the mode would have no reason to exist.

**Model**: **Qwen2.5-VL-3B-Instruct** (~7.5 GB), the same family as the prompt
enhancer, loaded then unloaded per call so nothing stays in VRAM during
generation. Like the enhancer, it is *text* work, so it runs on the secondary
text GPU when you have one. It needs `transformers>=4.49` — the
`Qwen2_5_VLForConditionalGeneration` class does not exist in 4.48, verified —
so the installer pins that range for this add-on only. ⚠️ **Qwen Research
licence: non-commercial**, the same condition as the already-shipped enhancer.

### Layers → PSD

Cuts an image into layers and writes a **PSD** (and/or separate transparent
PNGs), either by letting SAM sweep the image or by **clicking the areas
yourself**. Same add-on as click-to-cutout — nothing extra to download.

**Read this before using it: the layers are flat cut-outs.** Move an object and
you reveal a hole, because the background behind it never existed. This is for
masking, retouching a region or exporting an element — *not* for recomposing a
scene. Doing it properly would mean inpainting behind every layer, one diffusion
pass each; that is a deliberate omission, not an oversight.

**Writing the PSD is done in-house, and that was the surprise.** No usable
library exists: [`psd-tools`](https://pypi.org/project/psd-tools/1.9.28) reads
well but "does not support editing of layer structure, such as adding or removing
a layer", and [`pytoshop`](https://pypi.org/project/pytoshop/) — the only writer
— is from 2018 and no longer builds on a modern Python (verified: its `setup.py`
fails against current setuptools). The format is documented and needs nothing but
`struct`, so `atelier/engine/psd.py` writes it directly: ~180 lines, pure Python,
no compiled dependency, runs as-is in the portable Python. Two decisions carry
the file size — each layer is **cropped to its bounding box** (a PSD stores the
layer position, so keeping the full canvas for a 200 px object would multiply the
weight by twenty) and channels use the format's native **PackBits RLE**. A
512×384 three-layer file lands at 44 KB instead of 1 MB uncompressed. The output
is validated against `psd-tools` in the test suite — an independent reader is the
only honest way to check you produced a valid file rather than one that merely
pleases you.

**The two real difficulties are not in the plumbing.** SAM segments *appearance*,
not meaning: on a photo it happily returns forty to eighty nested masks — a
shirt, a button, a fold, a reflection — and, worse, "zones" made of specks
scattered across the whole frame. And SAM gives **no depth order**: that comes
from Depth Anything V2 when it is installed — median depth under each mask, far
to near. Without it, large areas go to the back, which is an approximation and is
labelled as one rather than presented as a result.

**Large images used to be the wall, and it was an ordering problem.** The two
operations that decide the cut — *are these the same zone?* and *do they
touch?* — are **quadratic** in the number of zones, and they were running at
full resolution. Measured on a 4096×4096 image: one IoU costs **77 ms**, so 150
zones spend **14 minutes** just de-duplicating, plus 70 s on adjacency. Two
changes fixed it. Each mask is reduced **once** to a 256×256 *coverage grid*
(the fraction of lit pixels per block, 262 KB instead of 16.8 MB) and every
comparison happens there — 1770 IoU pairs drop from 136 s to **0.07 s**, and the
approximation stays within 0.02 of the exact value, far from the 0.75 duplicate
threshold. And near-duplicates are now dropped **before** cleaning rather than
after: a 24-point sweep probes the image 576 times and returns the sky or the
tarmac dozens of times over, while cleaning costs 0.7–1.8 s *per mask*. Filtering
300 raw masks went from **3.4 minutes to 26 seconds**, and duplicates are
rejected as SAM produces them, so memory stays near 400 MB instead of 5 GB.

**Everything useful happens in the cleanup**, in `atelier/engine/masks.py` —
written in plain numpy rather than pulling in scipy or OpenCV, since the
segmentation add-on is heavy enough and these operations are a few dozen lines.
Connected components use a union-find over per-row *runs*, not pixels: a
million-pixel union-find in Python would take seconds, while the number of runs
is in the thousands (18 ms on a 1200×900 mask).

- **Split into connected pieces.** A "layer" made of thirty specks in the four
  corners is not a layer, it is noise no editor can use. Each piece becomes its
  own zone, and pieces under the minimum area vanish.
- **Fill interior holes**, which is what gave the cut-outs their swiss-cheese
  look. A notch *open to the edge* is kept — it is part of the silhouette.
- **Keep contained zones.** Being inside a larger mask does not mean redundant,
  it means *in front*: the car on the road, the figure against a wall, the
  window on a façade. An earlier coverage filter deleted exactly those, and it
  was the single worst behaviour of the first version.
- **Merge only what actually touches.** With CLIP installed, pieces sharing a
  label are glued back into one object — body, door and wheel become a car.
  Adjacency is measured **pixel to pixel**, on a reduced grid. It used to be
  measured on *bounding boxes*, and that was catastrophic: on a wide photo a
  car's box covers half the frame, so a chain of boxes linked the car, the
  spray at the far end and the fence in the background into a single 25.7%
  "vehicle" layer. Two more guards came with the fix — an uncertain label never
  merges (propagating a coin toss glues unrelated things), and no merged group
  may exceed 35% of the image, because at that size it is a background, not an
  object.
- **Disjoint layers.** Fronts are subtracted from backs, so showing every layer
  reproduces the source image exactly and no pixel is painted twice — verified
  in the tests by compositing the PSD back and comparing to the original.
  Cutting drops layers from the *middle* of the list, so it now reports which
  ones survived: the caller used to truncate its label list from the end, which
  shifted every name after the first casualty — the car took the wall's name.
  Layers reduced to a thin fringe by whatever sits in front are dropped too.
- **Feathered edges** (1 px), because a binary mask pasted as-is has the
  staircase border that gives automatic cut-outs away.
- **Names you can read.** "Zone 9 — 0.48%" teaches nobody anything; layers are
  named from what is already known about them — depth band, position, dominant
  colour, size: *"foreground · bottom · orange — 3.8%"*. Colour matching weights
  lightness over hue, otherwise charcoal grey gets called dark green.

**With CLIP installed, the tool stops seeing shapes and starts seeing objects.**
It is an optional add-on (`openai/clip-vit-base-patch32`, ~600 MB, one click, on
the same footing as Depth) and it does three things that change the result, not
just the labels:

- **It merges the pieces of one object.** SAM returns "body", "door" and "wheel"
  as three masks. Labelled *vehicle* and adjacent, they become **one layer** —
  what a human calls a car. Adjacency is required: two cars at opposite ends of
  the frame share a label but are not the same object.
- **It discards what is nothing.** A flat fill, a patch of blur, a meaningless
  fragment. A zero-shot classifier cannot say "nothing" — forced to choose, it
  labels a blurry piece of asphalt *car*. So the vocabulary carries deliberate
  **junk categories** whose only job is to absorb those, plus a margin test:
  a label that wins by a hair is a coin toss, not information.
- **It orders the stack by meaning** when Depth is not installed — sky at the
  back because it is the sky, not because it is large. Each vocabulary entry
  declares a typical depth for exactly this.

The vocabulary lives in `atelier/engine/vocab.py` as **data, not code**: that is
the file to edit when labels land wide, with no change to the pipeline. Each
entry carries several phrasings, because CLIP scores an image against a *text* —
"a car" and "a racing car seen head-on" do not score alike on the same crop, and
the best variant wins. Crops are fed as the bounding box with margin, with the
outside of the mask **faded toward neutral grey** rather than cut to black: a raw
box drowns a thin object in its surroundings, while a black cut-out strips the
context CLIP was trained on.

### Prompt enhancer (AI)
The **✨ Enhance prompt** button (in each generation tab) runs a small instruct
LLM (*Qwen2.5-3B-Instruct*, PyTorch ~6 GB, one-click install) that rewrites your
idea into a detailed **English** prompt (subject, lighting, composition, style).
The model is loaded then unloaded per call → **no VRAM conflict** with generation.
It outputs only the enhanced prompt, injected straight into the prompt field. The
system prompt **detects intent from keywords** (medium/style/subject/mood) and
keeps the output medium-coherent; a **strength** selector (Light / Medium / Strong)
controls how far it expands. Krea 2 uses a Krea-specific system prompt.

**Several proposals at once.** Pick 1, 2 or 4 under **🎨 Enhancement options**;
they are produced in a *single* model load (`num_return_sequences`), so four cost
barely more than one. They appear under the prompt field and clicking one puts it
in the prompt. A grey line under the menus states in plain words what the button
will do before you press it.

**The active style preset is passed in as a constraint**, so the LLM writes *with*
it rather than against it — no camera, lens, lighting or processing wording that
would contradict the style, and no restating of the prefix itself.

---

## Convert to GGUF

**⚙️ System → 🔧 Convert to GGUF** quantizes a model you got from somewhere else
— a **checkpoint / safetensors / diffusion** file — into a lighter **GGUF** that
fits on your card. It is **100% CPU** (no diffusion involved): a few minutes
depending on the size and the disk, done once, and then you reuse the GGUF.

1. Drop your model into **`models/custom/`**, then hit **↻ Refresh**.
2. Pick the quantization.
3. **Convert** — the GGUF is written into the same `models/custom/` folder and
   becomes selectable as a **local model** in the generation tabs (through their
   *Refresh local files* button).

| Quantization | Bits | Notes |
|---|---|---|
| `f16` | 16 | no loss, and no saving either |
| `q8_0` | 8 | near lossless, larger |
| `q5_1` | 5 | **recommended** — the size/quality compromise |
| `q4_1` | 4 | light |
| `q4_0` | 4 | the lightest |

⚠️ sd.cpp only implements `q8_0 / q5_1 / q5_0 / q4_1 / q4_0 / f16` **when
converting** — there are **no k-quants** here. That is a limit of the engine's
conversion path, not of the app; the k-quants you see in the catalog were
produced upstream.

---

## Sharing on your LAN

Colleagues can generate from their **Mac/PC** using **your** machine and its GPU,
without installing anything — just a link in a browser.

1. On your PC, run **`run-lan.bat`** (instead of `run.bat`).
2. The address to share is printed, e.g. `http://192.168.1.42:7860`.
3. Colleagues on the **same Wi-Fi/network** open it in their browser. That’s it.

Options:
- **Password**: `run-lan.bat --auth name:password` (prompted on connect).
- **Firewall**: on first launch Windows may ask to allow Python — accept (private
  networks). Otherwise allow port 7860 in the firewall.

> Generations run **on your PC**: don’t turn it off during use. One generation is
> processed at a time (automatic queue).

---

## Distributing a portable package

To share the GUI so friends install nothing:

1. On a machine where **everything already works** (Python + engine in `bin\`),
   run **`make_portable.bat`**.
2. It produces `Turboslop-5000-portable.zip` with the code, the **portable
   Python** and the **engine** — but no models.
3. Friends **unzip** and run **`run.bat`**. No GitHub download needed: they only
   fetch the **models** from the Model Catalog tab (via Hugging Face).

This works even if someone’s network filters GitHub — the engine is already in the
ZIP. To update the GGUF engine later, run **`update-engine.bat`**.

---

## Models & sources

`config/models.yaml` is the single source of truth (sources, defaults, presets).
Quantization tokens (`{quant}` for diffusion, `{enc_quant}` for the encoder) are
resolved from your hardware; the downloader picks the closest matching file.

**Flux.2 Klein 9B** (family `flux2`, edit model)
- diffusion — [`unsloth/FLUX.2-klein-9B-GGUF`](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) (distilled, 4 steps, CFG 1.0)
- VAE — [`Comfy-Org/flux2-klein-9B`](https://huggingface.co/Comfy-Org/flux2-klein-9B) (`flux2-vae.safetensors`)
- text encoder — [`unsloth/Qwen3-8B-GGUF`](https://huggingface.co/unsloth/Qwen3-8B-GGUF) (official Qwen3-8B, via `--llm`, offloaded to RAM)

**Krea 2 Turbo** (family `krea2`, sd.cpp)
- diffusion — [`vantagewithai/Krea-2-Turbo-GGUF`](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF) (8 steps, CFG 1.0)
- text encoder — [`Qwen/Qwen3-VL-4B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (official Qwen3-VL-4B-Instruct, via `--llm`, offloaded to RAM)
- VAE — [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) (`wan_2.1_vae.safetensors`)


> Why the Klein source changed too: the previous repo
> (`leejet/FLUX.2-klein-9B-GGUF`) publishes **only Q4_0 and Q8_0**. Measured
> against the real fallback logic, *every* card ended up on Q4_0 (5.6 GB) — a
> 12 GB RTX 3060 asking for `Q5_K_M` and a 16 GB RTX 4080 asking for `Q6_K`
> alike. The current repo carries `Q2_K` → `Q8_0` with genuinely distinct sizes
> (`Q5_K_S` 6.94 GB ≠ `Q5_K_M` 7.02 GB), same base model and same filenames, so
> the targeted quant actually exists. On 12 GB that is a whole quality tier
> recovered: 7.02 GB instead of 5.62 GB.

> Why the Turbo source changed: the previous repo (`realrebelai/KREA-2_GGUFs`,
> `TURBO/` folder) had no `Q5_K_M` and no `Q2_K`. On a 12 GB card the auto
> profile asked for `Q5_K_M` and silently fell back to `Q5_K_S`. The current
> repo carries the full ladder from `Q2_K` (4.9 GB) to `Q8_0`, so the targeted
> quant actually exists — and small cards finally get a rung. Already-downloaded
> files are untouched; the change only affects new downloads.
>
> Honest footnote: in both `vantagewithai` repos the `_K_S` and `_K_M` files of
> a given bit width are **byte-identical in size** (Q5_K_S and Q5_K_M are both
> 8.87 GB). So what the switch really buys is the missing rungs — `Q2_K` at the
> bottom, and no more "quantization adjusted" fallback notice — not a smaller
> or larger file at Q5.


**Upscalers** — [`wbruna/upscalers-sdcpp-gguf`](https://huggingface.co/wbruna/upscalers-sdcpp-gguf) (ESRGAN), [`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (SeedVR2), `stabilityai/stable-diffusion-xl-base-1.0` + `madebyollin/sdxl-vae-fp16-fix` (creative), [`sczhou/CodeFormer`](https://github.com/sczhou/CodeFormer) + [`xinntao/facexlib`](https://github.com/xinntao/facexlib) (faces).

To delete a model, use **🗑️ Delete** in the Model Catalog — shared files
(encoders/VAEs used by another model) are preserved.

---

## Project layout

```
app.py                       # Gradio entry point + presentation() (theme/CSS/head)
config/
  models.yaml                # catalog: sources, defaults, presets (source of truth)
  style_presets.json         # bundled system-prompt styles
  krea2_styles.csv           # 139 photographic styles (© ghleg, MIT)
  krea2_art_styles.csv       # 397 artistic styles
atelier/
  settings.py                # paths + persisted preferences (userdata/)
  hardware.py                # GPU/RAM detection + optimization profiles
  quant.py                   # the quantization ladder
  registry.py                # catalog, file resolution, status, recommendations
  downloader.py              # on-demand Hugging Face / Civitai downloads
  styles.py                  # system-prompt / style presets
  sampling.py                # samplers & schedulers: verdicts and their reasoning
  benchmark.py               # "measure this machine": placements, medians, spread
  inventory.py               # what is on disk, with sizes, by category
  storage.py                 # moving models to another drive (+ junctions/symlinks)
  imgcheck.py                # image-display diagnostic (cache, MIME, last upload)
  diagnostics.py             # machine report for bug traces
  net.py                     # LAN address discovery
  i18n.py                    # identity seam — the interface is English (see above)
  engine/
    sdcpp.py                 # build/run sd-cli commands, error typing, DiskWatch
    sdserver.py              # resident engine (sd-server HTTP), opt-in
    generate.py              # generation pipeline (model + hardware + LoRA) + ESRGAN + HD
    highres.py               # 🔍 High resolution: Flux.2 as its own upscaler
    outpaint.py              # directional outpaint: canvas plan, fill, tone match, composite
    trellis.py               # trellis.cpp server: image → GLB, transient or resident
    tools.py                 # PyTorch tools as subprocesses (depth, bg, SAM, faces, SeedVR2…)
    masks.py                 # mask cleanup and layer naming — pure numpy
    psd.py                   # PSD writer — pure Python, no compiled dependency
    vocab.py                 # CLIP scene vocabulary as DATA, for layer labelling
  ui/
    theme.py                 # theme + CSS
    widgets.py               # shared image/gallery button lists (Gradio 6)
    preview.py               # data: URI fallback preview
    generate_tab.py · xanax_tab.py (hard-wired style) · library_tab.py
    toolkit_tab.py · outpaint_tab.py · threed_tab.py
    settings_tab.py · manage_tab.py · convert_tab.py
scripts/
  get_sdcpp.py               # downloads the stable-diffusion.cpp binary
  get_trellis.py             # downloads the trellis.cpp binary + GGUF weights
  update_app.py              # update.bat: code update, manifest, backup, rollback
  maintenance.py             # purge, orphan detection, engine capability check
  _torch_setup.py            # shared PyTorch-CUDA install helpers
  setup_tools.py             # installs the PyTorch add-ons
  setup_seedvr2.py           # installs SeedVR2 in its own Python 3.12 venv
  tools/_device.py           # CUDA / Metal-MPS / CPU picker shared by the runners
  tools/run_*.py             # inference runners (depth, rembg, sam, layers, describe,
                             #   enhance, face, ultimate_upscale)
tests/                       # 400+ unit tests, run with `python -m unittest discover -s tests`
```

---

## Managing disk space & uninstalling

The **🧹 Manage & help** tab is the single place to see what the app has
downloaded and to reclaim space. It lists every item with its **size**, grouped
into four categories:

| Category | What's in it |
|---|---|
| **Engines** | `sd-cli` (stable-diffusion.cpp) and the trellis.cpp 3D binary |
| **Models** | each catalog model separately, the ESRGAN upscaler pack, the ~10 GB trellis 3D set |
| **Toolkit add-ons** | depth, background removal, SAM, CLIP labelling, image → prompt, prompt enhancer, face restoration, SeedVR2, creative SDXL upscale |
| **Your data** ⚠️ | LoRAs, custom models, generated images/3D, temp files |

Tick what you want to remove, tick **“I confirm”**, then delete. Sizes refresh
after each operation.

- Everything outside **Your data** is **re-downloadable** from inside the app
  (Model Catalog, one-click installers).
- Items marked **⚠️** are *yours* — LoRAs, hand-placed/converted models and your
  generated images. Deleting them is not recoverable from the app.
- **Temp files** (`tmp/`) are always safe to clear.
- Deletion is restricted to paths **inside the project folder** — the tool
  never touches anything elsewhere on your disk, and base folders are recreated
  right after.

### Moving models to another drive

Models don't have to live inside the project folder. In **🧹 Manage & help →
📁 Models location** you can point them at any absolute path — typically a fast
**NVMe** (much quicker loads) or a bigger drive:

- **📦 Move models here** — transfers the existing files, then saves the new
  location. Same drive = instant; across drives = a real copy (can take a
  while). Free space is checked first, and if any file fails to move the
  location is **not** switched.
- **🔗 Point here without moving** — reuse a folder that already contains your
  models (e.g. shared with another install).
- **↩️ Back to the project folder** — restore the default `models/`.

The setting is stored as `models_dir` in `userdata/preferences.json` and applies
on **restart** (several modules resolve the path at import time).

**Moving only *some* items** — a second accordion lets you relocate **selected
items** (e.g. the ~16 GB trellis 3D set) to another drive and leaves a
**junction/symlink** behind. The app keeps finding them at the original path, so
there is **no setting and no restart** involved; a **↩️ Bring back** button undoes
it. Handy to keep image models on the NVMe while parking bulky ones elsewhere.

> The destination drive must stay connected — relocated items become unreachable
> if it isn't. On Windows the link is a directory *junction*, which does not
> require administrator rights.

> Repeatedly loading models does **not** wear out an SSD — flash endurance is
> consumed by *writes* (TBW), not reads. Putting models on your NVMe is exactly
> what it's for.

### What else lives in Manage & help

- **📖 In-app documentation for every option** — generation tabs, settings and
  hardware, Outpaint, Image → 3D, Convert to GGUF, the Toolkit, and network &
  maintenance. It is the fastest way to know what a slider actually does, and it
  is written to be read *while* the slider is in front of you.
- **🩺 Diagnose image display** — the three-layer test for the broken-image-icon
  problem, described in [Troubleshooting](#troubleshooting). Press the button
  instead of opening the browser console.
- **🌐 Network, sharing & maintenance** — what `run-lan.bat`, `maintenance.bat`
  and `update.bat` each do, in the same place as the buttons that need them.

---

## Gradio version

The app targets **Gradio 6** (`gradio>=6.0,<7` in `requirements.txt`) and will
not run on 5.x — the 6.0 release removed parameters it used. `maintenance.bat`
names the problem if an old version is still installed, rather than letting it
surface as a `TypeError` while the interface is being built.

What the move changed, and why it is worth knowing:

- **`show_download_button` / `show_copy_button` are gone**, replaced by a
  `buttons=[…]` list. The catch is the new default for an image:
  `["download", "share", "fullscreen"]`. That **share** button posts to Hugging
  Face Spaces Discussions — meaningless in an app running on your own machine,
  and it never appeared under Gradio 5 locally. Every image and gallery here
  therefore declares its buttons explicitly, from the named lists in
  [`atelier/ui/widgets.py`](atelier/ui/widgets.py); `tests/test_gradio6.py`
  fails if a component is added without them.
- **Theme, CSS and `<head>` moved** from the `Blocks(...)` constructor to
  `launch(...)`. Forget them and nothing breaks — the interface simply renders
  unstyled — so they are grouped in `app.presentation()` and a test checks they
  reach `launch()`.
- **`launch(show_api=…)` is gone**; API visibility is now set per event
  listener. Nothing to expose here anyway.
- The install is **118 MB lighter** (200 MB → 82 MB for the `gradio` package),
  which matters for the portable Windows folder.

What it did **not** change: the cross-volume upload race described below is
present in Gradio 6 exactly as in 5.50 — same `os.rename`, same fallback to a
background copy, just moved to another module. The fix in `app.py` is still
required.

## Troubleshooting

- **“sd-cli binary not found”** → re-run `install.bat`, or download the engine
  manually. On a portable Windows install there is no global `python`; use the
  embedded one:
  `python\python.exe scripts\get_sdcpp.py --variant cuda`
  (fetches the **win-cuda12** build *and* the **cudart** runtime side by side).
- **“No NVIDIA GPU detected”** → check drivers / `nvidia-smi`.
- **An image shows as a broken-image icon** — the browser got a response, just
  not an image.
  **The main cause, found and fixed, was self-inflicted.** Pinning the image
  cache inside the project (see below) put it on a different drive from
  `%TEMP%` on any machine whose project does not live on `C:`. Gradio
  moves an uploaded file with `os.rename`, which cannot cross volumes; it then
  falls back to copying **in a background task** — *after* already answering
  with the final path. The browser asks for the image immediately,
  `FileResponse` puts the size of the **partial** file in `Content-Length`,
  then keeps reading as the copy grows it, and h11 cuts the response with
  `LocalProtocolError: Too much data for declared Content-Length`. Truncated
  response, broken icon — while the tool itself gets the complete file once the
  copy finishes, which is why the image was *used* correctly but never
  *displayed*. Measured on an 11 MB upload: the endpoint returned a path to a
  **0 KB** file, and the next request announced 65 536 bytes and delivered
  65 536 of 11 234 505. The upload's temp file is now created next to the
  cache, so `os.rename` is an in-place rename again and there is nothing left
  to copy afterwards. `tests/test_upload_volume.py` fails if that ever stops
  being true.
  Two earlier causes had already been closed off. Gradio does not serve images
  from where they live; it copies them into a cache and serves *that*, and the
  cache defaulted to `%TEMP%` — a folder Windows Storage Sense, disk cleanup and
  antivirus all consider fair game. When the copy vanishes the request 404s and
  the image breaks, intermittently, which is what makes it maddening to
  diagnose. The cache is now pinned to `tmp/gradio` inside the project. And
  anything the app hands over *by path* rather than through that cache — a
  generated image sent to a tool, a live preview, a mask — used to get a **403**,
  because `allowed_paths` was never declared at launch; `outputs/` and `tmp/` are
  now declared (and only those: on `--listen` that list is what the machine
  exposes). A third was closed off later: the cache was pinned with
  `os.environ.setdefault`, so a `GRADIO_TEMP_DIR` already present in your
  environment — left by another Gradio app or an old install — silently took
  precedence and put the cache back in `%TEMP%`, reintroducing the exact bug the
  line exists to prevent. It is now set outright.
  **If it still happens, don't open the browser console** — go to
  **⚙️ System → 🧹 Manage & help → 🩺 Diagnose image display** and press the
  button. It answers in three layers, because the first two can pass while
  imports still break:
  1. **The chain itself** — writes a file into the cache and serves it over
     HTTP, naming the causes that actually bite on Windows: a cache on a
     **network or removable drive**, a **full disk**, an **unwritable folder**,
     a **path too long** for the Win32 API, and an **antivirus** locking each
     new file (visible as an abnormal read-back time).
  2. **One tile per format** (PNG, JPEG, WEBP, GIF, BMP) with the MIME type
     Python resolves for each. On Windows `mimetypes` initialises from the
     **registry**, so a missing or hijacked entry makes Gradio serve that
     format as a download and the browser shows nothing — which hits *some*
     extensions and spares the rest, exactly the shape of a bug that breaks
     imports while the general test passes. The tile that fails names the
     format.
  3. **Your last imported file**, displayed, with what it really is: size,
     dimensions, actual format. This is the one that settles it. If the file
     is not listed at all, the **upload** is failing, not the display. If it
     is listed, the report calls out a **truncated file**, an **extension that
     does not match the content** (a `.png` that is really a JPEG), and images
     so large the browser gives up decoding them.

  Meanwhile, the two tools you drive by **clicking on the image** — 🪄 Cut out
  (SAM) and 🧩 Layers in manual mode — carry a **🖼️ Fallback preview** panel
  (folded, so it costs nothing when the normal thumbnail works). It shows the
  same image as a `data:` URI: the pixels travel inside the page, so there is
  no request, no path, no MIME type and no filename involved — it cannot fail
  for any of the reasons above. It is for *seeing*, not for clicking: the click
  still happens on the component itself.
  Note that emptying `tmp/` from **Manage & clean** *while the app is running*
  legitimately breaks already-displayed images until you reload the page.
- **Model shows “to download”** → Model Catalog tab → **Download**.
- **Out of memory** → Settings: lower the quantization, enable offload/tiling, or
  reduce the resolution. For very tight setups, try a per-generation preset.
- **A Toolkit tool runs on CPU (very slow)** → its installer prints
  `CUDA: True/False`; if False, fix NVIDIA drivers and reinstall the tool.
- **Creative SDXL upscale OOM** → lower the scale or tile size (it auto-offloads
  under 12 GB, but a huge target can still exceed memory).
- **The console fills with `StarletteDeprecationWarning: 'HTTP_422_...' is
  deprecated`** → noise from Gradio's own code, one line per queued request, so
  a single generation buries the console. Nothing to fix on our side and it will
  go away with a Gradio update; it is filtered out. It escaped the existing
  filter for a precise reason: `StarletteDeprecationWarning` subclasses
  **`UserWarning`**, not `DeprecationWarning`. The new filter is deliberately
  narrow — silencing every `UserWarning` from Gradio would also have hidden
  *"A function returned too many output values"*, which is exactly what revealed
  that nine **Stop** buttons computed a cancellation message and threw it away
  (`outputs=None`). Pressing Stop now writes that message where you can see it:
  the status line, or appended to the log rather than replacing it.
- **`ConnectionResetError [WinError 10054]` in the console** → harmless, and
  silenced since. It is a known Python bug on Windows (bpo-39010): when the
  browser drops a connection (F5, tab closed, cancelled image load), asyncio's
  Proactor loop calls `socket.shutdown()` on an already-dead socket and prints
  `Exception in callback`. The request is already finished server-side. `app.py`
  now intercepts it *and* completes the cleanup the raised exception used to
  skip — the socket was actually leaking, which is why Python also logged
  `unclosed transport`. Only connection errors are caught; anything else still
  propagates.
- **Image → 3D: `CUDA error: no kernel image is available for execution on the
  device`** → the installed binary has no machine code for your card. Because
  CUDA errors are *sticky*, the failure surfaces on the next ggml op — usually
  `IM2COL` — which sends the diagnosis off in the wrong direction.
  For a long time this was a dead end: upstream's `CMakeLists.txt` **overrode**
  the complete architecture list its own CI passed, pinning trellis's kernels
  (`deform_conv.cu`, `decimate_qem.cu`) to `86;120` — so only RTX 30xx and 50xx
  got machine code, and updating changed nothing. The fix was to install the
  Vulkan build.
  **v0.6.0 (19 Aug 2026) fixed it upstream**: the pin became a mere default
  (`if(NOT CMAKE_CUDA_ARCHITECTURES)`), and the release now ships **two** CUDA
  packages — verified in `.github/workflows/release.yml` at that tag:
  `cuda` (CUDA 13.1) builds `75;80;86;89;90;120` — **Turing and newer** — and
  `cuda12` (CUDA 12.9) builds `60;61;70` — **Pascal and Volta**. So an RTX
  2080 Ti and a GTX 1080 Ti each have their package now.
  The installer picks the archive **from your card**, and **Vulkan stays the
  fallback** for anything neither list covers (and when the card cannot be
  identified). The only substitution allowed is *toward* Vulkan: swapping
  `cuda` for `cuda12` would install a binary built for architectures the card
  does not have. Press **⬆️ Update the binary** in the 3D tab; the ~10 GB of
  models are not re-downloaded. Force a choice with
  `python scripts/get_trellis.py --binary --force --backend cuda12`.
  The app surfaces the diagnosis itself: when the trellis server dies
  mid-generation the HTTP connection is cut and `requests` raises a bare
  `ConnectionResetError`, which says nothing — the real cause is captured from
  the server's output and reported instead.
- **A Toolkit add-on fails at import (`DTensor`, `diffusers`, numpy…)** → a
  shared package drifted. All add-ons share one Python, so the last installer to
  run decides the versions. Run `maintenance.bat`: it names the offending
  package, then reinstall that add-on.

---

## Acknowledgments

This project is just glue around other people's hard work. Heartfelt thanks to
everyone below — all credit for the models and tools goes to their original
authors. Please read and respect each model's own license on its page.

### Engine & framework
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** —
  **leejet** & contributors. The inference engine this whole project rests on.
- **[Gradio](https://github.com/gradio-app/gradio)** — the web UI (6.x).
- **[PyTorch](https://pytorch.org)**, **[Hugging Face](https://huggingface.co)**
  `transformers` / `diffusers` / `huggingface_hub` — the optional Toolkit tools.

### Image models
- **Flux.2 Klein** — base model by **Black Forest Labs**; GGUF by
  [Unsloth](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) (the app moved
  off `leejet/FLUX.2-klein-9B-GGUF`, which publishes only Q4_0 and Q8_0 — see
  [Models & sources](#models--sources)); VAE by
  [Comfy-Org](https://huggingface.co/Comfy-Org/flux2-klein-9B); text encoder
  **Qwen3-8B** by **Alibaba / Qwen team**
  ([official GGUF by Unsloth](https://huggingface.co/unsloth/Qwen3-8B-GGUF)).
- **Krea 2** — base model by **Krea AI**; GGUF by
  [vantagewithai](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF)
  (previously [realrebelai](https://huggingface.co/realrebelai/KREA-2_GGUFs),
  which was missing rungs of the ladder); text encoder
  **Qwen3-VL-4B-Instruct** by **Alibaba / Qwen team**
  ([official GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF));
  **WAN 2.1** VAE by **Alibaba / Wan team**, repackaged by
  [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged).
- **TRELLIS.2 / trellis.cpp** — the native engine by
  [pwilkin](https://github.com/pwilkin/trellis.cpp); GGUF weights by
  [ilintar](https://huggingface.co/ilintar/trellis2-gguf); TRELLIS by
  **Microsoft Research**.

### Upscalers
- **ESRGAN models (GGUF)** collected by
  [wbruna](https://huggingface.co/wbruna/upscalers-sdcpp-gguf) — including
  **Real-ESRGAN** (Xintao Wang et al., Tencent ARC) and community models
  (UltraSharp, foolhardy Remacri, Nomos, LSDIR, NickelbackFS, StarSample…). Credit
  to each upstream author; see the repo for individual sources/licenses.
- **SeedVR2 3B / 7B** standalone integration by
  [numz](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), using the
  upstream Q8/Q4 GGUF models and low-VRAM block swapping.
- **Face restoration:** **CodeFormer** by
  [Shangchen Zhou et al. (S-Lab, NTU)](https://github.com/sczhou/CodeFormer),
  non-commercial S-Lab License 1.0; detection, alignment and face parsing by
  **facexlib** ([Xintao Wang](https://github.com/xinntao/facexlib)); network
  implementation from **spandrel**
  ([chaiNNer](https://github.com/chaiNNer-org/spandrel)).
- **Creative upscale (Ultimate SD Upscale style):** **SDXL** by
  [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0);
  fp16-fix VAE by [Ollin Boer Bohan / madebyollin](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix);
  **ControlNet Tile** by [xinsir](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0).
  The tiled-redraw method is inspired by **Ultimate SD Upscale**
  ([Coyote-A](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111)).

### Toolkit
- **Depth Anything V2** —
  [depth-anything](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) team.
- **RMBG-1.4** background removal — **BRIA AI**
  ([briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4), **non-commercial** license).
- **Segment Anything** — **Meta AI**
  ([facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)).
- **Image → prompt** — **Qwen2.5-VL-3B-Instruct** by **Alibaba / Qwen team**
  ([Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)),
  Qwen Research licence (non-commercial).
- **Prompt enhancer** — **Qwen2.5-3B-Instruct** by **Alibaba / Qwen team**
  ([Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)).
- The **Krea prompting guide** informed the Krea 2 enhancer system prompt.

### Built with
- **[Claude](https://claude.ai/code)** (Anthropic) — vibe-coded iteratively in
  natural language.

This is an independent, non-commercial hobby project, **not affiliated with or
endorsed by** any of the above. If you are an author and want a credit corrected
or removed, please open an issue.


### Reliable ×4 enlargement (Turboslop 5000)

- **Toolkit → Upscale** keeps native sd.cpp ESRGAN. Choose a ×4 model and one
  pass for ×4. Output dimensions are validated; only a confirmed GPU memory
  error triggers smaller tiles. Temporary files are isolated per job.
- **Toolkit → ×4 Faithful** installs optional Spandrel/PyTorch and the official
  RealESRGAN_x4plus baseline. Click **Refresh models** after installation.
  Import native ×4 RGB `.pth` / `.safetensors` weights for DAT, HAT, RealPLKSR,
  or other architectures supported by Spandrel. These models do not run in
  sd.cpp simply because their file extension is accepted.
- Spandrel uses overlapping, feathered tiles with a context border, CPU
  assembly, and GPU tile retries without lowering the requested resolution.
  Half precision is used only on CUDA when the model supports it; a full
  precision checkbox is available. No prompt or diffusion pass is involved.
- Original file uploads preserve alpha and embedded RGB ICC profiles. Alpha
  is enlarged separately with Lanczos; it is not reconstructed by the RGB
  network. Output is 8-bit PNG. EXIF orientation is applied before enlargement.
- Model choice still matters: compare full-size crops of text, faces, diagonal
  edges and flat colours. No neural model guarantees recovery of true missing
  detail. 4xNomos8kDAT by Helaman is an optional photo candidate (CC-BY-4.0):
  https://openmodeldb.info/models/4x-Nomos8kDAT .
- SeedVR2's existing resolution control is now accurately labelled as the
  maximum output edge. It is separate from the exact ×4 workflow.

Validation: pixel/metadata tests and native error-handling tests run without a
GPU. Actual Spandrel model inference, Windows CUDA installation, VRAM usage and
visual quality require validation on the target PC; no GPU speedup is claimed.
