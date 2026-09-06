#!/usr/bin/env python3
"""Turboslop 5000 — studio d'inférence d'images en local (Gradio).

Onglets : Génération (Flux.2 Klein 9B / Krea 2 Turbo, GGUF) · Xanax (style figé) ·
Catalogue de modèles ·
Toolkit (profondeur, détourage, SAM, upscale) · Outpaint · Image → 3D · Réglages.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

# Le Python portable n'ajoute pas le dossier projet au chemin d'import.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# --------------------------------------------------------------------------- #
#  Cache de fichiers de Gradio : DANS le projet, pas dans le dossier temporaire
#  du système.
#
#  Gradio ne sert pas les images depuis leur emplacement d'origine : il en copie
#  une version dans son cache, et c'est CETTE copie que le navigateur demande.
#  Par défaut ce cache vit dans %TEMP% (Windows) ou /tmp — deux endroits que le
#  système se croit autorisé à vider quand bon lui semble : Storage Sense, le
#  nettoyage de disque, un antivirus, ou simplement un redémarrage. La copie
#  disparaît alors sous les pieds du navigateur, la requête renvoie 404, et
#  l'image affiche une icône cassée — de façon intermittente, ce qui est la
#  signature du problème.
#
#  Le placer sous tmp/ le met à l'abri de ces nettoyages, et le rend visible
#  dans « Gestion & nettoyage » avec le reste. À définir AVANT d'importer
#  gradio : la variable est lue au chargement du module.
#
#  Écriture FERME, pas setdefault : une variable GRADIO_TEMP_DIR déjà présente
#  dans l'environnement (posée par une autre application Gradio, ou restée
#  d'une ancienne installation) reprenait la main en silence et remettait le
#  cache dans %TEMP% — c'est-à-dire qu'elle réintroduisait exactement le bug
#  que cette ligne existe pour empêcher, sans que rien ne le signale. Le
#  diagnostic de « Gestion & aide » affiche le dossier réellement utilisé.
# --------------------------------------------------------------------------- #
os.environ["GRADIO_TEMP_DIR"] = os.path.join(_ROOT, "tmp", "gradio")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

# Avertissements de dépréciation de Gradio : masqués pour ne pas inquiéter au
# démarrage. Ils annonçaient les retraits de la 6.0, désormais tous traités.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

# Gradio appelle des constantes Starlette dépréciées à CHAQUE requête mise en
# file (`HTTP_422_UNPROCESSABLE_ENTITY`), ce qui noie la console sous des
# dizaines de lignes identiques pendant une génération. Rien à corriger de
# notre côté : c'est du code de Gradio, et ça disparaîtra avec une mise à jour.
#
# Le filtre précédent ne pouvait pas l'attraper : `StarletteDeprecationWarning`
# hérite de **UserWarning**, pas de DeprecationWarning. D'où un filtre séparé,
# et volontairement ÉTROIT — masquer tous les UserWarning de Gradio aurait
# aussi caché « A function returned too many output values », qui vient de
# signaler un vrai défaut chez nous (des boutons Stop qui calculaient un
# message puis le jetaient).
try:
    from starlette.exceptions import StarletteDeprecationWarning as _SDW

    warnings.filterwarnings("ignore", category=_SDW, module="gradio")
except Exception:  # noqa: BLE001 - la classe peut disparaître en amont
    warnings.filterwarnings(
        "ignore", message=r".*HTTP_\d{3}_\w+.* is deprecated", module="gradio")

import gradio as gr


def _patch_gradio_client() -> None:
    """Contourne un bug de gradio_client sur les schémas booléens (au démarrage)."""
    try:
        import gradio_client.utils as gcu
        _orig = gcu._json_schema_to_python_type

        def _safe(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            return _orig(schema, defs)

        gcu._json_schema_to_python_type = _safe
    except Exception:  # noqa: BLE001
        pass


def _upload_on_the_same_volume() -> None:
    """Le dépôt d'une image doit atterrir sur le MÊME disque que le cache.

    C'est la cause de l'icône d'image cassée à l'import, et c'est une
    régression que la ligne GRADIO_TEMP_DIR ci-dessus a introduite.

    Le trajet, dans Gradio 5.50 : le corps de la requête est écrit dans un
    `NamedTemporaryFile()` SANS `dir=` — donc dans %TEMP%, sur `C:` — puis
    déplacé vers le cache par `os.rename` (`routes.py`). Quand la destination
    est sur un AUTRE disque, `os.rename` lève, et Gradio se rabat sur une
    copie EN TÂCHE DE FOND :

        try:
            os.rename(temp_file.file.name, dest)
        except OSError:
            files_to_copy.append(...)          # copie plus tard
        output_files.append(dest)              # ... mais on répond tout de suite

    La réponse part avec le chemin final alors que la copie n'a pas commencé.
    Le navigateur demande aussitôt l'image ; `FileResponse` lit la taille du
    fichier PARTIEL, l'annonce en `Content-Length`, puis continue de lire
    pendant que la copie l'agrandit. h11 s'en aperçoit et coupe :

        LocalProtocolError: Too much data for declared Content-Length

    Réponse tronquée, donc icône cassée — mais le fichier finit bien d'être
    copié, ce qui explique le symptôme déroutant : l'image ne s'affiche pas
    alors qu'elle est correctement utilisée par l'outil. Et c'est
    intermittent : les petits fichiers gagnent la course, les gros la perdent.

    Tant que le cache était dans %TEMP% (le défaut de Gradio), départ et
    arrivée étaient sur le même volume, `os.rename` réussissait et la course
    n'existait pas. En déplaçant le cache dans le projet — pour le protéger du
    ménage de Windows — on a créé le cas « deux disques » sur toute machine
    dont le projet ne vit pas sur `C:`.

    Le correctif ne touche pas au trajet : il fait simplement écrire le
    fichier temporaire à côté de sa destination. `os.rename` redevient un
    renommage sur place, instantané, et il n'y a plus rien à copier après
    coup.
    """
    try:
        import gradio.route_utils as ru

        upload_tmp = os.path.join(_ROOT, "tmp", "upload")
        os.makedirs(upload_tmp, exist_ok=True)
        _orig = ru.NamedTemporaryFile

        def _same_volume(*args, **kwargs):
            kwargs.setdefault("dir", upload_tmp)
            return _orig(*args, **kwargs)

        ru.NamedTemporaryFile = _same_volume
    except Exception:  # noqa: BLE001
        pass


def _disable_brotli() -> None:
    """Désactive la compression Brotli de Gradio : son middleware calcule mal le
    Content-Length et casse le service des images (erreurs « Too much/little data
    for declared Content-Length »), ce qui faisait planter l'upscale ET l'aperçu
    temps réel. On le rend transparent (compression inutile en local)."""
    try:
        import gradio.brotli_middleware as bm

        async def _passthrough(self, scope, receive, send):
            await self.app(scope, receive, send)

        bm.BrotliMiddleware.__call__ = _passthrough
    except Exception:  # noqa: BLE001
        pass


def _quiet_connection_reset() -> None:
    """Windows : « ConnectionResetError [WinError 10054] » dans la boucle asyncio.

    Quand le navigateur ferme brutalement une connexion (F5, onglet fermé,
    chargement d'image annulé), la boucle Proactor de Windows appelle
    `_call_connection_lost`, dont le bloc `finally` fait un
    `socket.shutdown()` sur une socket déjà morte. Ça lève, asyncio l'imprime
    en « Exception in callback », et ça inquiète pour rien : la requête est
    finie côté serveur. Bug Python connu (bpo-39010).

    On ne se contente PAS d'avaler l'exception : elle interrompt le `finally`
    en plein milieu, donc `close()`, le détachement du serveur et le drapeau de
    fin ne s'exécutent jamais — la socket fuirait. On termine donc le ménage
    nous-mêmes. Seules les erreurs de connexion sont interceptées ; tout le
    reste continue de remonter normalement.
    """
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport as _T
    except Exception:  # noqa: BLE001
        return
    orig = _T._call_connection_lost

    def _call_connection_lost(self, exc):
        try:
            orig(self, exc)
        except (ConnectionResetError, ConnectionAbortedError):
            try:
                if getattr(self, "_sock", None) is not None:
                    self._sock.close()
                self._sock = None
                server = getattr(self, "_server", None)
                if server is not None:
                    server._detach()
                    self._server = None
                self._called_connection_lost = True
            except Exception:  # noqa: BLE001
                pass

    _T._call_connection_lost = _call_connection_lost


_patch_gradio_client()
_upload_on_the_same_volume()
_disable_brotli()
_quiet_connection_reset()

from atelier import APP_NAME, __version__, hardware, i18n, net, settings
from atelier.ui.convert_tab import build_convert_tab
from atelier.ui.generate_tab import build_generative_tab
from atelier.ui.library_tab import build_library_tab
from atelier.ui.manage_tab import build_manage_tab
from atelier.ui.outpaint_tab import build_outpaint_tab
from atelier.ui.settings_tab import build_settings_tab
from atelier.ui.threed_tab import build_threed_tab
from atelier.ui.theme import CSS, theme
from atelier.ui.toolkit_tab import build_toolkit_tab
from atelier.ui.xanax_tab import build_xanax_tab

# Force le thème choisi (clair/sombre) quel que soit le réglage du navigateur/OS.
def _head_for(mode: str) -> str:
    mode = "dark" if mode == "dark" else "light"
    return (
        "<script>"
        "if(!new URLSearchParams(window.location.search).has('__theme')){"
        "const u=new URL(window.location);"
        f"u.searchParams.set('__theme','{mode}');"
        "window.location.replace(u);}"
        "</script>")


def presentation() -> dict:
    """Thème, CSS et `<head>` — à passer à `launch()`, plus au constructeur.

    Gradio 6 a déplacé ces trois paramètres de `Blocks(...)` vers `launch(...)`.
    Les regrouper ici plutôt que de les recopier au point d'appel évite le
    piège de la migration : oubliés, rien ne casse et rien ne prévient —
    l'interface s'affiche simplement sans son thème.
    """
    prefs = settings.load_prefs()
    return {"theme": theme(), "css": CSS,
            "head": _head_for(prefs.get("theme", "light"))}


def build_app() -> gr.Blocks:
    settings.ensure_dirs()
    gpus = hardware.detect_gpus()
    sd_cli = settings.find_sd_cli()
    # Thème / CSS / <head> ne se posent plus ici : voir presentation(),
    # passé à launch().
    with gr.Blocks(title=f"{APP_NAME} {__version__}") as demo:
        # En-tête sur une ligne : titre, sous-titre, puis une pastille qui dit
        # sur QUOI ça tourne. C'est l'information qu'on veut avoir sous les yeux
        # en permanence quand on choisit une résolution ou un facteur d'upscale
        # — pas enfouie dans Réglages.
        _subtitle = i18n.t("Local image generation")
        if gpus:
            _best = max(gpus, key=lambda g: g.vram_gb)
            _chip = (f"<span class='chip ok'>{_best.name} · "
                     f"{_best.vram_gb:.0f} GB</span>")
        else:
            _chip = (f"<span class='chip warn'>{i18n.t('mode CPU')}</span>")
        gr.HTML(
            f"<div id='atelier-header'><h1>🎨 {APP_NAME}</h1>"
            f"<span class='sub'>{_subtitle} · v{__version__}</span>"
            f"{_chip}</div>")

        # Alertes de démarrage : UN bandeau compact, pas un empilement. Deux
        # blocs Markdown pleine largeur coûtaient une centaine de pixels du
        # premier écran, en permanence, pour un message qu'on lit une fois.
        # Sur Mac Apple Silicon, detect_gpus() renvoie le GPU intégré : pas
        # d'avertissement, il n'y a rien à installer. L'alerte GPU ne vise que
        # les PC où une carte NVIDIA est attendue mais absente.
        alerts = []
        if sd_cli is None:
            alerts.append(i18n.t(
                "**`sd-cli` binary not found** — run `install.bat` / "
                "`install.sh`."))
        if not gpus:
            alerts.append(i18n.t(
                "**No GPU detected** — CPU mode (very slow). Check your "
                "NVIDIA drivers / `nvidia-smi`."))
        if alerts:
            gr.Markdown("⚠️ " + "  ·  ".join(alerts),
                        elem_id="atelier-alerts")

        # Image en attente d'envoi vers le Toolkit : (chemin, destination).
        pending_toolkit = gr.State(None)
        # Image en attente d'envoi vers l'onglet « Image → 3D » (chemin).
        pending_3d = gr.State(None)
        # Image en attente d'envoi vers l'onglet « Outpaint » (chemin).
        pending_outpaint = gr.State(None)
        # Champs « Prompt » des onglets de génération, indexés par modèle.
        # « 📝 Image → prompt » y écrit DIRECTEMENT : c'est le seul flux qui
        # remonte des Outils vers la génération, et il ne peut pas passer par
        # un State consommé au changement d'onglet — une sélection
        # programmatique ne déclenche pas `Tabs.select`.
        prompt_boxes: dict = {}
        # Six onglets racine, pas onze. Au-delà, Gradio replie la barre dans un
        # menu « … » : sur la version précédente, « Gestion » et « Réglages »
        # étaient littéralement invisibles au premier coup d'œil. Le regroupement
        # n'est donc pas cosmétique — il rend deux fonctions atteignables.
        #
        # La règle de rangement : ce qui PRODUIT une image reste à la racine ;
        # ce qui la RETOUCHE va dans « Outils » ; ce qui administre la machine
        # va dans « Système ».
        with gr.Tabs() as tabs:
            prompt_boxes["flux2-klein-9b"] = build_generative_tab(
                "flux2-klein-9b", "🟣 Flux.2 Klein 9B",
                pending_toolkit=pending_toolkit, tabs=tabs,
                pending_3d=pending_3d, pending_outpaint=pending_outpaint)
            prompt_boxes["krea2-turbo"] = build_generative_tab(
                "krea2-turbo", "⚡ Krea 2 Turbo",
                pending_toolkit=pending_toolkit, tabs=tabs,
                pending_3d=pending_3d, pending_outpaint=pending_outpaint)
            # « Xanax » : style figé, aucun réglage de style exposé. Un seul
            # onglet pour les deux modèles — ils partagent tout sauf le moteur.
            build_xanax_tab("💊 Xanax")
            build_library_tab()

            # « Outils » : tout ce qui part d'une image existante. Les envois
            # « depuis la génération » visent l'onglet racine ; chaque sous-onglet
            # se sélectionne ensuite via son propre gestionnaire (voir plus bas).
            with gr.Tab("🧰 Tools", id="tools"):
                with gr.Tabs() as tool_tabs:
                    build_toolkit_tab(pending_toolkit=pending_toolkit,
                                      tabs=tabs, parent_tabs=tool_tabs,
                                      prompt_boxes=prompt_boxes)
                    build_outpaint_tab(pending_outpaint=pending_outpaint,
                                       tabs=tabs, parent_tabs=tool_tabs)
                    build_threed_tab(pending_3d=pending_3d, tabs=tabs,
                                     parent_tabs=tool_tabs)

            with gr.Tab("⚙️ System", id="system"):
                with gr.Tabs():
                    build_settings_tab()
                    build_manage_tab()
                    build_convert_tab()

    return demo


def _print_lan_banner(port: int, auth: bool) -> None:
    urls = [f"http://{ip}:{port}" for ip in net.lan_ips()]
    line = "═" * 64
    print("\n" + line)
    print("  " + i18n.t("{app} is reachable on your local network!").format(
        app=APP_NAME))
    print("  " + i18n.t("Share this address with colleagues (Mac/PC, same "
                        "Wi-Fi),"))
    print("  " + i18n.t("to open in Safari or Chrome:"))
    for u in urls or [f"http://<IP-de-ce-PC>:{port}"]:
        print(f"      →  {u}")
    if auth:
        print("  " + i18n.t("(they will be asked for a username/password)"))
    print("  " + i18n.t("If access fails: allow the port in the Windows "
                        "firewall."))
    print(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=f"{APP_NAME} {__version__}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="lien public temporaire gradio.live")
    ap.add_argument("--listen", action="store_true",
                    help="expose on the local network (0.0.0.0)")
    ap.add_argument("--auth", default=None,
                    help="password-protect: user:password")
    args = ap.parse_args()

    host = "0.0.0.0" if args.listen else args.host
    auth = None
    if args.auth and ":" in args.auth:
        u, p = args.auth.split(":", 1)
        auth = (u, p)

    demo = build_app().queue()
    port = net.find_free_port(args.port, host=host)

    if args.listen:
        _print_lan_banner(port, auth is not None)

    # `show_api` n'existe plus dans Gradio 6 : la visibilité de la page d'API
    # se règle écouteur par écouteur (`api_visibility`). Sans intérêt ici —
    # l'application est locale et ne publie rien.
    demo.launch(server_name=host, server_port=port, share=args.share,
                auth=auth, inbrowser=not args.listen,
                allowed_paths=settings.served_paths(), **presentation())


if __name__ == "__main__":
    main()
