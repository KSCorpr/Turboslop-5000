#!/usr/bin/env python3
"""Maintenance : vérifie l'installation et nettoie ce qui traîne.

`update.bat` fait désormais les mises à jour proprement (il sait supprimer ce
qui a disparu du projet). Ce script reste utile pour deux choses : rattraper
les copies mises à jour à la main — dézipper par-dessus AJOUTE et REMPLACE,
mais n'efface JAMAIS ce qui a été retiré en amont — et vérifier que
l'installation est saine.

  • supprime le CODE des fonctions retirées (table REMOVED_FEATURES) ;
  • CHIFFRE les DONNÉES qu'elles ont laissées (poids, dépôts clonés) sans les
    supprimer — plusieurs gigaoctets ne s'effacent pas sans prévenir ;
  • repère les ADD-ONS orphelins de tools_repo/ (dossiers ne correspondant à
    aucun add-on du code actuel) et les MODÈLES orphelins de models/ (plus
    référencés par le catalogue) ;
  • purge les __pycache__ (.pyc d'anciens modules) et le dossier tmp/ ;
  • vérifie que tout compile, que le catalogue YAML est valide, que les
    dépendances et le binaire sd-cli sont présents, et qu'aucun fichier de
    l'application n'a disparu depuis la dernière mise à jour.

Par défaut il ne supprime AUCUNE donnée : il affiche l'espace récupérable et la
commande pour le libérer.

    maintenance.bat                 # vérifie et nettoie le code seulement
    maintenance.bat --purge         # + supprime les données des fonctions
                                    #   retirées et les orphelins
    (./maintenance.sh sur Linux/Mac)

Ne touche jamais à models/custom/, loras/, outputs/, userdata/, python/, bin/.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
#  FONCTIONS RETIRÉES DU PROJET
#
#  Une mise à jour par copier-coller ajoute et écrase, mais n'efface JAMAIS. Une
#  fonction retirée laisse donc deux traces bien différentes :
#    · son CODE, qui nous appartient -> on le supprime sans rien demander ;
#    · ses DONNÉES (poids téléchargés, dépôts clonés), qui pèsent parfois des
#      gigaoctets -> on les CHIFFRE et on les signale, mais on ne supprime
#      qu'avec « --purge », parce que l'utilisateur peut vouloir les récupérer
#      ailleurs avant.
#
#  Ajouter une entrée ici est la SEULE chose à faire quand on retire une
#  fonction : le nettoyage, le calcul de taille et le message suivent.
#
#  ⚠️ UN NOM DE FICHIER PEUT ÊTRE REPRIS. C'est arrivé : `atelier/engine/
#  sdserver.py` était listé ici (ancien backend serveur, retiré) et un nouveau
#  module du même nom est arrivé des mois plus tard. La maintenance l'effaçait
#  à chaque passage, et l'application ne démarrait plus. Une table de noms ne
#  peut pas savoir ça — c'est pourquoi `clean_removed_features` demande
#  maintenant au CODE ACTUEL s'il utilise le fichier avant de le supprimer.
# --------------------------------------------------------------------------- #
REMOVED_FEATURES = [
    {"name": "Onglet Upscale (ancienne version)",
     "files": ["atelier/ui/creative_tab.py",
               "scripts/tools/run_creative_upscale.py"],
     "dirs": []},
    {"name": "Module Midjourney",
     "files": ["atelier/mjparams.py"], "dirs": []},
    {"name": "Backend ComfyUI",
     "files": ["atelier/engine/comfyui.py",
               "scripts/get_comfyui.py",
               "config/comfyui_workflows/flux2.json",
               "config/comfyui_workflows/krea2.json",
               "config/comfyui_workflows/krea2int8.json",
               "config/comfyui_workflows/krea2convrot.json"],
     "dirs": ["config/comfyui_workflows", "comfyui"]},
    {"name": "Build maison du moteur (CI du projet)",
     "files": ["update-engine-ci.bat",
               ".github/workflows/build-sdcpp.yml"],
     "dirs": []},
    {"name": "Génération vidéo (LTX-2.3, MiniMax-H3)",
     "files": ["atelier/ui/video_tab.py", "atelier/engine/video.py"],
     "dirs": []},
    # La sonde MiniMax-H3 a répondu à sa question (l'encodeur ne tient pas sur
    # une carte de 12 Go) ; elle est retirée avec le reste de MiniMax. Déclarée
    # ici pour que les copies déjà installées soient nettoyées à la maintenance.
    {"name": "Sonde MiniMax-H3",
     "files": ["scripts/try_minimax.py", "try-minimax.bat", "try-minimax.sh",
               "tests/test_try_minimax.py"],
     "dirs": []},
]

# Dossiers de données à NE JAMAIS toucher.
PROTECTED = {"python", "bin", "models", "loras", "outputs", "userdata", ".git"}

OK, WARN, ERR, INFO = "  [OK] ", "  [!] ", "  [X] ", "  [i] "
_problems = 0


def _warn(msg: str) -> None:
    global _problems
    _problems += 1
    print(WARN + msg)


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _human(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


def _still_in_service(rel: str) -> bool:
    """Ce fichier est-il utilisé par le code ACTUEL ?

    Un nom de fichier peut être repris des mois après le retrait de ce qu'il
    désignait. La table REMOVED_FEATURES ne peut pas le deviner : elle ne
    connaît que des chaînes de caractères. On demande donc au code actuel —
    et un module que quelqu'un importe est vivant, quoi qu'en dise la table.

    C'est un garde-fou, pas une devinette : en cas de doute (fichier hors
    atelier/, analyse impossible), on répond « oui, en service ». Refuser une
    suppression coûte un fichier mort de plus ; l'accepter à tort a coûté une
    application qui ne démarrait plus.
    """
    if not rel.endswith(".py"):
        return False
    path = ROOT / rel
    if not path.is_file():
        return False
    try:
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    except ValueError:
        return True
    if not module.startswith("atelier."):
        # Un script ou un runner : il n'est importé par personne par
        # construction, la table reste seule juge.
        return False
    try:
        return module in _reachable_modules()
    except Exception:  # noqa: BLE001
        return True


def clean_removed_features(purge: bool) -> int:
    """Nettoie ce que les fonctions retirées ont laissé derrière elles.

    Le CODE part sans discussion (c'est le nôtre, et le garder fait tourner de
    l'ancien code par accident) — SAUF s'il est encore utilisé, cf.
    `_still_in_service`. Les DONNÉES sont d'abord CHIFFRÉES et signalées :
    supprimer plusieurs gigaoctets de poids sans prévenir n'est pas à nous de
    le décider. Renvoie l'espace récupérable restant, en octets.
    """
    print("• Fonctions retirées (code + données laissées derrière)…")
    touched = False
    recoverable = 0
    for feat in REMOVED_FEATURES:
        gone: list[str] = []
        for rel in feat["files"]:
            f = ROOT / rel
            if f.exists():
                if _still_in_service(rel):
                    _warn(f"{rel} est listé comme retiré mais le code actuel "
                          "l'utilise : NON supprimé. Le nom a été repris — "
                          "retirez-le de REMOVED_FEATURES.")
                    continue
                try:
                    f.unlink()
                    gone.append(rel)
                except OSError as exc:
                    _warn(f"impossible de supprimer {rel} : {exc}")
        if gone:
            touched = True
            print(OK + f"{feat['name']} : {len(gone)} fichier(s) de code "
                  "supprimé(s).")
        for rel in feat["dirs"]:
            d = ROOT / rel
            if not d.is_dir():
                continue
            size = _dir_size(d)
            if size == 0:
                # Dossier vide : aucune donnée en jeu, on peut l'enlever.
                try:
                    shutil.rmtree(d)
                    print(OK + f"{feat['name']} : dossier vide {rel}/ supprimé.")
                    touched = True
                except OSError:
                    pass
                continue
            if purge:
                shutil.rmtree(d, ignore_errors=True)
                if d.exists():
                    _warn(f"suppression partielle : {rel}/")
                else:
                    print(OK + f"{feat['name']} : {rel}/ supprimé "
                          f"({_human(size)} libérés).")
                    touched = True
            else:
                recoverable += size
                print(INFO + f"{feat['name']} : {rel}/ occupe encore "
                      f"{_human(size)}.")
    if not touched and recoverable == 0:
        print(OK + "rien à nettoyer (propre).")
    return recoverable


def _known_addon_dirs() -> set[str]:
    """Add-ons LÉGITIMES, déduits du code plutôt que recopiés à la main.

    La liste était recopiée à la main et avait déjà pris du retard : `clip` et
    `describe` — le modèle image → prompt, 7,5 Go — étaient signalés comme
    orphelins, donc proposés à la suppression par `--purge`. On énumère
    maintenant les chemins déclarés par tools.py lui-même : ajouter un add-on
    suffit, en retirer un le rend automatiquement orphelin, et il n'y a plus de
    seconde liste à tenir à jour."""
    from atelier.engine import tools
    dirs = [value for name, value in vars(tools).items()
            if name.endswith("_DIR") and isinstance(value, Path)
            and value != tools.TOOLS_DIR]
    out = set()
    for d in dirs:
        try:
            out.add(d.relative_to(tools.TOOLS_DIR).parts[0])
        except ValueError:
            pass
    return out


def report_orphan_addons(purge: bool) -> int:
    """Dossiers de tools_repo/ ne correspondant à aucun add-on du code actuel."""
    print("• Add-ons orphelins (tools_repo/)…")
    try:
        from atelier.engine import tools
        base, known = tools.TOOLS_DIR, _known_addon_dirs()
    except Exception as exc:  # noqa: BLE001
        _warn(f"analyse impossible : {exc}")
        return 0
    if not base.is_dir():
        print(OK + "aucun add-on installé.")
        return 0
    # Les dossiers déjà nommés dans REMOVED_FEATURES sont traités plus haut :
    # les recompter ici gonflerait le total d'espace récupérable.
    declared = {Path(rel).name for f in REMOVED_FEATURES for rel in f["dirs"]}
    orphans = [d for d in sorted(base.iterdir())
               if d.is_dir() and d.name not in known and d.name not in declared]
    if not orphans:
        print(OK + "aucun add-on orphelin (propre).")
        return 0
    total = 0
    for d in orphans:
        size = _dir_size(d)
        total += size
        print(f"    - {d.name}  ({_human(size)})")
    if purge:
        freed = 0
        for d in orphans:
            sz = _dir_size(d)
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                freed += sz
                print(OK + f"supprimé : {d.name}")
            else:
                _warn(f"suppression partielle : {d.name}")
        print(OK + f"{_human(freed)} libérés.")
        return 0
    print(INFO + f"{len(orphans)} add-on(s) d'une version précédente = "
          f"{_human(total)} récupérables.")
    return total


def clean_pycache() -> None:
    print("• Caches Python (__pycache__ / .pyc)…")
    n = 0
    for p in ROOT.rglob("__pycache__"):
        if p.is_dir() and not any(part in PROTECTED for part in p.parts):
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    print(OK + f"{n} dossier(s) __pycache__ purgé(s).")


def clean_tmp() -> None:
    print("• Dossier tmp/…")
    tmp = ROOT / "tmp"
    n = 0
    if tmp.is_dir():
        for p in tmp.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
                n += 1
            except OSError:
                pass
    print(OK + f"{n} élément(s) temporaire(s) effacé(s).")


def _expected_model_dirs() -> set[str]:
    """Noms de dossiers (owner__repo) attendus d'après le catalogue courant :
    tous les composants des modèles + upscalers."""
    from atelier import registry, settings
    prefs = settings.load_prefs()
    repos: set[str] = set()
    for m in registry.load_base_models(prefs):
        repos.update(c.repo for c in m.components)
    up = registry.upscaler_config().get("repo")
    if up:
        repos.add(up)
    return {settings.model_repo_dir(r).name for r in repos if r}


def report_orphan_models(prune: bool) -> int:
    print("• Modèles orphelins (dossiers plus référencés par le catalogue)…")
    try:
        from atelier import settings
        models_dir = settings.MODELS_DIR
        expected = _expected_model_dirs()
    except Exception as exc:  # noqa: BLE001
        _warn(f"analyse impossible : {exc}")
        return 0
    if not models_dir.is_dir():
        print(OK + "aucun dossier models/.")
        return 0
    orphans = [d for d in sorted(models_dir.iterdir())
               if d.is_dir() and d.name != "custom" and d.name not in expected]
    if not orphans:
        print(OK + "aucun modèle orphelin (propre).")
        return 0
    total = 0
    for d in orphans:
        size = _dir_size(d)
        total += size
        print(f"    - {d.name}  ({_human(size)})")
    if prune:
        freed = 0
        for d in orphans:
            sz = _dir_size(d)
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                freed += sz
                print(OK + f"supprimé : {d.name}")
            else:
                _warn(f"suppression partielle : {d.name}")
        print(OK + f"{_human(freed)} libérés.")
        return 0
    print(INFO + f"{len(orphans)} dossier(s) orphelin(s) = "
          f"{_human(total)} récupérables.")
    return total


def compile_check() -> None:
    print("• Compilation (syntaxe)…")
    import compileall
    ok = True
    for target in ("atelier", "scripts"):
        ok &= compileall.compile_dir(str(ROOT / target), quiet=1, force=True)
    ok &= compileall.compile_file(str(ROOT / "app.py"), quiet=1, force=True)
    if ok:
        print(OK + "tout le code Python compile.")
    else:
        global _problems
        _problems += 1
        print(ERR + "erreur(s) de syntaxe ci-dessus — mise à jour incomplète ?")


def check_catalog() -> None:
    print("• Catalogue de modèles (config/models.yaml)…")
    try:
        import yaml
        cat = yaml.safe_load((ROOT / "config" / "models.yaml")
                             .read_text(encoding="utf-8")) or {}
        models = [m.get("id") for m in cat.get("base_models", [])]
        print(OK + f"YAML valide — modèles : {', '.join(models) or '(aucun)'}.")
    except Exception as exc:  # noqa: BLE001
        _warn(f"models.yaml illisible : {exc}")


def check_deps() -> None:
    print("• Dépendances Python…")
    missing = []
    for mod in ("gradio", "yaml", "PIL", "requests", "huggingface_hub"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    if missing:
        _warn(f"manquantes : {', '.join(missing)} → relancez install.bat "
              "(ou install.sh).")
    else:
        print(OK + "présentes.")
    check_gradio_major()
    check_diffusers()


# Version de Gradio sous laquelle l'application est écrite. Une 5.x installée
# ne « manque » pas — elle est là, elle s'importe, et elle plante à la
# construction de la première image sur un argument inconnu. C'est le genre de
# panne qu'on veut voir NOMMÉE ici plutôt qu'à travers un TypeError.
GRADIO_MAJOR = 6


def check_gradio_major() -> None:
    print("• Version de Gradio…")
    try:
        import gradio
    except Exception as exc:  # noqa: BLE001
        _warn(f"gradio introuvable : {exc}")
        return
    version = getattr(gradio, "__version__", "0")
    try:
        major = int(str(version).split(".")[0])
    except ValueError:
        _warn(f"version illisible : {version}")
        return
    if major < GRADIO_MAJOR:
        _warn(f"gradio {version} installé — l'application demande la "
              f"{GRADIO_MAJOR}.x.")
        _warn("  Les composants image refuseront `buttons=` et l'interface "
              "ne se construira pas.")
        _warn("  Correctif : relancez install.bat (ou "
              "`pip install -U -r requirements.txt`).")
    elif major > GRADIO_MAJOR:
        _warn(f"gradio {version} installé, l'application est écrite pour la "
              f"{GRADIO_MAJOR}.x — à vérifier.")
    else:
        print(OK + f"gradio {version}.")


def check_diffusers() -> None:
    """Cohérence des paquets PARTAGÉS par les add-ons PyTorch.

    Tous vivent dans le même Python : un add-on installé avec une contrainte
    plus large écrase la version dont un autre a besoin, et la casse ne se voit
    qu'au premier usage de l'autre — sous forme d'une erreur illisible à
    l'import. On vérifie donc chaque paquet épinglé par l'installeur.
    """
    print("• Paquets partagés par les add-ons PyTorch…")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from setup_tools import _PINS
    except Exception:  # noqa: BLE001
        print(OK + "non vérifiable (installeur absent).")
        return

    import importlib.metadata as md
    checked = bad = 0
    for name, spec in sorted(_PINS.items()):
        try:
            got = md.version(name)
        except Exception:  # noqa: BLE001
            continue                      # paquet absent = add-on non installé
        checked += 1
        if not _spec_ok(got, spec):
            bad += 1
            _warn(f"{name} {got} installé — attendu « {spec} ».")
    if not checked:
        print(OK + "aucun add-on PyTorch installé.")
    elif bad:
        _warn("  Un add-on a changé une version sous les autres.")
        _warn("  Correctif : relancez l'installation de l'add-on concerné "
              "(Toolkit → Installer), qui repose les bonnes versions.")
    else:
        print(OK + f"{checked} paquet(s) conforme(s).")


def _spec_ok(version: str, spec: str) -> bool:
    """Version conforme à un spec pip simple (« ==x », « >=a,<b »).

    Comparaison numérique par composants : « 1.26.4 » < « 2 » proprement, sans
    dépendre de packaging (absent du Python embarqué minimal).
    """
    def key(v: str):
        out = []
        for part in v.split("."):
            num = "".join(c for c in part if c.isdigit())
            out.append(int(num) if num else 0)
        return tuple(out)

    if "==" in spec:
        return version == spec.split("==")[-1].strip()
    for clause in spec.split(","):
        clause = clause.strip()
        for op in (">=", "<=", "!=", "<", ">"):
            if op in clause:
                bound = clause.split(op, 1)[1].strip()
                # retire un eventuel prefixe de nom de paquet
                bound = bound.split()[0] if bound else bound
                a, b = key(version), key(bound)
                a, b = a + (0,) * (len(b) - len(a)), b + (0,) * (len(a) - len(b))
                ok = {">=": a >= b, "<=": a <= b, "<": a < b,
                      ">": a > b, "!=": a != b}[op]
                if not ok:
                    return False
                break
    return True


# --------------------------------------------------------------------------- #
#  CAPACITÉS ATTENDUES DU MOTEUR
#
#  Le dépôt et le binaire sd-cli se mettent à jour SÉPARÉMENT : copier le code
#  par-dessus l'ancien ne touche pas à bin/. Une fonction de l'application peut
#  donc réclamer une option que le moteur installé ne connaît pas encore, et
#  l'utilisateur ne le découvre qu'au moment où ça casse.
#
#  On liste donc ici ce dont le code a réellement besoin. Chaque entrée dit
#  quelle FONCTION dépend de quelle option : un « --hires manquant » ne parle à
#  personne, « l'onglet HD ne marchera pas » si.
# --------------------------------------------------------------------------- #
ENGINE_FEATURES = [
    {"option": "--hires", "needed_by": "l'onglet « 🚀 HD »",
     "blocking": True},
    {"option": "--upscale-tile-size",
     "needed_by": "l'agrandissement ESRGAN sans coutures", "blocking": False},
    {"option": "--max-vram",
     "needed_by": "l'exécution segmentée (HD sur carte serrée)",
     "blocking": False},
    {"option": "--diffusion-conv-direct",
     "needed_by": "la convolution directe (Réglages)", "blocking": False},
    {"option": "--preview", "needed_by": "l'aperçu temps réel",
     "blocking": False},
]


def check_engine(update: bool) -> bool:
    """Présence ET capacités du moteur. Renvoie True si une MAJ est conseillée."""
    print("• Moteur stable-diffusion.cpp (sd-cli)…")
    try:
        from atelier import settings
        from atelier.engine import sdcpp
        sd = settings.find_sd_cli()
    except Exception as exc:  # noqa: BLE001
        _warn(f"vérification impossible : {exc}")
        return False

    if sd is None:
        # Pas de « --variant cuda » en dur : sur Mac ce serait un mauvais
        # conseil (il n'existe que des builds Metal). get_sdcpp déduit seul.
        if update:
            print(INFO + "binaire absent → installation…")
            return not _run_get_sdcpp()
        _warn("binaire sd-cli introuvable → maintenance.bat --update-engine "
              "(ou install.bat)")
        return True
    print(OK + f"trouvé : {sd}")

    opts = sdcpp.supported_options(sd)
    if not opts:
        _warn("le binaire ne répond pas à « -h » : impossible de vérifier ses "
              "capacités. S'il ne démarre pas non plus, réinstallez-le "
              "(maintenance.bat --update-engine).")
        return False

    missing = [f for f in ENGINE_FEATURES if f["option"] not in opts]
    if not missing:
        print(OK + f"{len(ENGINE_FEATURES)} capacité(s) attendue(s) présente(s).")
        return False
    for f in missing:
        line = f"{f['option']} absent → {f['needed_by']} ne fonctionnera pas."
        if f["blocking"]:
            _warn(line)
        else:
            print(INFO + line)
    if update:
        print(INFO + "mise à jour du moteur…")
        return not _run_get_sdcpp(force=True)
    print(INFO + "Moteur d'une version antérieure au code. Pour l'aligner :")
    print("        maintenance.bat --update-engine"
          "   (./maintenance.sh --update-engine)")
    return True


def _run_get_sdcpp(force: bool = False) -> bool:
    """Lance scripts/get_sdcpp.py dans CE Python. True si ça a réussi.

    Sous-process plutôt qu'import : le script est fait pour être un programme
    (il appelle sys.exit), et un échec de téléchargement ne doit pas emporter
    la maintenance avec lui.
    """
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts" / "get_sdcpp.py")]
    if force:
        cmd.append("--force")
    print(INFO + "$ " + " ".join(cmd))
    try:
        code = subprocess.call(cmd, cwd=str(ROOT))
    except OSError as exc:
        _warn(f"lancement impossible : {exc}")
        return False
    if code == 0:
        # Le cache d'options est indexé sur (chemin, mtime, taille) : un
        # nouveau binaire produit une clé différente, la relecture est donc
        # automatique. On revérifie pour AFFICHER le résultat, pas pour purger.
        print(OK + "moteur installé/mis à jour.")
        return True
    _warn(f"la mise à jour du moteur a échoué (code {code}). Réseau ? "
          "Réessayez, ou lancez update-engine.bat.")
    return False


def _module_map() -> dict[str, "Path"]:
    """Nom de module -> fichier, pour tout atelier/."""
    def module_of(path: Path) -> str:
        rel = path.relative_to(ROOT).with_suffix("")
        parts = [x for x in rel.parts if x != "__init__"]
        return ".".join(parts)

    return {module_of(x): x for x in (ROOT / "atelier").rglob("*.py")}


def _imports_of(path: "Path") -> set[str]:
    """Modules cités par les imports de ce fichier (relatifs résolus)."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    rel = path.relative_to(ROOT).with_suffix("")
    parts = [x for x in rel.parts if x != "__init__"]
    module = ".".join(parts)
    # Le paquet CONTENANT le fichier — et un __init__.py est contenu par son
    # propre paquet, pas par celui du dessus. Sans cette distinction,
    # « from . import sdserver » écrit dans atelier/engine/__init__.py
    # résolvait vers « atelier.sdserver », qui n'existe pas : le module
    # importé passait pour orphelin, et le garde-fou de suppression pour
    # inutile. Exactement le module qu'on venait d'effacer par erreur.
    if path == ROOT / "app.py":
        pkg = ""
    elif path.name == "__init__.py":
        pkg = module
    else:
        pkg = module.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:                       # import relatif
                up = pkg.split(".")
                base = ".".join(up[:len(up) - node.level + 1]
                                + ([base] if base else []))
            found.add(base)
            found.update(f"{base}.{a.name}" for a in node.names)
    return found


_REACHABLE: "set[str] | None" = None


def _reachable_modules() -> set[str]:
    """Modules de atelier/ réellement atteints depuis les points d'entrée.

    Le calcul sert deux fois — signaler les orphelins, et protéger un fichier
    dont le nom a été repris — donc il est fait une seule fois.

    RGLOB sur scripts/, pas glob : les runners d'outils vivent dans
    scripts/tools/ et sont eux aussi des points d'entrée. Les oublier faisait
    passer pour orphelin tout module importé uniquement par eux — un faux
    positif qui pousse à supprimer du code vivant, soit exactement l'inverse
    du but.
    """
    global _REACHABLE
    if _REACHABLE is not None:
        return _REACHABLE
    files = _module_map()
    seen: set[str] = set()
    queue = [ROOT / "app.py"] + sorted((ROOT / "scripts").rglob("*.py"))
    while queue:
        path = queue.pop()
        for name in _imports_of(path):
            if name in seen or name not in files:
                continue
            seen.add(name)
            # Importer « atelier.ui.generate_tab » importe forcément le paquet
            # « atelier.ui » : sans ça, chaque __init__.py serait signalé
            # orphelin alors qu'il est la condition de tous ses modules.
            #
            # Et le paquet est EMPILÉ, pas seulement marqué : un __init__.py
            # contient du code, donc des imports. Le marquer « vu » sans le
            # lire faisait dépendre le résultat de l'ordre de parcours — si le
            # paquet était rencontré comme parent avant d'être rencontré comme
            # import, ses propres imports n'étaient jamais suivis. C'est ce qui
            # rendait `atelier.engine.sdserver` invisible : il n'est importé
            # que depuis `atelier/engine/__init__.py`.
            parts = name.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent not in seen:
                    seen.add(parent)
                    if parent in files:
                        queue.append(files[parent])
            queue.append(files[name])
    _REACHABLE = seen
    return seen


def check_orphan_modules() -> None:
    """Modules Python de atelier/ que plus RIEN n'importe.

    Complément générique à REMOVED_FEATURES : celle-ci ne connaît que les
    fonctions qu'on a pensé à y déclarer. Ici on part de app.py et des scripts,
    on suit les imports, et tout module de atelier/ jamais atteint est un reste
    d'une version précédente — quelle qu'elle soit, déclarée ou non.
    """
    print("• Modules Python orphelins (plus importés par personne)…")
    files = _module_map()
    seen = _reachable_modules()
    orphans = sorted(m for m in files if m not in seen and m != "atelier")
    if not orphans:
        print(OK + "aucun module orphelin (propre).")
        return
    for m in orphans:
        print(f"    - {files[m].relative_to(ROOT)}")
    _warn(f"{len(orphans)} module(s) que rien n'importe — probablement des "
          "restes d'une version précédente. Vérifiez avant de supprimer : un "
          "module chargé dynamiquement apparaîtrait ici à tort.")


USAGE = """\
Maintenance — Turboslop 5000

  maintenance.bat                   vérifie et nettoie le CODE (aucune donnée
                                    supprimée, l'espace récupérable est chiffré)
  maintenance.bat --update-engine   + aligne le moteur sd-cli sur le code
  maintenance.bat --purge           + supprime les données des fonctions
                                    retirées et les orphelins
  maintenance.bat --all             tout : purge + mise à jour du moteur
                                    (« après une MAJ, tout est nickel »)

Pour mettre à jour l'APPLICATION elle-même : update.bat (ce script ne
télécharge rien).

(./maintenance.sh … sur Linux/Mac)
Ne touche jamais à models/custom/, loras/, outputs/, userdata/, python/.
"""


def check_install_complete() -> None:
    """Des fichiers de l'application ont-ils disparu depuis la mise à jour ?

    C'est le diagnostic qui manquait le jour où un module s'est volatilisé :
    l'application ne démarrait plus, avec un ImportError qui nommait le module
    mais pas la cause. Le manifeste de `update.bat` sait exactement ce qui
    devrait être là.
    """
    print("• Intégrité de l'installation…")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from update_app import MANIFEST, missing_files
    except Exception as exc:  # noqa: BLE001
        print(INFO + f"non vérifiable ({exc}).")
        return
    if not MANIFEST.is_file():
        print(INFO + "jamais mise à jour par update.bat — rien à comparer.")
        return
    missing = missing_files()
    if not missing:
        print(OK + "tous les fichiers de la dernière mise à jour sont là.")
        return
    for rel in missing[:10]:
        print(f"    - {rel}")
    if len(missing) > 10:
        print(f"    … et {len(missing) - 10} autre(s)")
    _warn(f"{len(missing)} fichier(s) de l'application ont disparu. "
          "Relancez update.bat : il les remettra.")


def main() -> int:
    # « --purge » supprime TOUT ce qui reste des fonctions retirées : dossiers
    # d'add-ons, modèles orphelins, données laissées derrière. « --prune-models »
    # est conservé comme alias historique (il ne visait que les modèles).
    if "--help" in sys.argv or "-h" in sys.argv:
        print(USAGE)
        return 0
    everything = "--all" in sys.argv
    purge = everything or "--purge" in sys.argv
    update_engine = everything or "--update-engine" in sys.argv
    prune_models = purge or "--prune-models" in sys.argv
    print("=" * 60)
    print("  Maintenance — Turboslop 5000")
    modes = []
    if purge:
        modes.append("suppression des restes des fonctions retirées")
    if update_engine:
        modes.append("mise à jour du moteur")
    if modes:
        print("  (" + " + ".join(modes) + ")")
    print("=" * 60)
    recoverable = clean_removed_features(purge)
    clean_pycache()
    clean_tmp()
    check_catalog()
    recoverable += report_orphan_addons(purge)
    recoverable += report_orphan_models(prune_models)
    check_orphan_modules()
    check_install_complete()
    compile_check()
    check_deps()
    engine_stale = check_engine(update_engine)
    print("-" * 60)
    if recoverable > 0:
        # Un chiffre global, puis la commande exacte : c'est tout ce qu'il faut
        # pour décider, sans avoir à additionner les lignes soi-même.
        print(f"💾 {_human(recoverable)} récupérables (restes de fonctions "
              "retirées).")
        print("   Pour libérer :  maintenance.bat --purge"
              "   (./maintenance.sh --purge sur Linux/Mac)")
    if engine_stale and not update_engine:
        print("🔧 Le moteur est en retard sur le code.")
        print("   Pour tout aligner d'un coup :  maintenance.bat --all")
    if recoverable > 0 or (engine_stale and not update_engine):
        print("-" * 60)
    if _problems == 0:
        print("✅ Tout est propre et vérifié. Vous pouvez lancer run.bat.")
    else:
        print(f"⚠️  Terminé avec {_problems} point(s) d'attention "
              "ci-dessus (voir les lignes [!]/[X]).")
    print("=" * 60)
    return 0 if _problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
