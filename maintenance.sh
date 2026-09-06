#!/usr/bin/env bash
# Nettoyage et verification apres mise a jour par copier-coller.
#   ./maintenance.sh                  verifie et nettoie le code
#   ./maintenance.sh --update-engine  + aligne le moteur sd-cli sur le code
#   ./maintenance.sh --purge          + supprime les donnees des fonctions retirees
#   ./maintenance.sh --all            tout d'un coup
set -e
cd "$(dirname "$0")"
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "============================================================"
    echo "  Maintenance - Turboslop 5000"
    echo "------------------------------------------------------------"
    echo "  Les DONNEES (poids, add-ons d'anciennes versions) sont"
    echo "  seulement CHIFFREES ici, pas supprimees."
    echo
    echo "  Options :"
    echo "      ./maintenance.sh --update-engine   aligne le moteur"
    echo "      ./maintenance.sh --purge           efface les restes"
    echo "      ./maintenance.sh --all             les deux"
    echo "============================================================"
    echo
fi

python scripts/maintenance.py "$@"
