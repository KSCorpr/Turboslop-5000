#!/usr/bin/env bash
# Lance Atelier en mode réseau local (accessible aux autres machines du LAN).
# Pour un mot de passe : ./run-lan.sh --auth nom:motdepasse
set -e
cd "$(dirname "$0")"
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi
python app.py --listen "$@"
