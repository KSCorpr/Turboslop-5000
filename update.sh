#!/usr/bin/env bash
# Met à jour l'application (le code) depuis GitHub. Voir update.bat.
cd "$(dirname "$0")" || exit 1
PY="./python/bin/python3"
[ -x "$PY" ] || PY="python3"
exec "$PY" scripts/update_app.py "$@"
