#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY="python/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi
"$PY" scripts/get_sdcpp.py --rollback
