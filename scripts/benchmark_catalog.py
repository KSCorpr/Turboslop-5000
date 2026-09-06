#!/usr/bin/env python3
"""Measure catalogue parsing overhead; this is not an image-generation benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atelier import registry, settings
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    path = settings.CONFIG_DIR / "models.yaml"

    def original():
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    assert original() == registry._catalog()
    results = {}
    for name, fn in [("original_yaml_each_call", original),
                     ("cached_yaml_with_stat_and_deepcopy", registry._catalog)]:
        results[name] = statistics.median(timeit.repeat(fn, number=args.iterations, repeat=3))
    print(json.dumps({"scope": "catalogue reads only; not GPU generation",
                      "iterations_per_repeat": args.iterations, "repeats": 3,
                      "median_seconds": results,
                      "ratio": results["original_yaml_each_call"] /
                               results["cached_yaml_with_stat_and_deepcopy"]}, indent=2))


if __name__ == "__main__":
    main()
