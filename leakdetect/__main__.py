"""Command line: run the dataset-level checks (Layer 4) on a CSV, and optionally the static check (Layer 1) on a
modelling script.

    python -m leakdetect data.csv --target card --positive yes --drop rownames
    python -m leakdetect data.csv --target y --positive 1 --script train.py --json

Layers 2 (runtime provenance) and 3 (feature-store point-in-time) need live Python objects / a feature store, so
they are used through the Python API (see demo/). Exit status is 1 when a finding reaches --fail-severity, so the
command can gate a CI job.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .aggregator import aggregate, to_json, to_markdown
from .layer1_static import analyze_file
from .layer4_stats import run_layer4


def _same(value, raw):
    """Does the target value correspond to the command-line text? Numbers compare numerically (so `1` matches 1.0),
    booleans case-insensitively, everything else as text."""
    if isinstance(value, (bool, np.bool_)):
        return str(value).lower() == raw.lower()
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return float(raw) == float(value)
        except ValueError:
            return False
    return str(value) == raw


def _resolve_positive(df, target, raw):
    values = df[target].dropna().unique()
    match = next((v for v in values if _same(v, raw)), None)
    if match is None:
        sys.exit(f"error: --positive {raw!r} is not a value of '{target}' (values: {sorted(map(str, values))[:10]})")
    return match


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="leakdetect", description=__doc__.split("\n\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--target", required=True, help="name of the target column")
    ap.add_argument("--positive", required=True, help="the target value treated as the positive class")
    ap.add_argument("--drop", nargs="*", default=[], metavar="COL", help="columns to ignore (row ids, ...)")
    ap.add_argument("--script", help="python file with the modelling pipeline, checked statically (Layer 1)")
    ap.add_argument("--json", action="store_true", help="print the JSON report instead of markdown")
    ap.add_argument("--fail-severity", type=int, default=4, help="exit 1 at/above this severity; 0 never fails")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    for col in (args.target, *args.drop):
        if col not in df.columns:
            sys.exit(f"error: column {col!r} not in {args.csv} (columns: {list(df.columns)[:12]}...)")
    results = [run_layer4(df, args.target, _resolve_positive(df, args.target, args.positive), drop_cols=args.drop)]
    if args.script:
        results.append(analyze_file(args.script))
    report = aggregate(results)
    print(to_json(report) if args.json else to_markdown(report))
    worst = report["summary"]["max_severity"]
    return int(args.fail_severity > 0 and worst >= args.fail_severity)


if __name__ == "__main__":
    sys.exit(main())
