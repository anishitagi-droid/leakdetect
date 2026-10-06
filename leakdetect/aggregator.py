"""Aggregator: normalise, de-duplicate, sort by severity, emit JSON + markdown.

Skipped layers are listed explicitly in the report so "nothing found" is never confused with "did not run".
"""

from __future__ import annotations
import json
import re
from dataclasses import replace
from .schema import Finding

FAMILY = {
    "fit-before-split": "pipeline-fit",
    "fit-contamination": "pipeline-fit",
    "fit-on-test-data": "pipeline-fit",
    "cv-preprocessing-leakage": "cv",
    "cv-loop-leakage": "cv",
    "feature-selection-before-split": "selection",
    "target-encoding-before-split": "target-encoding",
    "oversampling-before-split": "oversampling",
    "exact-duplicate-across-splits": "duplicates",
    "near-duplicate-across-splits": "duplicates",
    "single-feature-predictive-power": "target-proxy",
    "high-mutual-information": "target-proxy",
    "perfect-separation-rule": "target-proxy",
    "pit-violation": "pit",
    "stale-join": "pit",
    "unmatched-value": "pit",
    "missing-joined-value": "pit",
    "labels-missing-from-training-set": "pit",
    "backfill-leakage": "backfill",
    "online-offline-skew": "skew",
    "train-test-too-similar": "split-similarity",
    "train-test-shift": "shift",
    "post-period-feature": "post-period",
}


def _norm(finding: Finding) -> str:
    # Layer 1 labels steps "StandardScaler@L12", Layer 2 "StandardScaler#0": compare on the class name.
    return re.split(r"[@#( ]", finding.feature_or_step.lower())[0]


def _merge(findings):
    """Group by (family, estimator/feature); keep max severity and concatenate evidence across layers."""
    groups = {}
    for f in findings:
        key = (FAMILY.get(f.leakage_type, f.leakage_type), _norm(f))
        if key not in groups:
            groups[key] = (replace(f), {f.layer}, [f.leakage_type])
            continue
        base, layers, types = groups[key]
        layers.add(f.layer)
        if f.leakage_type not in types:
            types.append(f.leakage_type)
        base.severity = max(base.severity, f.severity)
        base.evidence += f" | [layer {f.layer}, {f.leakage_type}] {f.evidence}"
    merged = []
    for base, layers, types in groups.values():
        base.leakage_type = " + ".join(types)
        merged.append({**base.to_dict(), "flagged_by_layers": sorted(layers)})
    return sorted(merged, key=lambda d: (-d["severity"], d["feature_or_step"]))


def aggregate(results):
    raw = [f for r in results if r.status == "ok" for f in r.findings]
    findings = _merge(raw)
    return {
        "summary": {
            "findings": len(findings),
            "max_severity": max((f["severity"] for f in findings), default=0),
            "layers_run": [r.layer for r in results if r.status == "ok"],
            "layers_skipped": {r.layer: r.reason for r in results if r.status == "skipped"},
        },
        "findings": findings,
        "notes": {r.layer: r.notes for r in results if r.notes},
        "raw_finding_count": len(raw),
    }


def to_markdown(report) -> str:
    s = report["summary"]
    L = [
        "# Leakage report",
        "",
        f"**{s['findings']} finding(s)** after de-duplication ({report['raw_finding_count']} raw), "
        f"max severity **{s['max_severity']}/5**. Layers run: {s['layers_run']}.",
        "",
    ]
    if s["layers_skipped"]:
        L.append("**Layers NOT run (absence of findings from these means nothing):**")
        L += [f"- Layer {k}: {v}" for k, v in s["layers_skipped"].items()]
        L.append("")
    for f in report["findings"]:
        L += [
            f"## [{f['severity']}/5] {f['feature_or_step']} — {f['leakage_type']}",
            f"*layers: {f['flagged_by_layers']}*",
            "",
            f"**Evidence:** {f['evidence']}",
            "",
            f"**Fix:** {f['suggested_fix']}",
            "",
        ]
    if report["notes"]:
        L.append("## Caveats")
        for k, v in report["notes"].items():
            L += [f"- Layer {k}: {n}" for n in v]
    return "\n".join(L)


def to_json(report) -> str:
    return json.dumps(report, indent=2)
