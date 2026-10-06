"""Shared finding / result schema. Every layer returns a LayerResult so that
"ran and found nothing" is distinguishable from "could not run"."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    layer: int
    feature_or_step: str
    leakage_type: str
    severity: int  # 1 (informational) .. 5 (almost certainly invalidates results); clamped on construction
    evidence: str
    suggested_fix: str

    def __post_init__(self):
        self.severity = max(1, min(5, int(self.severity)))

    def to_dict(self):
        return asdict(self)


@dataclass
class LayerResult:
    layer: int
    status: str  # "ok" | "skipped"
    reason: str = ""
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # skipped sub-checks, caveats
    extras: dict = field(default_factory=dict)  # supporting tables for callers; not part of the report

    @classmethod
    def skipped(cls, layer: int, reason: str) -> LayerResult:
        return cls(layer, "skipped", reason)
