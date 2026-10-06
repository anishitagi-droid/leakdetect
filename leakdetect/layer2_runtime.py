"""Layer 2 - runtime provenance tracker.

How it works
  * Rows are identified by a content hash. `track_provenance` wraps fit / fit_transform / transform; inside a
    `with ProvenanceTracker()` block, fit records the rows the estimator saw and transform records lineage
    (output-row hash -> input-row hash), so scaled/encoded rows trace back to the raw rows that were fit on.
  * `tracked_train_test_split` / `register_split` label every row train/val/test via its lineage root.
  * A fit is CONTAMINATED if its fit set contains rows labelled val/test (checked at fit time if labels exist,
    and retroactively whenever a split is registered).
  * `find_cross_split_duplicates` detects exact and near duplicates across train/test.

Limitations (surfaced, not hidden)
  * Row identity is content-based: a transform that changes the row count drops lineage (noted); exact duplicate
    rows are ambiguous by construction and are reported separately, not counted as contamination.
  * Only instrumented estimators are seen; clone() creates uninstrumented copies of an instrumented INSTANCE.
    Hand-written pandas statistics are invisible here (Layer 1 catches some statically).
  * Outside a `with ProvenanceTracker()` block, instrumented estimators behave exactly like the originals.
  * A split must be registered; otherwise the layer reports "skipped", not "clean".
"""

from __future__ import annotations

import contextvars
import functools
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split as _sk_split

from .categories import SEVERITY, category_of
from .schema import Finding, LayerResult

LAYER = 2
MAX_LINEAGE_DEPTH = 64  # cycle guard when following output-row -> input-row links
BRUTE_MAX_PAIRS = 2_250_000  # above this many train x test pairs, near-duplicate search switches to LSH
LOW_DIM_COLS = 8  # at or below this width, "identical rows" are weak evidence of repeated entities
HELD_OUT = {"test", "val"}
_TRACKED_METHODS = ("fit", "fit_transform", "transform")
_current = contextvars.ContextVar("leakdetect_tracker", default=None)


def _as_frame(X):
    if hasattr(X, "toarray"):  # scipy sparse
        X = X.toarray()
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X).reshape(len(X), -1))


def row_hashes(X) -> np.ndarray:
    return pd.util.hash_pandas_object(_as_frame(X), index=False).to_numpy()


@dataclass
class FitEvent:
    name: str
    cls: str
    supervised: bool
    n_rows: int
    roots: frozenset[int]  # lineage roots of the rows the estimator was fit on
    flagged: bool = False


class ProvenanceTracker:
    """Context manager recording fits, row lineage and split membership for instrumented estimators."""

    def __init__(self):
        self.labels: dict[int, set[str]] = {}  # lineage root -> {"train", "test", ...}
        self.source: dict[int, int] = {}  # output-row hash -> input-row hash
        self.events: list[FitEvent] = []
        self.findings: list[Finding] = []
        self.notes: list[str] = []
        self.splits_registered = 0
        self._busy = False  # re-entrancy guard: fit_transform calls fit, which must not be recorded twice
        self._token = None

    def __enter__(self):
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc):
        _current.reset(self._token)

    def root(self, h):
        for _ in range(MAX_LINEAGE_DEPTH):
            nxt = self.source.get(h, h)
            if nxt == h:
                break
            h = nxt
        return h

    def call(self, fn, est, args, kwargs, method):
        """Run `fn(est, ...)` and record what it saw."""
        if self._busy:
            return fn(est, *args, **kwargs)
        self._busy = True
        try:
            X = args[0] if args else kwargs.get("X")
            if X is not None and method != "transform":
                self._on_fit(est, X, args[1] if len(args) > 1 else kwargs.get("y"))
            out = fn(est, *args, **kwargs)
            if X is not None and method != "fit":
                self._on_transform(est, X, out)
            return out
        finally:
            self._busy = False

    def _on_fit(self, est, X, y):
        hashes = row_hashes(X).tolist()
        cls = type(est).__name__
        event = FitEvent(
            f"{cls}#{len(self.events)}", cls, y is not None, len(hashes), frozenset(self.root(h) for h in hashes)
        )
        self.events.append(event)
        if self.splits_registered:
            self._check_event(event)

    def _on_transform(self, est, X, out):
        try:
            hin, hout = row_hashes(X).tolist(), row_hashes(out).tolist()
        except (TypeError, ValueError) as e:  # exotic output types: never break the user's code
            self.notes.append(f"lineage dropped for {type(est).__name__}: {e}")
            return
        if len(hin) != len(hout):
            self.notes.append(
                f"lineage dropped for {type(est).__name__}.transform: row count changed {len(hin)} -> {len(hout)}"
            )
            return
        self.source.update({b: a for a, b in zip(hin, hout, strict=True) if a != b})

    def register_split(self, **parts):
        """register_split(train=X_tr, test=X_te, val=X_va): label rows via lineage roots, then re-check all fits."""
        for label, arr in parts.items():
            if arr is not None:
                for h in row_hashes(arr).tolist():
                    self.labels.setdefault(self.root(h), set()).add(label)
        self.splits_registered += 1
        for event in self.events:
            self._check_event(event)

    def _check_event(self, ev: FitEvent):
        if ev.flagged:
            return
        held, ambiguous = set(), 0
        for r in ev.roots:
            lab = self.labels.get(r)
            if not lab:
                continue
            if lab <= HELD_OUT:
                held.add(r)
            elif "train" in lab and len(lab) > 1:  # identical row present in train AND held-out
                ambiguous += 1
        if not held:
            return
        ev.flagged = True
        cat = category_of(ev.cls) or "preproc"
        which = "/".join(sorted({lab for r in held for lab in self.labels[r]}))
        msg = (
            f"{ev.cls} was fit on {ev.n_rows} rows; {len(held)} distinct rows ({len(held) / len(ev.roots):.0%}) "
            f"were later assigned to {which}. Held-out data influenced the fitted state."
        )
        if ambiguous:
            msg += (
                f" ({ambiguous} additional fit rows are exact duplicates present in both train and test; not counted.)"
            )
        if ev.supervised:
            msg += " Supervised fit: the target was available."
        self.findings.append(
            Finding(
                LAYER,
                ev.name,
                "fit-contamination",
                SEVERITY[cat] + (ev.supervised and cat == "preproc"),
                msg,
                "Fit on the training rows only (split first), or fit inside a Pipeline within each fold.",
            )
        )

    def result(self) -> LayerResult:
        if not self.splits_registered:
            return LayerResult.skipped(
                LAYER,
                "no split registered (use tracked_train_test_split or "
                "register_split); contamination cannot be determined.",
            )
        notes = [*self.notes, f"{len(self.events)} fit event(s) tracked, {self.splits_registered} split(s) registered."]
        return LayerResult(LAYER, "ok", findings=list(self.findings), notes=notes)


# ---- instrumentation
def _wrap(fn, method):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        tracker = _current.get()
        if tracker is None:  # outside a `with ProvenanceTracker()` block: behave as the original
            return fn(self, *args, **kwargs)
        return tracker.call(fn, self, args, kwargs, method)

    wrapper._leakdetect_wrapped = True
    return wrapper


def track_provenance(obj):
    """Instrument an estimator CLASS (use as a decorator; patches the class permanently) or an INSTANCE (patches
    only that object). Records only inside a `with ProvenanceTracker()` block. Returns `obj`."""
    for name in _TRACKED_METHODS:
        fn = getattr(obj if isinstance(obj, type) else type(obj), name, None)
        if fn is None or getattr(fn, "_leakdetect_wrapped", False):
            continue
        wrapped = _wrap(fn, name)
        setattr(obj, name, wrapped if isinstance(obj, type) else types.MethodType(wrapped, obj))
    return obj


def tracked_train_test_split(*arrays, near_dup_eps=None, **kwargs):
    """Drop-in for sklearn's train_test_split that registers row membership from the FIRST array.
    near_dup_eps: also run near-duplicate detection at that tolerance (see find_cross_split_duplicates)."""
    tracker = _current.get()
    if tracker is None:
        raise RuntimeError("tracked_train_test_split needs an active tracker: use `with ProvenanceTracker():`")
    out = _sk_split(*arrays, **kwargs)
    train, test = out[0], out[1]
    tracker.register_split(train=train, test=test)
    tracker.findings += find_cross_split_duplicates(train, test)
    if near_dup_eps:
        tracker.findings += find_cross_split_duplicates(train, test, mode="near", eps=near_dup_eps)
    return out


# ---- duplicate detection
def _numeric_matrix(A, B):
    dfA, dfB = _as_frame(A), _as_frame(B)
    both = pd.get_dummies(pd.concat([dfA, dfB], ignore_index=True), drop_first=True, dtype=float).astype(float)
    sd = both.std(ddof=0).replace(0, np.nan)
    M = ((both - both.mean()) / sd).dropna(axis=1, how="all").fillna(0.0).to_numpy()
    return M[: len(dfA)], M[len(dfA) :]


def _near_pairs_bruteforce(A, B, eps):
    dist = cdist(A, B)
    i, j = np.nonzero(dist <= eps * np.sqrt(A.shape[1]))
    return list(zip(i.tolist(), j.tolist(), dist[i, j].tolist(), strict=True))


def _near_pairs_lsh(A, B, eps, k=4, n_tables=12, seed=0):
    """p-stable (Euclidean) LSH; candidates are verified by true distance, so there are no false positives."""
    d = A.shape[1]
    r = eps * np.sqrt(d)
    w = 4.0 * r
    rng = np.random.default_rng(seed)
    cand = set()
    for _ in range(n_tables):
        proj, offset = rng.normal(size=(d, k)), rng.uniform(0, w, size=k)
        ha, hb = (np.floor((M @ proj + offset) / w).astype(np.int64) for M in (A, B))
        buckets = {}
        for j, row in enumerate(hb):
            buckets.setdefault(row.tobytes(), []).append(j)
        cand.update((i, j) for i, row in enumerate(ha) for j in buckets.get(row.tobytes(), ()))
    if not cand:
        return []
    ij = np.array(sorted(cand))
    dist = np.linalg.norm(A[ij[:, 0]] - B[ij[:, 1]], axis=1)
    keep = dist <= r
    return list(zip(ij[keep, 0].tolist(), ij[keep, 1].tolist(), dist[keep].tolist(), strict=True))


def _low_dim_caveat(ncols):
    if ncols > LOW_DIM_COLS:
        return ""
    return (
        f" NOTE: compared on only {ncols} column(s); in low-dimensional or discrete representations "
        f"different records can look identical, so confirm on the raw, full-width data before "
        f"concluding there are repeated entities."
    )


def _exact_duplicates(X_train, X_test, ncols):
    test_hashes = row_hashes(X_test).tolist()
    common = set(row_hashes(X_train).tolist()) & set(test_hashes)
    if not common:
        return []
    n_dup = sum(h in common for h in test_hashes)
    share = n_dup / len(test_hashes)
    return [
        Finding(
            LAYER,
            "train/test rows",
            "exact-duplicate-across-splits",
            5 if share > 0.05 else 4,
            f"{len(common)} distinct row(s) appear identically in both train and test; "
            f"{n_dup}/{len(test_hashes)} test rows ({share:.1%}) have an exact copy in train "
            f"(compared on {ncols} column(s))." + _low_dim_caveat(ncols),
            "Deduplicate before splitting, or use group-aware splitting (GroupShuffleSplit).",
        )
    ]


def _near_duplicates(X_train, X_test, eps, ncols):
    A, B = _numeric_matrix(X_train, X_test)
    find = _near_pairs_bruteforce if len(A) * len(B) <= BRUTE_MAX_PAIRS else _near_pairs_lsh
    pairs = find(A, B, eps)
    if not pairs:
        return []
    test_rows = {j for _, j, _ in pairs}
    share = len(test_rows) / len(B)
    return [
        Finding(
            LAYER,
            "train/test rows",
            "near-duplicate-across-splits",
            4 if share > 0.02 else 3,
            f"{len(pairs)} train/test pair(s) lie within {eps} RMS standardised units; "
            f"{len(test_rows)}/{len(B)} test rows ({share:.1%}) have a near-copy in train. "
            f"Near-duplicates can be legitimate similar applicants - verify before concluding leakage."
            + _low_dim_caveat(ncols),
            "Inspect the pairs; if they are the same entity, split by entity group.",
        )
    ]


def find_cross_split_duplicates(X_train, X_test, mode="exact", eps=0.05):
    """mode='exact': identical rows. mode='near': rows within `eps` RMS standardised units per feature
    (brute force when cheap, otherwise LSH)."""
    if mode not in ("exact", "near"):
        raise ValueError("mode must be 'exact' or 'near'")
    ncols = _as_frame(X_train).shape[1]
    if mode == "exact":
        return _exact_duplicates(X_train, X_test, ncols)
    return _near_duplicates(X_train, X_test, eps, ncols)
