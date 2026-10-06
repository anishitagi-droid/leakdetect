"""Layer 3 - point-in-time (PIT) correctness engine for feature stores.

Generic, vendor-neutral interface (long format DataFrames):
  FeatureRecord : entity_id, feature_name, value, computed_at  [, valid_from, valid_to, max_source_timestamp]
  LabelRecord   : entity_id, event_timestamp, label
  training_set  : entity_id, event_timestamp, <one column per feature>   (what was ACTUALLY joined)

Checks
  * check_point_in_time    - recompute the correct as-of value (latest computed_at STRICTLY before
                             event_timestamp, within freshness_window) and compare to the joined value.
  * check_backfill_leakage - a record claims to describe time T (valid_from) but was computed from source
                             data newer than T (needs `max_source_timestamp`).
  * check_online_offline_skew - same (entity, timestamp, feature) via batch vs online path must agree.

Each check returns findings or an explicit skip reason; nothing returns an empty list for "could not run".

Limitations
  * The join is verified against the records you give it. If the records themselves are missing or
    were overwritten in place (no history kept), PIT correctness cannot be reconstructed.
  * Mapping a wrong joined value back to the record it came from is by value equality; when several
    records share a value, the EARLIEST future match is used (conservative: understates the leak).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Finding, LayerResult

LAYER = 3
KEYS = ["entity_id", "event_timestamp"]
SKEW_KEYS = [*KEYS, "feature_name"]
_DELTA_SEVERITY = ((pd.Timedelta(hours=1), 2), (pd.Timedelta(days=1), 3), (pd.Timedelta(days=30), 4))


def _sev_from_delta(delta: pd.Timedelta) -> int:
    return next((sev for limit, sev in _DELTA_SEVERITY if delta < limit), 5)


def _close(a, b, tol):
    """Element-wise equality: numeric within `tol`, otherwise exact; NaN equals NaN. Returns a bool ndarray."""
    a, b = np.asarray(a), np.asarray(b)
    try:
        x, y = a.astype(float), b.astype(float)
    except (TypeError, ValueError):
        return (a == b) | (pd.isna(a) & pd.isna(b))
    return np.isclose(x, y, rtol=0, atol=tol) | (np.isnan(x) & np.isnan(y))


def _require_unique(df, keys, name, why):
    if df.duplicated(keys).any():
        raise ValueError(f"{name} has duplicate {tuple(keys)} keys; {why}")


def _trace_mismatches(bad, recs, tol):
    """Find which stored record each wrong joined value came from (by value equality).
    Returns (deltas: per-row Timedelta of the EARLIEST future match, n_stale, n_unmatched, n_null) where n_null counts
    rows whose joined value is NULL although the store had a value at that time."""
    null = bad.joined_val.isna().to_numpy()
    cand = (
        bad[~null]
        .reset_index(names="row")
        .merge(recs.rename(columns={"value": "rec_value", "computed_at": "rec_at"}), on="entity_id")
    )
    cand = cand[_close(cand.rec_value, cand.joined_val, tol)]
    future = cand[cand.rec_at >= cand.event_timestamp]
    deltas = (future.rec_at - future.event_timestamp).groupby(future.row).min()
    matched = cand.row.nunique()
    return deltas, matched - len(deltas), int((~null).sum()) - matched, int(null.sum())


def _pit_findings(feature, n, window, deltas, stale, unmatched, null):
    out = []
    if len(deltas):
        med = deltas.median()
        out.append(
            Finding(
                LAYER,
                feature,
                "pit-violation",
                _sev_from_delta(med),
                f"{len(deltas)}/{n} rows ({len(deltas) / n:.1%}) joined a '{feature}' value computed AFTER the label's "
                f"event_timestamp: median {med}, max {deltas.max()} in the future (earliest-match, conservative).",
                "Use an as-of join (strictly before event_timestamp, within the freshness window), e.g. "
                "pandas.merge_asof(allow_exact_matches=False) or the store's point-in-time retrieval.",
            )
        )
    if stale:
        out.append(
            Finding(
                LAYER,
                feature,
                "stale-join",
                2,
                f"{stale}/{n} rows joined a '{feature}' value older than the {window} freshness "
                f"window (past values, not a leak, but not what serving would see).",
                "Check the freshness window / TTL.",
            )
        )
    if null:
        out.append(
            Finding(
                LAYER,
                feature,
                "missing-joined-value",
                2,
                f"{null}/{n} rows have a NULL '{feature}' in the training set although the store held a "
                f"value for them as of the event time (join missed it, e.g. TTL too short).",
                "Check the join keys and the freshness window.",
            )
        )
    if unmatched:
        out.append(
            Finding(
                LAYER,
                feature,
                "unmatched-value",
                3,
                f"{unmatched}/{n} rows in '{feature}' differ from the correct as-of value and match no "
                f"stored record (imputed, transformed or corrupted after the join?).",
                "Trace how the training set value was produced.",
            )
        )
    return out


def check_point_in_time(label_records, feature_records, training_set, freshness_window, tol=1e-9):
    """Returns (findings, {feature: rows_checked})."""
    _require_unique(label_records, KEYS, "label_records", "the join to verify is ambiguous.")
    _require_unique(training_set, KEYS, "training_set", "the join to verify is ambiguous.")
    window = pd.Timedelta(freshness_window)
    labels = (
        label_records[KEYS]
        .assign(event_timestamp=pd.to_datetime(label_records.event_timestamp))
        .sort_values("event_timestamp")
    )
    keys = training_set[KEYS].assign(event_timestamp=pd.to_datetime(training_set.event_timestamp))
    findings, rows = [], {}
    absent = ~labels.set_index(KEYS).index.isin(keys.set_index(KEYS).index)
    if absent.any():
        findings.append(
            Finding(
                LAYER,
                "training_set",
                "labels-missing-from-training-set",
                1,
                f"{int(absent.sum())}/{len(labels)} label rows have no row in the training set (dropped by the join or "
                f"later filtering); they were not verified.",
                "Ignore if the drop is intentional.",
            )
        )
    for f in sorted(feature_records.feature_name.unique()):
        if f not in training_set.columns:
            findings.append(
                Finding(
                    LAYER,
                    f,
                    "feature-missing-from-training-set",
                    1,
                    f"Feature '{f}' has records but no column in the training set; not verified.",
                    "Ignore if intentionally excluded.",
                )
            )
            continue
        recs = (
            feature_records.loc[feature_records.feature_name == f, ["entity_id", "value", "computed_at"]]
            .assign(computed_at=lambda d: pd.to_datetime(d.computed_at))
            .sort_values("computed_at")
        )
        correct = pd.merge_asof(
            labels,
            recs,
            left_on="event_timestamp",
            right_on="computed_at",
            by="entity_id",
            allow_exact_matches=False,
            tolerance=window,
        )
        merged = correct.merge(keys.assign(joined_val=training_set[f].to_numpy()), on=KEYS)
        bad = merged[~_close(merged.value, merged.joined_val, tol)]
        rows[f] = len(merged)
        if not bad.empty:
            findings += _pit_findings(f, len(merged), window, *_trace_mismatches(bad, recs, tol))
    return findings, rows


def check_backfill_leakage(feature_records):
    missing = {"valid_from", "max_source_timestamp"} - set(feature_records.columns)
    if missing:
        return None, f"backfill check SKIPPED: feature_records lacks columns {sorted(missing)}."
    fr = feature_records.assign(
        valid_from=pd.to_datetime(feature_records.valid_from),
        max_source_timestamp=pd.to_datetime(feature_records.max_source_timestamp),
    )
    fr["lag"] = fr.max_source_timestamp - fr.valid_from
    totals = fr.feature_name.value_counts()
    out = []
    for f, g in fr[fr.lag > pd.Timedelta(0)].groupby("feature_name"):
        out.append(
            Finding(
                LAYER,
                f,
                "backfill-leakage",
                _sev_from_delta(g.lag.median()),
                f"{len(g)}/{totals[f]} records of '{f}' describe a time (valid_from) earlier "
                f"than the newest source data used to compute them: median {g.lag.median()}, "
                f"max {g.lag.max()}.",
                "Recompute the backfill using only source data with timestamp <= valid_from.",
            )
        )
    return out, None


def check_online_offline_skew(offline, online, tol=1e-6):
    """offline/online: long DataFrames with entity_id, event_timestamp, feature_name, value."""
    if offline is None or online is None:
        return None, "online/offline skew check SKIPPED: both paths must be supplied."
    for name, df in (("offline", offline), ("online", online)):
        _require_unique(df, SKEW_KEYS, f"{name} table", "the comparison would cross-join and report spurious skew.")
    m = offline[[*SKEW_KEYS, "value"]].merge(
        online[[*SKEW_KEYS, "value"]], on=SKEW_KEYS, how="outer", suffixes=("_off", "_on"), indicator=True
    )
    out = []
    for f, g in m.groupby("feature_name"):
        both = g[g["_merge"] == "both"]
        only_one_path = len(g) - len(both)
        n_diff = int((~_close(both.value_off, both.value_on, tol)).sum())
        if not (n_diff or only_one_path):
            continue
        max_diff = np.abs(both.value_off.astype(float) - both.value_on.astype(float)).max() if len(both) else np.nan
        affected = (n_diff + only_one_path) / len(g)
        out.append(
            Finding(
                LAYER,
                f,
                "online-offline-skew",
                4 if affected > 0.05 else 3,
                f"{n_diff} of {len(both)} comparable '{f}' values differ between batch and online "
                f"(max abs diff {max_diff:.4g}); {only_one_path} keys exist in only one path. "
                f"Affected: {affected:.1%}.",
                "Make both paths use one transformation definition; add a parity test.",
            )
        )
    return out, None


def run_layer3(
    label_records=None, feature_records=None, training_set=None, freshness_window=None, offline=None, online=None
):
    if feature_records is None:
        return LayerResult.skipped(
            LAYER, "no feature-store data supplied (feature_records); Layer 3 needs a feature store or a simulated one."
        )
    res = LayerResult(LAYER, "ok")
    if label_records is not None and training_set is not None and freshness_window is not None:
        findings, rows = check_point_in_time(label_records, feature_records, training_set, freshness_window)
        res.findings += findings
        res.notes.append(f"PIT check ran on {sum(rows.values())} feature-row comparisons.")
    else:
        res.notes.append("PIT join check SKIPPED: needs label_records, training_set and freshness_window.")
    for found, why in (check_backfill_leakage(feature_records), check_online_offline_skew(offline, online)):
        if why:
            res.notes.append(why)
        else:
            res.findings += found
    return res
