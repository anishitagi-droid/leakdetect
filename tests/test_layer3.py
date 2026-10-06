import numpy as np
import pandas as pd
import pytest
from leakdetect.layer3_pit import check_point_in_time, check_backfill_leakage, check_online_offline_skew, run_layer3

T0 = pd.Timestamp("2023-01-01")


def store(n=40, seed=0):
    r = np.random.default_rng(seed)
    rec = []
    lab = []
    for e in range(n):
        ev = T0 + pd.Timedelta(days=int(r.integers(60, 120)))
        lab.append((e, ev, int(r.integers(0, 2))))
        # a snapshot every 10 days, value drifts
        rec.extend((e, "spend", float(e + m), T0 + pd.Timedelta(days=m)) for m in range(0, 200, 10))
    return (
        pd.DataFrame(lab, columns=["entity_id", "event_timestamp", "label"]),
        pd.DataFrame(rec, columns=["entity_id", "feature_name", "value", "computed_at"]),
    )


def join(labels, recs, how, window="30D"):
    r = recs[recs.feature_name == "spend"].sort_values("computed_at")
    if how == "latest":
        v = r.groupby("entity_id").value.last()
        return labels[["entity_id", "event_timestamp"]].assign(spend=labels.entity_id.map(v).to_numpy())
    m = pd.merge_asof(
        labels.sort_values("event_timestamp"),
        r,
        left_on="event_timestamp",
        right_on="computed_at",
        by="entity_id",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(window),
    )
    return m[["entity_id", "event_timestamp", "value"]].rename(columns={"value": "spend"})


def test_correct_pit_join_is_clean():
    lab, rec = store()
    f, st = check_point_in_time(lab, rec, join(lab, rec, "asof"), "30D")
    assert f == [] and st["spend"] == len(lab)


def test_naive_latest_join_flagged_with_big_delta():
    lab, rec = store()
    f, _ = check_point_in_time(lab, rec, join(lab, rec, "latest"), "30D")
    v = [x for x in f if x.leakage_type == "pit-violation"]
    assert len(v) == 1 and v[0].severity >= 4 and "100.0%" in v[0].evidence


def test_exact_timestamp_match_is_a_violation_strictly_before():
    lab = pd.DataFrame({"entity_id": [1], "event_timestamp": [T0 + pd.Timedelta(days=10)], "label": [1]})
    rec = pd.DataFrame(
        {
            "entity_id": [1, 1],
            "feature_name": ["spend"] * 2,
            "value": [1.0, 2.0],
            "computed_at": [T0, T0 + pd.Timedelta(days=10)],
        }
    )
    ts = lab[["entity_id", "event_timestamp"]].assign(spend=[2.0])  # picked the value computed AT event time
    f, _ = check_point_in_time(lab, rec, ts, "30D")
    assert f and f[0].leakage_type == "pit-violation"


def test_value_older_than_freshness_window_is_stale_not_a_leak():
    lab, rec = store()
    joined_with_long_window = join(lab, rec, "asof", window="365D")
    types = {x.leakage_type for x in check_point_in_time(lab, rec, joined_with_long_window, "5D")[0]}
    assert "stale-join" in types and "pit-violation" not in types


def test_unmatched_value_flagged():
    lab, rec = store()
    ts = join(lab, rec, "asof")
    ts.loc[ts.index[:5], "spend"] = -999.0
    f, _ = check_point_in_time(lab, rec, ts, "30D")
    assert any(x.leakage_type == "unmatched-value" and x.severity == 3 for x in f)


def test_duplicate_keys_raise_instead_of_silently_misjoining():
    lab, rec = store()
    ts = pd.concat([join(lab, rec, "asof")] * 2)
    with pytest.raises(ValueError):
        check_point_in_time(lab, rec, ts, "30D")


def test_feature_named_value_does_not_break():
    lab, rec = store()
    rec["feature_name"] = "value"
    ts = join(lab, rec.assign(feature_name="spend"), "latest").rename(columns={"spend": "value"})
    f, _ = check_point_in_time(lab, rec, ts, "30D")
    assert any(x.leakage_type == "pit-violation" for x in f)


def test_backfill_leakage_detected_and_skipped_without_columns():
    _, rec = store()
    f, why = check_backfill_leakage(rec)
    assert f is None and "SKIPPED" in why
    rec["valid_from"] = rec.computed_at
    rec["max_source_timestamp"] = rec.computed_at
    rec.loc[rec.index[:20], "max_source_timestamp"] += pd.Timedelta(days=45)
    f, why = check_backfill_leakage(rec)
    assert why is None and f[0].leakage_type == "backfill-leakage" and f[0].severity == 5
    rec["max_source_timestamp"] = rec.valid_from
    assert check_backfill_leakage(rec)[0] == []


def test_online_offline_skew():
    base = pd.DataFrame(
        {"entity_id": range(100), "event_timestamp": T0, "feature_name": "spend", "value": np.arange(100.0)}
    )
    f, why = check_online_offline_skew(base, base.copy())
    assert why is None and f == []
    on = base.copy()
    on.loc[:9, "value"] += 5
    f, _ = check_online_offline_skew(base, on)
    assert f[0].leakage_type == "online-offline-skew"
    f, _ = check_online_offline_skew(base, on.iloc[10:])
    assert "only one path" in f[0].evidence
    assert check_online_offline_skew(base, None)[0] is None


def test_run_layer3_skips_explicitly():
    assert run_layer3().status == "skipped"
    _, rec = store()
    r = run_layer3(feature_records=rec)
    assert r.status == "ok" and any("SKIPPED" in n for n in r.notes)


def test_skew_check_rejects_duplicate_keys():
    base = pd.DataFrame({"entity_id": [1, 1], "event_timestamp": T0, "feature_name": "spend", "value": [1.0, 2.0]})
    with pytest.raises(ValueError):
        check_online_offline_skew(base, base.copy())


def test_label_rows_absent_from_training_set_are_reported_not_called_unmatched():
    """Regression (found on real stock data): dropped rows were counted as 'values matching no stored record'."""
    lab, rec = store()
    ts = join(lab, rec, "latest").iloc[5:]
    kinds = {x.leakage_type: x for x in check_point_in_time(lab, rec, ts, "30D")[0]}
    assert set(kinds) == {"pit-violation", "labels-missing-from-training-set"}  # absent rows are not "wrong values"
    absent = kinds["labels-missing-from-training-set"]
    assert absent.severity == 1 and "5/40" in absent.evidence


def test_null_joined_value_is_its_own_finding():
    lab, rec = store()
    ts = join(lab, rec, "asof")
    ts.loc[ts.index[:4], "spend"] = np.nan
    kinds = {x.leakage_type for x in check_point_in_time(lab, rec, ts, "30D")[0]}
    assert kinds == {"missing-joined-value"}
