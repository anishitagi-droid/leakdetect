import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from leakdetect.layer2_runtime import (
    ProvenanceTracker,
    track_provenance,
    tracked_train_test_split,
    find_cross_split_duplicates,
    _near_pairs_bruteforce,
    _near_pairs_lsh,
    _numeric_matrix,
)


@track_provenance
class TScaler(StandardScaler):
    pass


@track_provenance
class TSelect(SelectKBest):
    pass


def data(n=400, d=6, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = (X.f0 + r.normal(size=n) > 0).astype(int).to_numpy()
    return X, y


def test_scaler_fit_on_full_data_then_split_is_contamination():
    X, y = data()
    with ProvenanceTracker() as t:
        Xs = TScaler().fit_transform(X)
        tracked_train_test_split(Xs, y, test_size=0.25, random_state=0)
    r = t.result()
    f = [x for x in r.findings if x.leakage_type == "fit-contamination"]
    assert r.status == "ok" and len(f) == 1 and "25%" in f[0].evidence  # 100 of 400 rows


def test_clean_split_first_no_contamination():
    X, y = data()
    with ProvenanceTracker() as t:
        Xtr, Xte, *_ = tracked_train_test_split(X, y, test_size=0.25, random_state=0)
        s = TScaler()
        s.fit_transform(Xtr)
        s.transform(Xte)
    r = t.result()
    assert r.status == "ok" and r.findings == []


def test_split_registered_before_fit_is_caught_at_fit_time():
    X, y = data()
    with ProvenanceTracker() as t:
        tracked_train_test_split(X, y, test_size=0.25, random_state=0)
        TScaler().fit(X)  # fit on everything AFTER splitting
    assert [f.leakage_type for f in t.result().findings] == ["fit-contamination"]


def test_fit_on_test_partition_only_flagged_as_all_held_out():
    X, y = data()
    with ProvenanceTracker() as t:
        _, Xte, *_ = tracked_train_test_split(X, y, test_size=0.25, random_state=0)
        TScaler().fit(Xte)
    f = t.result().findings
    assert f and "100%" in f[0].evidence


def test_supervised_selection_gets_higher_severity_than_scaler():
    X, y = data()
    with ProvenanceTracker() as t:
        TSelect(k=2).fit(X, y)
        TScaler().fit(X)
        tracked_train_test_split(X, y, test_size=0.25, random_state=0)
    sev = {
        f.feature_or_step.split("#")[0]: f.severity
        for f in t.result().findings
        if f.leakage_type == "fit-contamination"
    }
    assert sev["TSelect"] > sev["TScaler"]


def test_no_split_registered_is_skipped_not_clean():
    X, _ = data()
    with ProvenanceTracker() as t:
        TScaler().fit_transform(X)
    r = t.result()
    assert r.status == "skipped" and "no split registered" in r.reason


def test_nested_fit_transform_not_double_recorded():
    X, _ = data()
    with ProvenanceTracker() as t:
        TScaler().fit_transform(X)
    assert len(t.events) == 1


def test_instance_wrapping_works():
    X, y = data()
    with ProvenanceTracker() as t:
        sc = track_provenance(StandardScaler())
        Xs = sc.fit_transform(X)
        tracked_train_test_split(Xs, y, test_size=0.25, random_state=0)
    assert len(t.result().findings) == 1


def test_exact_duplicates_across_split_detected():
    X, y = data()
    Xd = pd.concat([X, X.iloc[:40]], ignore_index=True)
    yd = np.concatenate([y, y[:40]])
    with ProvenanceTracker() as t:
        tracked_train_test_split(Xd, yd, test_size=0.3, random_state=1)
    f = [x for x in t.result().findings if x.leakage_type == "exact-duplicate-across-splits"]
    assert len(f) == 1 and f[0].severity >= 4


def test_no_duplicates_no_finding():
    X, y = data()
    with ProvenanceTracker() as t:
        tracked_train_test_split(X, y, test_size=0.3, random_state=1)
    assert not [x for x in t.result().findings if "duplicate" in x.leakage_type]


def test_near_duplicates_detected_and_lsh_matches_bruteforce():
    X, _ = data(n=600)
    noisy = X.iloc[:60] + np.random.default_rng(5).normal(scale=0.005, size=(60, X.shape[1]))
    tr, te = X.iloc[60:360], pd.concat([noisy, X.iloc[360:]], ignore_index=True)
    tr = pd.concat([X.iloc[:60], tr], ignore_index=True)  # train holds the originals
    f = find_cross_split_duplicates(tr, te, mode="near", eps=0.05)
    assert f and f[0].leakage_type == "near-duplicate-across-splits"
    A, B = _numeric_matrix(tr, te)
    bf = {(i, j) for i, j, _ in _near_pairs_bruteforce(A, B, 0.05)}
    lsh = {(i, j) for i, j, _ in _near_pairs_lsh(A, B, 0.05)}
    assert lsh <= bf  # LSH verified by true distance -> no false positives
    recall = len(lsh) / len(bf)
    assert recall >= 0.95, recall


def test_duplicate_evidence_warns_when_compared_on_few_columns():
    r = np.random.default_rng(0)
    Xlow = pd.DataFrame({"a": r.integers(0, 3, 300), "b": r.integers(0, 2, 300)})
    f = find_cross_split_duplicates(Xlow.iloc[:200], Xlow.iloc[200:], mode="exact")
    assert f and "only 2 column(s)" in f[0].evidence and "(compared on 2 column(s))" in f[0].evidence


def test_instrumented_estimator_outside_tracker_records_nothing_and_still_works():
    X, _ = data()
    out = TScaler().fit_transform(X)
    assert out.shape == X.shape
    with ProvenanceTracker() as t:
        pass
    assert t.events == []


def test_tracked_split_without_tracker_raises_instead_of_hiding_state():
    X, y = data()
    with pytest.raises(RuntimeError, match="active tracker"):
        tracked_train_test_split(X, y)


def test_fit_on_train_with_cross_split_duplicates_is_not_contamination():
    X, y = data()
    Xd, yd = pd.concat([X, X.iloc[:40]], ignore_index=True), np.concatenate([y, y[:40]])
    with ProvenanceTracker() as t:
        Xtr, *_ = tracked_train_test_split(Xd, yd, test_size=0.3, random_state=1)
        TScaler().fit(Xtr)
    kinds = {f.leakage_type for f in t.result().findings}
    assert "exact-duplicate-across-splits" in kinds
    assert "fit-contamination" not in kinds  # a row present in train AND test is ambiguous, not held-out


def test_exact_duplicate_severity_scales_with_share_of_test_rows():
    X, _ = data()
    train = X.iloc[:300]
    heavy = pd.concat([train.iloc[:10], X.iloc[300:390]], ignore_index=True)  # 10% of test copied from train
    light = pd.concat([train.iloc[:2], X.iloc[300:398]], ignore_index=True)  # 2%
    assert find_cross_split_duplicates(train, heavy)[0].severity == 5
    assert find_cross_split_duplicates(train, light)[0].severity == 4
