import warnings

import numpy as np
import pandas as pd
import pytest

from leakdetect.layer4_stats import _material_gap, group_leakage, prepare_xy, run_layer4, split_similarity_check

warnings.filterwarnings("ignore")


def synth(n=800, leak=None, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({f"x{i}": r.normal(size=n) for i in range(6)})
    p = 1 / (1 + np.exp(-(0.8 * df.x0 + 0.5 * df.x1)))
    df["y"] = (r.random(n) < p).astype(int)
    if leak == "proxy":  # noisy near-copy of the label
        df["leaky"] = df.y + r.normal(scale=0.05, size=n)
    if leak == "conditional_zero":  # exists only when y==1 (like AER expenditure)
        df["after_event"] = np.where(df.y == 1, np.abs(r.normal(size=n)) + 0.2, 0.0)
    return df


def feats(res, ltype=None):
    return {f.feature_or_step for f in res.findings if ltype is None or f.leakage_type == ltype}


def test_clean_synthetic_data_no_target_proxy_findings():
    r = run_layer4(synth(), "y", 1)
    assert not [
        f for f in r.findings if f.leakage_type in ("single-feature-predictive-power", "perfect-separation-rule")
    ]


def test_label_proxy_feature_caught():
    r = run_layer4(synth(leak="proxy"), "y", 1)
    assert "leaky" in feats(r, "single-feature-predictive-power")


def test_conditional_zero_feature_caught_by_stump_rule():
    r = run_layer4(synth(leak="conditional_zero"), "y", 1)
    assert "after_event" in feats(r, "perfect-separation-rule")


def test_train_test_too_similar_detected_when_test_is_copy_of_train():
    X, _, _ = prepare_xy(synth(), "y", 1)
    tr = X.iloc[:500]
    te = tr.iloc[:200] + np.random.default_rng(1).normal(scale=1e-4, size=(200, X.shape[1]))
    f, _ = split_similarity_check(tr, te, n_perm=200)
    assert f and f[0].leakage_type == "train-test-too-similar"


def test_random_split_not_flagged_as_similar():
    X, _, _ = prepare_xy(synth(), "y", 1)
    f, _ = split_similarity_check(X.iloc[:600], X.iloc[600:], n_perm=200)
    assert not [x for x in f if x.leakage_type == "train-test-too-similar"]


def test_post_period_metadata_flagged_and_skip_noted():
    r = run_layer4(synth(), "y", 1, feature_metadata={"x0": "application", "x1": "post_application"})
    assert feats(r, "post-period-feature") == {"x1"}
    assert any("SKIPPED" in n for n in run_layer4(synth(), "y", 1).notes)


def test_prepare_xy_imputes_with_missing_indicator_and_leaves_no_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "c": ["u", None, "v", "u"], "y": [0, 1, 0, 1]})
    X, _, notes = prepare_xy(df, "y", 1)
    assert not X.isna().any().any()
    assert {"a__missing", "c_u", "c_v"} <= set(X.columns)  # missing category becomes the dropped baseline level
    assert X["a__missing"].tolist() == [0, 1, 0, 0] and any("median-imputed" in n for n in notes)
    assert X["a"].tolist() == [1.0, 3.0, 3.0, 4.0]  # NaN -> median of [1, 3, 4]


def test_prepare_xy_drops_id_like_and_datetime_columns_and_says_so():
    n = 120
    df = pd.DataFrame(
        {
            "x": np.arange(n) % 7,
            "name": [f"person {i}" for i in range(n)],
            "when": pd.date_range("2020-01-01", periods=n),
            "y": np.arange(n) % 2,
        }
    )
    X, _, notes = prepare_xy(df, "y", 1, max_levels=50)
    assert list(X.columns) == ["x"] and len(notes) == 2


@pytest.mark.parametrize(
    "kwargs", [{"positive": "nope"}, {"positive": 1, "nan_target": True}, {"positive": 1, "one_class": True}]
)
def test_prepare_xy_rejects_unusable_targets_with_a_clear_error(kwargs):
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    if kwargs.get("nan_target"):
        df.loc[0, "y"] = np.nan
    if kwargs.get("one_class"):
        df["y"] = 1
    with pytest.raises(ValueError):
        prepare_xy(df, "y", kwargs["positive"])


def test_vectorised_ks_equals_scipy_including_heavy_ties():
    from scipy.stats import ks_2samp
    from leakdetect.layer4_stats import _ks_per_column, _ks_prepare

    rng = np.random.default_rng(1)
    P = np.column_stack(
        [
            rng.integers(0, 4, 300),
            rng.normal(size=300),
            np.round(rng.normal(size=300), 1),
            rng.integers(0, 2, 300) * 1.0,
        ]
    )
    in_a = rng.permutation(np.arange(300) < 180)
    order, last = _ks_prepare(P)
    ref = [ks_2samp(P[in_a, j], P[~in_a, j]).statistic for j in range(4)]
    assert np.allclose(_ks_per_column(in_a, order, last, 180, 120), ref)


def test_monotone_auc_is_direction_aware():
    from leakdetect.layer4_stats import _folds, _monotone_cv_auc

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    A = np.column_stack([-y + rng.normal(scale=0.2, size=400), rng.normal(size=400)])  # col0 negatively associated
    auc = _monotone_cv_auc(A, y, _folds(y, 0))
    assert auc[0] > 0.95 and 0.4 < auc[1] < 0.6


def test_wide_pure_noise_produces_no_mi_or_separation_flags():
    """Regression: the old 'MI z >= 2' rule flagged 42 of 750 pure-noise features."""
    from leakdetect.layer4_stats import high_mutual_information, perfect_separation

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(252, 400)), columns=[f"n{i}" for i in range(400)])
    y = (rng.random(252) < 0.75).astype(int)
    assert high_mutual_information(X, y)[0] == [] and perfect_separation(X, y) == []


def test_raw_dataframe_with_nans_strings_and_a_copy_of_the_target_is_caught_end_to_end():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "survived": y,
            "alive": np.where(y == 1, "yes", "no"),
            "age": rng.normal(30, 10, n),
            "sex": rng.choice(["m", "f"], n),
            "name": [f"p{i}" for i in range(n)],
        }
    )
    df.loc[rng.random(n) < 0.2, "age"] = np.nan
    r = run_layer4(df, "survived", 1)
    assert {f.feature_or_step for f in r.findings if f.severity >= 4} == {"alive_yes"}
    assert any("dropped" in note and "name" in note for note in r.notes)


def test_partial_tail_rule_is_severity_3_but_complete_rule_is_severity_4():
    from leakdetect.layer4_stats import perfect_separation

    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.7).astype(int)
    tail = rng.normal(scale=0.01, size=n)
    tail[np.flatnonzero(y == 1)[:80]] += 1.0  # 80 rows, all positive: a clean but PARTIAL tail (~19% of positives)
    full = np.where(y == 1, rng.random(n) + 1, 0.0)  # zero iff negative: separates both classes completely
    found = {f.feature_or_step: f.severity for f in perfect_separation(pd.DataFrame({"tail": tail, "full": full}), y)}
    assert found == {"tail": 3, "full": 4}


# ---- messy-input regressions found by fuzzing the entry points (each used to crash with an opaque sklearn error)
def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"y": rng.integers(0, 2, n), "a": rng.normal(size=n), "b": rng.normal(size=n)})


def test_infinite_values_are_treated_as_missing_and_reported():
    df = _frame()
    df.loc[3, "a"], df.loc[4, "b"] = np.inf, -np.inf
    res = run_layer4(df, "y", 1)
    assert any("2 infinite value(s)" in n for n in res.notes)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: pd.concat([_frame(), _frame()[["a"]]], axis=1), "duplicate column names"),
        (lambda: _frame(10), "too few"),
        (lambda: _frame().assign(y=lambda d: (np.arange(len(d)) == 0).astype(int)), "fewer than 2 rows"),
        (lambda: _frame()[["y"]], "no usable feature columns"),
    ],
    ids=["duplicate-columns", "ten-rows", "one-row-class", "no-features"],
)
def test_unusable_inputs_raise_a_clear_error(make, message):
    with pytest.raises(ValueError, match=message):
        run_layer4(make(), "y", 1)


def test_class_with_three_rows_uses_fewer_folds_instead_of_failing():
    df = _frame().assign(y=lambda d: (np.arange(len(d)) < 3).astype(int))
    assert run_layer4(df, "y", 1).status == "ok"


# ---- impact estimate
def _leaky_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-0.8 * x0))).astype(int)  # x0 carries moderate signal (AUC ~0.7)
    return pd.DataFrame({"y": y, "x0": x0, "leak": y + rng.normal(scale=0.05, size=n)})


def test_impact_reports_how_far_auc_falls_without_the_flagged_feature():
    res = run_layer4(_leaky_frame(), "y", 1)
    impact = res.extras["impact"]
    assert impact["features"] == ["leak"]
    assert impact["with"] > 0.95 and 0.6 < impact["without"] < 0.8  # honest model keeps only x0's signal
    assert any("Removing ['leak']" in n for n in res.notes)


def test_no_impact_estimate_when_nothing_is_flagged_at_severity_4():
    res = run_layer4(_frame(), "y", 1)
    assert "impact" not in res.extras and not any("Removing" in n for n in res.notes)


def test_no_impact_estimate_when_every_feature_is_flagged():
    df = _leaky_frame()[["y", "leak"]]
    assert "impact" not in run_layer4(df, "y", 1).extras  # nothing would be left to fit


# ---- entity (group) leakage
def _repeated_measures(n_subjects=60, rows=6, identifiable=True, seed=0, noise=0.3):
    """Subject-level random labels. If `identifiable`, every row carries its subject's fixed feature vector, so a
    random split lets the model memorise subjects (a leak by construction); otherwise features are pure noise."""
    rng = np.random.default_rng(seed)
    subject = np.repeat(np.arange(n_subjects), rows)
    label = rng.permutation(np.arange(n_subjects) % 2)[subject]  # constant within a subject, unrelated to anything
    signature = rng.normal(size=(n_subjects, 5))[subject] if identifiable else rng.normal(size=(len(subject), 5))
    X = signature + rng.normal(scale=noise, size=signature.shape)
    return pd.DataFrame(X, columns=list("abcde")).assign(y=label, subject=subject)


def _xy(df):
    return df[list("abcde")], df["y"].to_numpy()


def test_group_leakage_flags_memorisable_entities_with_a_large_gap():
    df = _repeated_measures()
    found, info = group_leakage(*_xy(df), df["subject"], "subject", repeats=3)
    assert len(found) == 1 and found[0].leakage_type == "group-leakage" and found[0].severity == 4
    assert info["random_auc"] > 0.85 and info["grouped_auc"] < 0.65 and info["same_group_share"] > 0.9


def test_meaningless_groups_over_independent_rows_are_not_flagged():
    df = _repeated_measures(identifiable=False)
    shuffled_ids = np.random.default_rng(1).permutation(df["subject"].to_numpy())  # no structure at all
    assert group_leakage(*_xy(df), shuffled_ids, repeats=3)[0] == []


def test_overlap_without_identifiable_features_is_not_flagged():
    df = _repeated_measures(identifiable=False)  # same subjects on both sides, but features carry no identity
    found, info = group_leakage(*_xy(df), df["subject"], repeats=3)
    assert found == [] and info["same_group_share"] > 0.9 and abs(info["gap"]) < 0.03


def test_group_check_through_run_layer4_excludes_the_column_and_explains_harmless_overlap():
    res = run_layer4(_repeated_measures(identifiable=False), "y", 1, groups="subject")
    assert "subject" not in res.extras["single_feature"].feature.tolist()
    assert any("no measurable inflation" in n and "share a 'subject'" in n for n in res.notes)


def test_group_check_input_validation():
    df = _repeated_measures()
    X, y = _xy(df)
    with pytest.raises(ValueError, match="not in the data"):
        run_layer4(df, "y", 1, groups="nope")
    with pytest.raises(ValueError, match="missing values"):
        run_layer4(df.assign(subject=df.subject.where(df.subject != 3)), "y", 1, groups="subject")
    with pytest.raises(ValueError, match="at least 10 distinct groups"):
        group_leakage(X, y, df.subject % 5)
    assert "own group" in group_leakage(X, y, np.arange(len(df)))[1]["skipped"]


def test_moderate_gap_is_flagged_at_severity_3_not_4():
    df = _repeated_measures(noise=1.1)  # measured gaps 0.066-0.092 over three seeds
    found, info = group_leakage(*_xy(df), df["subject"], "subject", repeats=3)
    assert [f.severity for f in found] == [3] and 0.03 <= info["gap"] < 0.10


def test_same_group_share_is_the_fraction_of_test_rows_with_a_sibling_in_train():
    # two rows per group, 75/25 split: P(sibling in train) ~ 0.75; SD of the share on 150 test rows ~ 0.035
    rng = np.random.default_rng(0)
    X, y = pd.DataFrame(rng.normal(size=(600, 3))), rng.integers(0, 2, 600)
    share = group_leakage(X, y, np.repeat(np.arange(300), 2), repeats=2)[1]["same_group_share"]
    assert 0.64 < share < 0.86


def test_material_gap_needs_both_practical_size_and_statistical_clarity():
    assert _material_gap(0.05, 0.01, 0.03)
    assert not _material_gap(0.02, 0.001, 0.03)  # statistically clear but too small to matter
    assert not _material_gap(0.05, 0.03, 0.03)  # large enough but within 3 SE of noise
    assert not _material_gap(-0.10, 0.01, 0.03)  # grouped folds scoring HIGHER is not leakage


def test_min_gap_parameter_is_respected():
    df = _repeated_measures()  # measured gap ~ +0.45
    assert group_leakage(*_xy(df), df["subject"], repeats=3, min_gap=0.6)[0] == []
