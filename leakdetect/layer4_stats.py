"""Layer 4 - statistical / empirical detection.

Works on any static tabular dataset. Sub-checks:
  1. single-feature predictive power: CV AUC of a monotone scorer and of a depth-2 tree, per feature
  2. high mutual information: a feature carrying >= `mi_ratio` of the target's entropy
  3. perfect-separation rule: depth-1 stump with a large, 100%-pure side that chance cannot explain
  4. train/test similarity: permutation test on the mean KS statistic
  5. post-period feature check (needs caller-supplied metadata; otherwise skipped)

Honest limitations (also surfaced in `notes`):
  * Findings are target-proxy EVIDENCE for human review, not proof of leakage: some strong predictors are legitimate.
  * Thresholds are heuristics. Checks 2 and 3 use absolute / multiplicity-corrected criteria on purpose: a
    "z >= 2 above the other features" rule flags ~2% of any feature set, which measured 42 false flags on pure noise.
  * Descriptive checks run on a median-imputed copy of the FULL data (missingness is kept as `<col>__missing`
    indicator columns). This is a diagnostic, not an evaluation, so fitting on all rows is intentional here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from .schema import Finding, LayerResult

LAYER = 4
MAX_LEVELS = 50  # categorical columns with more levels are dropped (IDs, free text, names)
MIN_ROWS = 20  # below this, cross-validation and nearest-neighbour MI estimates are meaningless


def prepare_xy(df: pd.DataFrame, target: str, positive, drop_cols=(), max_levels=MAX_LEVELS):
    """Returns (X, y, notes): a NaN-free float matrix, a 0/1 target, and a note for every transformation applied."""
    if df.columns.duplicated().any():
        raise ValueError(f"duplicate column names: {sorted(set(df.columns[df.columns.duplicated()]))}")
    if df[target].isna().any():
        raise ValueError(f"target '{target}' contains missing values; drop or fill those rows first")
    if positive not in set(df[target]):
        raise ValueError(f"positive label {positive!r} not in target values {sorted(map(str, df[target].unique()))}")
    y = (df[target] == positive).astype(int).to_numpy()
    if y.min() == y.max():
        raise ValueError("target has a single class; nothing to predict")
    if np.bincount(y).min() < 2:
        raise ValueError("the smaller class has fewer than 2 rows; it cannot be cross-validated")
    X, notes = df.drop(columns=[target, *drop_cols], errors="ignore"), []

    num = X.select_dtypes(include=["number", "bool"])
    n_inf = int(np.isinf(num.astype(float)).sum().sum())
    if n_inf:
        num = num.astype(float).replace([np.inf, -np.inf], np.nan)
        notes.append(f"treated {n_inf} infinite value(s) as missing")
    when = [c for c in X.columns if is_datetime64_any_dtype(X[c]) or is_timedelta64_dtype(X[c])]
    cat = X.drop(columns=[*num.columns, *when]).astype("object")
    wide = [c for c in cat if cat[c].nunique() > max_levels]
    all_nan = [c for c in num if num[c].isna().all()]
    for cols, why in (
        (when, "datetime/timedelta (convert to numeric features first)"),
        (wide, f"more than {max_levels} distinct categories (ID / free text?)"),
        (all_nan, "entirely missing"),
    ):
        if cols:
            notes.append(f"dropped {len(cols)} column(s), {why}: {cols[:8]}")
    num, cat = num.drop(columns=all_nan).astype(float), cat.drop(columns=wide).fillna("__missing__")
    missing = [c for c in num if num[c].isna().any()]
    if missing:
        notes.append(
            f"median-imputed {len(missing)} numeric column(s) and added `<col>__missing` indicators: {missing[:8]}"
        )
    parts = [
        num.fillna(num.median()),
        num[missing].isna().astype(float).add_suffix("__missing"),
        pd.get_dummies(cat, drop_first=True, dtype=float) if cat.shape[1] else cat,
    ]
    features = pd.concat(parts, axis=1).reset_index(drop=True)
    if features.shape[1] == 0:
        raise ValueError("no usable feature columns left" + (f": {'; '.join(notes)}" if notes else ""))
    return features, y, notes


def _folds(y, seed):
    n_splits = min(5, int(np.bincount(y).min()))  # a class with 3 rows cannot be split 5 ways
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _rank_auc(M, y):
    """ROC AUC of every column of M against y (ties averaged), vectorised."""
    n1 = int(y.sum())
    ranks = rankdata(M, axis=0)
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def _monotone_cv_auc(A, y, folds):
    """CV AUC of the best monotone scorer per column, direction chosen on the training fold. This is the ranking a
    one-feature logistic regression produces, without fitting 5 x n_features models."""
    out = np.zeros(A.shape[1])
    for tr, te in folds:
        auc_te = _rank_auc(A[te], y[te])
        out += np.where(_rank_auc(A[tr], y[tr]) >= 0.5, auc_te, 1 - auc_te)
    return out / len(folds)


def _tree_cv_auc(A, y, folds, seed):
    out = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        col = A[:, [j]]
        for tr, te in folds:
            tree = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(col[tr], y[tr])
            out[j] += roc_auc_score(y[te], tree.predict_proba(col[te])[:, 1])
    return out / len(folds)


def single_feature_power(X, y, auc_threshold=0.95, seed=0):
    A, folds = X.to_numpy(), _folds(y, seed)
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "auc_monotone": _monotone_cv_auc(A, y, folds),
            "auc_tree": _tree_cv_auc(A, y, folds, seed),
        }
    )
    table["auc"] = table[["auc_monotone", "auc_tree"]].max(axis=1)
    table = table.sort_values("auc", ascending=False).reset_index(drop=True)
    full = cross_val_score(
        RandomForestClassifier(200, random_state=seed, n_jobs=-1), X, y, cv=folds, scoring="roc_auc"
    ).mean()
    findings = [
        Finding(
            LAYER,
            r.feature,
            "single-feature-predictive-power",
            5 if r.auc >= 0.99 else 4,
            f"'{r.feature}' alone reaches CV AUC {r.auc:.3f} (monotone {r.auc_monotone:.3f}, "
            f"depth-2 tree {r.auc_tree:.3f}); full 5-fold random forest AUC is {full:.3f}. "
            f"A single column matching the entire model's performance is a strong target-proxy signal.",
            f"Check whether '{r.feature}' is observable before the label exists. "
            f"Drop it, or refit and report metrics without it.",
        )
        for r in table.itertuples()
        if r.auc >= auc_threshold
    ]
    return findings, table, full


def high_mutual_information(X, y, ratio_threshold=0.5, seed=0):
    """Flag features whose mutual information with the target is >= `ratio_threshold` of the target's entropy.
    Absolute criterion: unlike 'z-score above the other features', it does not flag a fixed share of any feature set."""
    discrete = np.array([X[c].nunique() <= 10 for c in X.columns])
    mi = pd.Series(mutual_info_classif(X, y, discrete_features=discrete, random_state=seed), index=X.columns)
    p = y.mean()
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # nats, same unit as sklearn's MI
    ratio = mi / entropy
    table = pd.DataFrame({"mi": mi, "ratio_of_target_entropy": ratio}).sort_values("mi", ascending=False)
    findings = [
        Finding(
            LAYER,
            f,
            "high-mutual-information",
            3,
            f"'{f}' shares {mi[f]:.3f} nats with the target = {ratio[f]:.0%} of the target's entropy "
            f"({entropy:.3f}); the median feature shares {ratio.median():.1%}.",
            "Manual review: is this feature a consequence of the outcome?",
        )
        for f in table.index
        if ratio[f] >= ratio_threshold
    ]
    return findings, table


def perfect_separation(X, y, min_support=0.10, alpha=0.01, strong_coverage=0.9):
    """Depth-1 stump per feature. Flags a side that is 100% one class, holds >= min_support of the rows and whose
    chance probability (class prior ** rows, Bonferroni-corrected over features x thresholds) is below alpha.
    Severity 4 when the rule classifies >= `strong_coverage` of its class perfectly (leak-grade) or chance is
    < 1e-12; otherwise 3, because a clean tail can also be a legitimately strong feature."""
    n, d = len(y), X.shape[1]
    prior = {0: 1 - y.mean(), 1: y.mean()}
    findings = []
    for c in X.columns:
        stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[[c]], y)
        if stump.tree_.node_count < 3:
            continue
        thr = stump.tree_.threshold[0]
        low = X[c].to_numpy() <= thr
        rules, strong = [], False
        for op, mask in (("<=", low), (">", ~low)):
            k = int(mask.sum())
            if k < min_support * n or y[mask].min() != y[mask].max():
                continue
            cls = int(y[mask][0])
            p = min(1.0, prior[cls] ** k * d * n)
            if p >= alpha:
                continue
            coverage = k / int((y == cls).sum())
            strong |= coverage >= strong_coverage or p < 1e-12
            rules.append(
                f"'{c} {op} {thr:.4g}' covers {k} rows ({k / n:.0%}), all class {cls} = {coverage:.0%} of "
                f"that class (chance probability {p:.1e} after correcting for {d} features x {n} thresholds)"
            )
        if rules:
            tail = (
                " Real-world features rarely give a perfect rule this complete."
                if strong
                else " Partial rule: this can also be a legitimately strong feature; review how it is measured."
            )
            findings.append(
                Finding(
                    LAYER,
                    c,
                    "perfect-separation-rule",
                    4 if strong else 3,
                    "; ".join(rules) + "." + tail,
                    f"Investigate how '{c}' is constructed; a value that exists only for one outcome "
                    f"is usually produced after the outcome.",
                )
            )
    return findings


def _ks_prepare(P):
    """Sort every column once. `last_of_tie` marks where a tie block ends: the ECDF gap is only evaluated there."""
    order = np.argsort(P, axis=0, kind="stable")
    sorted_vals = np.take_along_axis(P, order, axis=0)
    return order, np.vstack([sorted_vals[1:] != sorted_vals[:-1], np.ones((1, P.shape[1]), bool)])


def _ks_per_column(in_a, order, last_of_tie, na, nb):
    """Exact two-sample KS statistic for every column at once (== scipy ks_2samp's statistic)."""
    a = in_a[order]
    gap = np.abs(np.cumsum(a, axis=0) / na - np.cumsum(~a, axis=0) / nb)
    return np.where(last_of_tie, gap, 0.0).max(axis=0)


def split_similarity_check(X_train, X_test, n_perm=300, seed=0, alpha=0.01):
    """Mean per-feature KS stat of the actual split vs. random re-splits of the pooled data.
    Observed far BELOW the permutation distribution -> train/test too similar (contamination).
    Observed far ABOVE -> covariate shift (not leakage, reported as informational)."""
    num = [c for c in X_train.columns if X_train[c].nunique() > 2]
    if not num:
        return [], {"skipped": "no non-binary numeric features"}
    P = np.vstack([X_train[num].to_numpy(), X_test[num].to_numpy()])
    na, nb = len(X_train), len(X_test)
    order, last_of_tie = _ks_prepare(P)

    def mean_ks(in_a):
        return _ks_per_column(in_a, order, last_of_tie, na, nb).mean()

    obs_mask = np.arange(len(P)) < na
    obs = mean_ks(obs_mask)
    rng = np.random.default_rng(seed)
    perm = np.array([mean_ks(rng.permutation(obs_mask)) for _ in range(n_perm)])
    p_low = (np.sum(perm <= obs) + 1) / (n_perm + 1)
    p_high = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    info = {
        "observed_mean_ks": float(obs),
        "perm_mean": float(perm.mean()),
        "perm_sd": float(perm.std()),
        "p_too_similar": float(p_low),
        "p_shifted": float(p_high),
        "n_perm": n_perm,
    }
    out = []
    if p_low < alpha:
        out.append(
            Finding(
                LAYER,
                "train/test split",
                "train-test-too-similar",
                4,
                f"Mean KS {obs:.4f} is lower than {1 - p_low:.1%} of random re-splits "
                f"(perm mean {perm.mean():.4f}, p={p_low:.4f}). Possible duplicated or "
                f"leaked rows across the split.",
                "Run duplicate detection across splits.",
            )
        )
    if p_high < alpha:
        out.append(
            Finding(
                LAYER,
                "train/test split",
                "train-test-shift",
                2,
                f"Mean KS {obs:.4f} exceeds {1 - p_high:.1%} of random re-splits (p={p_high:.4f}). "
                f"This is covariate shift (non-random split), not leakage per se.",
                "Confirm the split is intentional (e.g. temporal).",
            )
        )
    return out, info


def post_period_features(feature_metadata):
    """Metadata-driven: {feature: 'application' | 'post_application' | 'unknown'}."""
    return [
        Finding(
            LAYER,
            f,
            "post-period-feature",
            5,
            f"'{f}' is declared as computed from information after the prediction point.",
            "Remove it or recompute using only pre-prediction data.",
        )
        for f, when in feature_metadata.items()
        if when == "post_application"
    ]


def run_layer4(
    df, target, positive, drop_cols=(), feature_metadata=None, test_size=0.25, seed=0, auc_threshold=0.95, mi_ratio=0.5
):
    if len(df) < MIN_ROWS:
        raise ValueError(f"{len(df)} rows is too few for these statistics (minimum {MIN_ROWS})")
    X, y, notes = prepare_xy(df, target, positive, drop_cols)
    res = LayerResult(LAYER, "ok", notes=notes)
    f1, tbl, full = single_feature_power(X, y, auc_threshold, seed)
    f2, mi_tbl = high_mutual_information(X, y, mi_ratio, seed)
    Xtr, Xte = train_test_split(X, test_size=test_size, random_state=seed, stratify=y)
    f4, ks_info = split_similarity_check(Xtr, Xte, seed=seed)
    res.findings += f1 + f2 + perfect_separation(X, y) + f4
    if feature_metadata is None:
        res.notes.append(
            "post-period feature check SKIPPED: no feature_metadata supplied "
            "(the dataset itself does not say when each feature is measured)."
        )
    else:
        res.findings += post_period_features(feature_metadata)
    res.notes.append(
        "Single-feature AUC / MI / stump findings are target-proxy EVIDENCE for human review, not proof of leakage."
    )
    res.extras = {"single_feature": tbl, "mutual_information": mi_tbl, "ks": ks_info, "full_model_auc": full}
    return res
