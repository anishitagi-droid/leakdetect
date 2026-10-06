"""End-to-end demo on AER Credit Card Applications (n=1319).
Layers 1, 2, 4 run on the REAL data. Layer 3 needs timestamps AER does not have, so it runs on a
SYNTHETIC feature-store scenario built around AER rows and labelled as synthetic everywhere."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leakdetect.aggregator import aggregate, to_json, to_markdown
from leakdetect.layer1_static import analyze_file
from leakdetect.layer2_runtime import (
    ProvenanceTracker,
    find_cross_split_duplicates,
    track_provenance,
    tracked_train_test_split,
)
from leakdetect.layer3_pit import run_layer3
from leakdetect.layer4_stats import prepare_xy, run_layer4

warnings.filterwarnings("ignore")
LEAK_COLS = ["expenditure", "share"]
FEATURE = "avg_monthly_spend"
MONTHS = np.arange(-6, 7)  # monthly snapshots from 6 months before to 6 after the application


@track_provenance
class TScaler(StandardScaler):
    pass


@track_provenance
class TSelect(SelectKBest):
    pass


def summarize(findings):
    return [(f.leakage_type, f.severity) for f in findings]


def rf_auc(X, y, seed):
    return cross_val_score(
        RandomForestClassifier(200, random_state=seed, n_jobs=-1),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="roc_auc",
    ).mean()


def run_layer2(X, y):
    with ProvenanceTracker() as leaky:  # fit on everything, then split
        Xs = TScaler().fit_transform(X)
        Xf = TSelect(f_classif, k=5).fit_transform(Xs, y)
        tracked_train_test_split(Xf, y, test_size=0.25, random_state=0, stratify=y)
    with ProvenanceTracker() as clean:  # split first, fit on train
        Xtr, Xte, _, _ = tracked_train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        scaler = TScaler()
        scaler.fit_transform(Xtr)
        scaler.transform(Xte)
    r_leaky, r_clean = leaky.result(), clean.result()
    print(f"[L2] leaky run: {len(r_leaky.findings)} findings | clean run: {len(r_clean.findings)} findings")

    exact = find_cross_split_duplicates(Xtr, Xte)
    near = []
    for eps in (0.01, 0.05, 0.10, 0.20):
        found = find_cross_split_duplicates(Xtr, Xte, mode="near", eps=eps)
        near.append(f"eps={eps}: {found[0].evidence.split(' lie')[0] if found else 'none'}")
    print(f"[L2] REAL AER exact duplicates across split: {len(exact)} | near-duplicates by eps: " + "; ".join(near))

    dup = X.sample(60, random_state=1)  # injected duplicates, with their true labels
    with ProvenanceTracker() as injected:
        tracked_train_test_split(
            pd.concat([X, dup], ignore_index=True), np.concatenate([y, y[dup.index]]), test_size=0.25, random_state=0
        )
    print(f"[L2] INJECTED 60 duplicated rows: {summarize(injected.result().findings)}")
    return r_leaky, r_clean


def make_store(df, event_ts, backfill_flaw, seed=0):
    """SYNTHETIC: before approval spend is an income-driven baseline; after, it equals AER expenditure."""
    noise = np.random.default_rng(seed).normal(0, 15, size=(len(df), len(MONTHS)))
    value = np.where(MONTHS < 0, df.income.to_numpy()[:, None] * 20, df.expenditure.to_numpy()[:, None]) + noise
    computed_at = event_ts.to_numpy()[:, None] + pd.to_timedelta(30 * MONTHS - 1, unit="D").to_numpy()
    lag = pd.to_timedelta(np.where(backfill_flaw & (MONTHS >= 5), 40, 0), unit="D").to_numpy()
    return pd.DataFrame(
        {
            "entity_id": np.repeat(df.rownames.to_numpy(), len(MONTHS)),
            "feature_name": FEATURE,
            "value": value.ravel(),
            "computed_at": computed_at.ravel(),
            "valid_from": computed_at.ravel(),
            "max_source_timestamp": (computed_at + lag).ravel(),
        }
    )


def asof_join(labels, recs):
    m = pd.merge_asof(
        labels.sort_values("event_timestamp"),
        recs.sort_values("computed_at"),
        left_on="event_timestamp",
        right_on="computed_at",
        by="entity_id",
        allow_exact_matches=False,
        tolerance=pd.Timedelta("45D"),
    )
    return m[["entity_id", "event_timestamp", "value"]].rename(columns={"value": FEATURE})


def run_layer3_synthetic(df, y):
    rng = np.random.default_rng(0)
    event_ts = pd.Series(pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 730, len(df)), unit="D"))
    labels = pd.DataFrame({"entity_id": df.rownames.to_numpy(), "event_timestamp": event_ts, "label": y})
    recs_leaky, recs_clean = make_store(df, event_ts, True), make_store(df, event_ts, False)
    latest = recs_leaky.sort_values("computed_at").groupby("entity_id").value.last()
    ts_naive = labels[["entity_id", "event_timestamp"]].assign(**{FEATURE: labels.entity_id.map(latest).to_numpy()})
    ts_clean = asof_join(labels, recs_clean)
    offline = ts_clean.rename(columns={FEATURE: "value"}).assign(feature_name=FEATURE)
    online_bad = offline.copy()
    online_bad.loc[online_bad.sample(frac=0.08, random_state=1).index, "value"] *= 1.1
    r_naive = run_layer3(labels, recs_leaky, ts_naive, "45D", offline=offline, online=online_bad)
    r_clean = run_layer3(labels, recs_clean, ts_clean, "45D", offline=offline, online=offline.copy())
    r_flawed_store = run_layer3(labels, recs_leaky, asof_join(labels, recs_leaky), "45D")
    print(f"[L3 SYNTHETIC] naive 'latest' join + flawed store + skewed online: {summarize(r_naive.findings)}")
    print(f"[L3 SYNTHETIC] correct as-of join on clean store (control):        {summarize(r_clean.findings)}")
    print(f"[L3 SYNTHETIC] correct as-of join on store WITH backfill flaw:     {summarize(r_flawed_store.findings)}")
    synthetic_auc = {
        "naive latest-join feature (SYNTHETIC)": rf_auc(ts_naive[[FEATURE]], y, 0),
        "correct as-of feature (SYNTHETIC)": rf_auc(ts_clean[[FEATURE]].fillna(0), y, 0),
    }
    print(
        "[IMPACT, SYNTHETIC by construction] single-feature RF AUC:", {k: round(v, 3) for k, v in synthetic_auc.items()}
    )
    return r_naive, r_clean


def holdout_accuracy(X, y, seed, fit_before_split):
    """Accuracy of scaler + SelectKBest(k=3) + LogReg, with the preprocessing fit before or after the split."""
    if fit_before_split:
        Z = SelectKBest(f_classif, k=3).fit_transform(StandardScaler().fit_transform(X), y)
        a, b, ya, yb = train_test_split(Z, y, test_size=0.25, random_state=seed, stratify=y)
    else:
        a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(a)
        selector = SelectKBest(f_classif, k=3).fit(scaler.transform(a), ya)
        a, b = selector.transform(scaler.transform(a)), selector.transform(scaler.transform(b))
    return LogisticRegression(max_iter=2000).fit(a, ya).score(b, yb)


def measure_impact(X, y):
    auc = {
        "all features (incl. expenditure, share)": [rf_auc(X, y, s) for s in range(10)],
        "without expenditure, share": [rf_auc(X.drop(columns=LEAK_COLS), y, s) for s in range(10)],
    }
    gap = {}
    for label, cols in (
        ("with expenditure/share", list(X.columns)),
        ("without expenditure/share", [c for c in X.columns if c not in LEAK_COLS]),
    ):
        leaky = np.array([holdout_accuracy(X[cols], y, s, True) for s in range(30)])
        clean = np.array([holdout_accuracy(X[cols], y, s, False) for s in range(30)])
        gap[label] = (leaky.mean(), clean.mean(), (leaky - clean).mean(), (leaky - clean).std() / np.sqrt(len(leaky)))
    print("\n[IMPACT] RF 5-fold AUC over 10 seeds:")
    for k, v in auc.items():
        print(f"   {k:45} {np.mean(v):.3f} ± {np.std(v):.3f}")
    print("[IMPACT] pipeline leak (scaler+SelectKBest k=3 before split) hold-out accuracy, 30 seeds, LogReg:")
    for k, (leaky_acc, clean_acc, diff, se) in gap.items():
        print(f"   {k:28} leaky {leaky_acc:.4f} clean {clean_acc:.4f}  diff {diff:+.4f} (SE {se:.4f})")
    return {
        "auc": {k: [float(np.mean(v)), float(np.std(v))] for k, v in auc.items()},
        "pipeline_leak_accuracy": {k: [float(x) for x in v] for k, v in gap.items()},
    }


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv("data/CreditCard.csv")
    X, y, _ = prepare_xy(df, "card", "yes", drop_cols=("rownames",))
    print(f"AER: n={len(df)}, features={list(X.columns)}, accept rate={y.mean():.3f}\n")

    r4 = run_layer4(df, "card", "yes", drop_cols=("rownames",))
    print(f"[L4] real AER: {len(r4.findings)} findings -> {sorted({f.feature_or_step for f in r4.findings})}")
    r1_leaky, r1_clean = analyze_file("demo/aer_leaky.py"), analyze_file("demo/aer_clean.py")
    print(f"[L1] leaky script: {len(r1_leaky.findings)} findings | clean script: {len(r1_clean.findings)} findings")
    r2_leaky, r2_clean = run_layer2(X, y)
    r3_naive, r3_clean = run_layer3_synthetic(df, y)
    impact = measure_impact(X, y)

    for name, results in (
        ("aer_real_data_leaky_pipeline", [r1_leaky, r2_leaky, r3_naive, r4]),
        ("aer_clean_pipeline_control", [r1_clean, r2_clean, r3_clean]),
    ):
        report = aggregate(results)
        (out / f"{name}.json").write_text(to_json(report))
        (out / f"{name}.md").write_text(to_markdown(report))
        print(
            f"\nsaved {name}: {report['summary']['findings']} findings (raw {report['raw_finding_count']}), "
            f"max sev {report['summary']['max_severity']}"
        )
    (out / "impact.json").write_text(json.dumps(impact, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report")
