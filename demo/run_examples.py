"""Three example datasets other than AER, each exercising a different part of the tool.

Titanic (seaborn copy)  Layer 4 on a raw CSV with NaNs/strings; `alive` is a literal copy of the target.
Parkinson's speech      Layer 1 + a measurement of what selection-before-split really costs when p >> n.
Repeated measures       Entity leakage: random vs entity-grouped folds (synthetic positive control, 3 real
datasets, shuffled-id false-positive control).
Stocks (real prices)    Layer 3 on real timestamps; the leak is a look-ahead (forward) as-of join BUG that I inject.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_aer import rf_auc

from leakdetect.aggregator import aggregate, to_markdown
from leakdetect.layer1_static import analyze_file
from leakdetect.layer3_pit import run_layer3
from leakdetect.layer4_stats import group_leakage, prepare_xy, run_layer4

warnings.filterwarnings("ignore")


def show(title, findings):
    print(
        f"  {title}: "
        + (", ".join(f"{f.feature_or_step}:{f.leakage_type}(sev {f.severity})" for f in findings) or "no findings")
    )


def titanic():
    print("== Titanic (raw CSV, 891 rows, NaNs + strings) ==")
    df = pd.read_csv("data/titanic_seaborn.csv")
    res = run_layer4(df, "survived", 1)
    show("Layer 4", sorted(res.findings, key=lambda f: -f.severity))
    X, y, _ = prepare_xy(df, "survived", 1)
    clean = X.drop(columns="alive_yes")
    print(
        f"  RF 5-fold AUC with `alive`: {np.mean([rf_auc(X, y, s) for s in range(5)]):.3f}   "
        f"without: {np.mean([rf_auc(clean, y, s) for s in range(5)]):.3f}"
    )
    print(
        f"  next-best single feature (not flagged): {res.extras['single_feature'].iloc[1].feature} "
        f"AUC {res.extras['single_feature'].iloc[1].auc:.3f}"
    )
    return res


def selection_inflation(df, shuffled, seeds=range(100), k=20):
    """Balanced accuracy of scaler+SelectKBest+LogReg with selection fit before vs after the split."""
    X, y = df.drop(columns=["rownames", "class"]), (df["class"] == "PD").astype(int).to_numpy()
    if shuffled:
        y = np.random.default_rng(0).permutation(y)
    leaky, clean = [], []
    for seed in seeds:
        sel = SelectKBest(f_classif, k=k).fit_transform(X, y)  # sees every row, including the future test rows
        a, b, ya, yb = train_test_split(sel, y, test_size=0.25, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(a)
        leaky.append(
            balanced_accuracy_score(
                yb, LogisticRegression(max_iter=2000).fit(scaler.transform(a), ya).predict(scaler.transform(b))
            )
        )
        a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(max_iter=2000))
        clean.append(balanced_accuracy_score(yb, pipe.fit(a, ya).predict(b)))
    leaky, clean = np.array(leaky), np.array(clean)
    return leaky.mean(), clean.mean(), (leaky - clean).mean(), (leaky - clean).std() / np.sqrt(len(leaky))


def parkinsons():
    print("\n== Parkinson's speech (252 rows x 750 features; p >> n) ==")
    df = pd.read_csv("data/pd_speech.csv")
    l4 = run_layer4(df, "class", "PD", drop_cols=("rownames",))
    show("Layer 4", l4.findings)
    l1_leaky, l1_clean = analyze_file("demo/pd_speech_leaky.py"), analyze_file("demo/pd_speech_clean.py")
    show("Layer 1 on pd_speech_leaky.py", l1_leaky.findings)
    show("Layer 1 on pd_speech_clean.py", l1_clean.findings)
    print("  measured effect of the flagged bug (balanced accuracy, 100 stratified splits, SelectKBest k=20):")
    for label, shuffled in (("real labels", False), ("SHUFFLED labels (true signal = none, expect 0.50)", True)):
        lk, cl, diff, se = selection_inflation(df, shuffled)
        print(f"    {label:50} leaky {lk:.3f}  clean {cl:.3f}  inflation {diff:+.3f} (SE {se:.3f})")
    return l4, l1_leaky


def stocks():
    print("\n== Stocks: real monthly prices 2000-2010 (5 symbols); look-ahead join bug injected ==")
    s = pd.read_csv("data/stocks.csv", parse_dates=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    g = s.groupby("symbol").price
    s["ret_1m"], s["mom_3m"] = g.pct_change(1), g.pct_change(3)
    s["label"] = (g.shift(-1) > s.price).astype(int)
    s = s[s.mom_3m.notna() & g.shift(-1).notna()].reset_index(drop=True)  # rows where features and label both exist
    recs = pd.concat(
        [s[["symbol", "date", f]].rename(columns={f: "value"}).assign(feature_name=f) for f in ("ret_1m", "mom_3m")]
    ).rename(columns={"symbol": "entity_id", "date": "computed_at"})
    labels = (
        s[["symbol", "date", "label"]]
        .rename(columns={"symbol": "entity_id"})
        .assign(event_timestamp=s.date + pd.Timedelta(days=1))[["entity_id", "event_timestamp", "label"]]
    )

    def join(direction):
        out = labels[["entity_id", "event_timestamp"]]
        for f in ("ret_1m", "mom_3m"):
            r = recs[recs.feature_name == f][["entity_id", "value", "computed_at"]].sort_values("computed_at")
            m = pd.merge_asof(
                labels.sort_values("event_timestamp"),
                r,
                left_on="event_timestamp",
                right_on="computed_at",
                by="entity_id",
                direction=direction,
                allow_exact_matches=False,
            )
            out = out.merge(
                m[["entity_id", "event_timestamp", "value"]].rename(columns={"value": f}),
                on=["entity_id", "event_timestamp"],
                how="left",
            )
        return out

    ok, bad = join("backward"), join("forward")
    r_ok = run_layer3(labels, recs, ok, "45D")
    r_bad = run_layer3(labels, recs, bad.dropna(), "45D")
    show("correct backward as-of join", r_ok.findings)
    show("buggy forward as-of join   ", r_bad.findings)
    for name, ts in (("correct", ok), ("buggy  ", bad.dropna())):
        y = labels.merge(ts, on=["entity_id", "event_timestamp"]).label
        auc = cross_val_score(LogisticRegression(), ts[["ret_1m", "mom_3m"]], y, cv=5, scoring="roc_auc").mean()
        print(f"  {name} join -> 5-fold LogReg AUC predicting next-month direction: {auc:.3f}")
    return r_ok, r_bad


def entity_leakage():
    print("\n== Entity leakage: AUC under random vs entity-grouped folds (5 repeats) ==")

    def show(name, X, y, groups, repeats=5):
        found, info = group_leakage(X, y, groups, repeats=repeats)
        verdict = f"FLAGGED (sev {found[0].severity})" if found else "not flagged"
        print(
            f"  {name:50} overlap {info['same_group_share']:4.0%}  random {info['random_auc']:.3f}  "
            f"grouped {info['grouped_auc']:.3f}  gap {info['gap']:+.3f}  {verdict}"
        )
        return bool(found)

    rng = np.random.default_rng(0)
    subject = np.repeat(np.arange(60), 6)
    label = rng.permutation(np.arange(60) % 2)[subject]  # constant within a subject, unrelated to anything
    signature = pd.DataFrame(rng.normal(size=(60, 5))[subject] + rng.normal(scale=0.3, size=(360, 5)))
    show("SYNTHETIC positive control (memorisable subjects)", signature, label, subject)
    chick = None
    for name, path, target, positive, group, drop in (
        ("Orthodont: Sex ~ distance, age", "data/Orthodont.csv", "Sex", "Male", "Subject", ("rownames",)),
        ("ChickWeight: Diet==1 ~ weight, time", "data/ChickWeight.csv", "Diet", 1, "Chick", ("rownames",)),
        ("VerbAgg: response ~ person + item covariates", "data/VerbAgg.csv", "r2", "Y", "id", ("rownames", "resp")),
    ):
        df = pd.read_csv(path)
        X, y, _ = prepare_xy(df, target, positive, drop_cols=(*drop, group))
        show(name, X, y, df[group].to_numpy())
        if group == "Chick":
            chick = (X, y, df[group].to_numpy())
    X, y, ids = chick
    flagged = sum(
        show(f"ChickWeight, shuffled ids (control {i + 1}/10)", X, y, rng.permutation(ids), repeats=3)
        for i in range(10)
    )
    print(f"  shuffled-id controls flagged: {flagged}/10 (expect 0)")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "report/examples")
    out.mkdir(parents=True, exist_ok=True)
    t = titanic()
    p4, p1 = parkinsons()
    entity_leakage()
    _, bad = stocks()
    (out / "examples.md").write_text(to_markdown(aggregate([t, p4, p1, bad])))
