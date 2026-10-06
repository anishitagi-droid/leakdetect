import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from leakdetect.aggregator import aggregate, to_markdown
from leakdetect.layer1_static import analyze_source
from leakdetect.layer2_runtime import ProvenanceTracker, track_provenance, tracked_train_test_split
from leakdetect.schema import Finding, LayerResult


LEAKY_SCRIPT = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = pd.read_csv('a')
y = X.pop('t')
Xs = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(Xs, y)
"""


def test_aggregator_dedups_layer1_and_layer2_same_step_and_sorts():
    l1 = analyze_source(LEAKY_SCRIPT)
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)))
    with ProvenanceTracker() as t:
        Xs = track_provenance(StandardScaler()).fit_transform(X)
        tracked_train_test_split(Xs, np.zeros(200), random_state=0)
    rep = aggregate([l1, t.result(), LayerResult.skipped(3, "no feature store")])
    assert rep["raw_finding_count"] == 2 and rep["summary"]["findings"] == 1
    assert rep["findings"][0]["flagged_by_layers"] == [1, 2]
    assert rep["summary"]["layers_skipped"] == {3: "no feature store"}
    md = to_markdown(rep)
    assert "NOT run" in md and "Layer 3" in md


def test_aggregator_severity_order():
    rs = LayerResult(
        4,
        "ok",
        findings=[Finding(4, "a", "pit-violation", 2, "e", "f"), Finding(4, "b", "backfill-leakage", 5, "e", "f")],
    )
    assert [f["feature_or_step"] for f in aggregate([rs])["findings"]] == ["b", "a"]


def test_aggregator_merged_finding_keeps_the_highest_severity_in_either_order():
    low = Finding(2, "StandardScaler#0", "fit-contamination", 2, "runtime", "fix")
    high = Finding(1, "StandardScaler@L7", "fit-before-split", 5, "static", "fix")
    for order in ((low, high), (high, low)):
        report = aggregate([LayerResult(f.layer, "ok", findings=[f]) for f in order])
        assert report["summary"]["findings"] == 1 and report["findings"][0]["severity"] == 5
