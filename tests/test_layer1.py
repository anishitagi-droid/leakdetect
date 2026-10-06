from leakdetect.layer1_static import analyze_source, analyze_steps, analyze_pipeline

HEAD = (
    "import pandas as pd\nfrom sklearn.model_selection import *\nfrom sklearn.preprocessing import *\n"
    "from sklearn.feature_selection import *\nX=pd.read_csv('a.csv'); y=X.pop('t')\n"
)


def types(src):
    r = analyze_source(HEAD + src)
    assert r.status == "ok", r.reason
    return sorted({f.leakage_type for f in r.findings}), r


def test_fit_transform_before_split_flagged():
    t, r = types("sc=StandardScaler()\nXs=sc.fit_transform(X)\nXtr,Xte,ytr,yte=train_test_split(Xs,y)\n")
    assert t == ["fit-before-split"]
    assert "line 7" in r.findings[0].evidence


def test_separate_fit_then_transform_then_split_flagged():
    t, _ = types("sc=StandardScaler()\nsc.fit(X)\nXs=sc.transform(X)\nXtr,Xte,ytr,yte=train_test_split(Xs,y)\n")
    assert t == ["fit-before-split"]


def test_fit_on_whole_then_split_raw_then_transform_parts_flagged():
    t, _ = types("sc=StandardScaler().fit(X)\nXtr,Xte,ytr,yte=train_test_split(X,y)\nA=sc.transform(Xtr)\n")
    assert t == ["fit-before-split"]


def test_clean_split_first_not_flagged():
    t, _ = types(
        "Xtr,Xte,ytr,yte=train_test_split(X,y)\nsc=StandardScaler()\nA=sc.fit_transform(Xtr)\nB=sc.transform(Xte)\n"
    )
    assert t == []


def test_selection_before_split_is_flagged_and_supervised():
    t, r = types("sel=SelectKBest(k=3)\nXs=sel.fit_transform(X,y)\nXtr,Xte,ytr,yte=train_test_split(Xs,y)\n")
    assert t == ["feature-selection-before-split"] and r.findings[0].severity >= 4


def test_target_encoding_before_split():
    src = (
        "from category_encoders import TargetEncoder\nte=TargetEncoder()\n"
        "Xe=te.fit_transform(X,y)\nXtr,Xte,ytr,yte=train_test_split(Xe,y)\n"
    )
    t, _ = types(src)
    assert t == ["target-encoding-before-split"]


def test_oversampling_before_split_severity5():
    t, r = types(
        "from imblearn.over_sampling import SMOTE\nXr,yr=SMOTE().fit_resample(X,y)\n"
        "Xtr,Xte,ytr,yte=train_test_split(Xr,yr)\n"
    )
    assert t == ["oversampling-before-split"] and r.findings[0].severity == 5


def test_oversampling_after_split_on_train_only_clean():
    t, _ = types(
        "from imblearn.over_sampling import SMOTE\nXtr,Xte,ytr,yte=train_test_split(X,y)\n"
        "Xr,yr=SMOTE().fit_resample(Xtr,ytr)\n"
    )
    assert t == []


def test_cv_preprocessing_outside_folds_flagged():
    t, _ = types(
        "Xs=StandardScaler().fit_transform(X)\nfrom sklearn.linear_model import LogisticRegression\n"
        "cross_val_score(LogisticRegression(),Xs,y)\n"
    )
    assert "cv-preprocessing-leakage" in t


def test_cv_with_pipeline_on_raw_data_clean():
    t, _ = types(
        "from sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import LogisticRegression\n"
        "p=make_pipeline(StandardScaler(),LogisticRegression())\ncross_val_score(p,X,y)\n"
    )
    assert t == []


def test_cv_loop_fit_on_full_data_flagged():
    t, _ = types("kf=KFold(5)\nfor tr,te in kf.split(X):\n    sc=StandardScaler()\n    A=sc.fit_transform(X)\n")
    assert "cv-loop-leakage" in t


def test_cv_loop_fit_on_fold_train_clean():
    t, _ = types(
        "kf=KFold(5)\nfor tr,te in kf.split(X):\n    sc=StandardScaler()\n    A=sc.fit_transform(X.iloc[tr])\n"
        "    B=sc.transform(X.iloc[te])\n"
    )
    assert t == []


def test_scaler_before_cv_loop_flagged():
    t, _ = types("Xs=StandardScaler().fit_transform(X)\nkf=KFold(5)\nfor tr,te in kf.split(Xs):\n    pass\n")
    assert "cv-preprocessing-leakage" in t


def test_no_split_is_skipped_not_clean():
    r = analyze_source("from sklearn.preprocessing import StandardScaler\nStandardScaler().fit_transform([[1]])\n")
    assert r.status == "skipped" and "no train_test_split" in r.reason


def test_syntax_error_skipped():
    assert analyze_source("def (:").status == "skipped"


def test_steps_leaky_and_clean():
    leaky = [("scale", None, "fit_transform"), ("split", None, "split"), ("model", None, "transform")]
    assert [f.leakage_type for f in analyze_steps(leaky).findings] == ["fit-before-split"]
    clean = [("split", None, "split"), ("scale", None, "fit_transform")]
    assert analyze_steps(clean).findings == []
    assert analyze_steps([("scale", None, "fit_transform")]).status == "skipped"


def test_sklearn_pipeline_assumption_is_documented_and_clean():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest

    r = analyze_pipeline(make_pipeline(StandardScaler(), SelectKBest(k=2)))
    assert r.status == "ok" and r.findings == [] and any("Assumed" in n for n in r.notes)


# ---- regression tests for false negatives found by adversarial probing
def test_wrapped_in_dataframe_flagged():
    t, _ = types(
        "sc=StandardScaler()\nXs=pd.DataFrame(sc.fit_transform(X),columns=X.columns)\n"
        "Xtr,Xte,ytr,yte=train_test_split(Xs,y)\n"
    )
    assert t == ["fit-before-split"]


def test_helper_function_hides_fit_flagged():
    t, _ = types(
        "def prep(df):\n    return StandardScaler().fit_transform(df)\nXs=prep(X)\n"
        "Xtr,Xte,ytr,yte=train_test_split(Xs,y)\n"
    )
    assert t == ["fit-before-split"]


def test_helper_function_that_splits_first_clean():
    t, _ = types(
        "def prep(df,y):\n    a,b,c,d=train_test_split(df,y)\n    sc=StandardScaler()\n"
        "    return sc.fit_transform(a),sc.transform(b)\nz=prep(X,y)\n"
    )
    assert t == []


def test_pandas_zscore_before_split_flagged():
    t, r = types("Xs=(X-X.mean())/X.std()\nXtr,Xte,ytr,yte=train_test_split(Xs,y)\n")
    assert t == ["fit-before-split"] and "Heuristic" in r.findings[0].evidence


def test_mean_only_printed_not_flagged():
    t, _ = types("print(X.mean())\nXtr,Xte,ytr,yte=train_test_split(X,y)\n")
    assert t == []


def test_train_only_statistics_clean():
    t, _ = types("Xtr,Xte,ytr,yte=train_test_split(X,y)\nm=Xtr.mean()\nA=(Xtr-m);B=(Xte-m)\n")
    assert t == []


def test_fit_on_test_partition_flagged():
    t, _ = types("Xtr,Xte,ytr,yte=train_test_split(X,y)\nsc=StandardScaler()\nA=sc.fit_transform(Xte)\n")
    assert t == ["fit-on-test-data"]


# ---- regression tests: leaks inside class methods used to be reported as "skipped: no split recognised"
def test_leak_inside_class_method_flagged():
    t, _ = types(
        "class P:\n    def run(self, X, y):\n        Xs = StandardScaler().fit_transform(X)\n"
        "        return train_test_split(Xs, y)\nP().run(X, y)\n"
    )
    assert t == ["fit-before-split"]


def test_clean_class_method_not_flagged():
    t, _ = types(
        "class P:\n    def run(self, X, y):\n        a, b, c, d = train_test_split(X, y)\n"
        "        sc = StandardScaler()\n        return sc.fit_transform(a), sc.transform(b)\np = P()\np.run(X, y)\n"
    )
    assert t == []


def test_unreferenced_function_is_not_analysed_and_says_so():
    r = analyze_source(HEAD + "def unused(X, y):\n    return train_test_split(StandardScaler().fit_transform(X), y)\n")
    assert r.status == "skipped" and "top-level" in r.reason
