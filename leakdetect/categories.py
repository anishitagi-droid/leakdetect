"""Estimator-name -> leakage category, shared by Layer 1 (static) and Layer 2 (runtime)."""

PREPROC = {
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "PCA",
    "TruncatedSVD",
    "KernelPCA",
    "PowerTransformer",
    "QuantileTransformer",
    "KBinsDiscretizer",
    "TfidfVectorizer",
    "CountVectorizer",
    "VarianceThreshold",
}
SELECTION = {
    "SelectKBest",
    "SelectPercentile",
    "SelectFromModel",
    "RFE",
    "RFECV",
    "SequentialFeatureSelector",
    "GenericUnivariateSelect",
    "SelectFpr",
    "SelectFdr",
    "SelectFwe",
}
TARGET_ENC = {
    "TargetEncoder",
    "CatBoostEncoder",
    "LeaveOneOutEncoder",
    "WOEEncoder",
    "MEstimateEncoder",
    "JamesSteinEncoder",
    "GLMMEncoder",
}
OVERSAMPLE = {
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE",
    "SMOTENC",
    "SVMSMOTE",
    "KMeansSMOTE",
    "RandomOverSampler",
    "RandomUnderSampler",
}
# estimators that cross-validate internally; RFECV is a selector and lives in SELECTION only
SEARCH_CV = {
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "LogisticRegressionCV",
    "RidgeCV",
    "LassoCV",
    "ElasticNetCV",
}
CV_SPLITTERS = {
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "StratifiedGroupKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "TimeSeriesSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "LeaveOneOut",
    "LeavePOut",
    "LeaveOneGroupOut",
}
CV_FUNCS = {"cross_val_score", "cross_val_predict", "cross_validate"}

SEVERITY = {"preproc": 3, "selection": 4, "target_encoding": 4, "oversample": 5}
LTYPE = {
    "preproc": "fit-before-split",
    "selection": "feature-selection-before-split",
    "target_encoding": "target-encoding-before-split",
    "oversample": "oversampling-before-split",
}

_CATEGORY = {
    name: cat
    for cat, names in (
        ("preproc", PREPROC),
        ("selection", SELECTION),
        ("target_encoding", TARGET_ENC),
        ("oversample", OVERSAMPLE),
    )
    for name in names
}


def category_of(cls_name):
    return _CATEGORY.get(cls_name)
