"""Parkinson's speech (252 x 750): feature selection fit on ALL rows before the split (Layer 1 demo)."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/pd_speech.csv")
y = (df["class"] == "PD").astype(int)
X = df.drop(columns=["rownames", "class"])

X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)  # LEAK: supervised selection saw the test rows
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression(max_iter=2000).fit(scaler.transform(X_train), y_train)
print(model.score(scaler.transform(X_test), y_test))
