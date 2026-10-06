"""Leak-free equivalent: split first; scaling and selection are fit inside a Pipeline on the training rows only."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/pd_speech.csv")
y = (df["class"] == "PD").astype(int)
X = df.drop(columns=["rownames", "class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
