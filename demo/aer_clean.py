"""Leak-free equivalent: split first, fit everything on train only (Layer 1 clean demo)."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/CreditCard.csv")
y = (df.card == "yes").astype(int)
X = pd.get_dummies(df.drop(columns=["rownames", "card"]), drop_first=True, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
sel = SelectKBest(f_classif, k=5)
X_train_f = sel.fit_transform(X_train_s, y_train)
X_test_f = sel.transform(X_test_s)
model = LogisticRegression(max_iter=1000).fit(X_train_f, y_train)
print(model.score(X_test_f, y_test))
