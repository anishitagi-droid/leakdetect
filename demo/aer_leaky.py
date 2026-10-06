"""Deliberately leaky preprocessing on AER Credit Card Applications (for Layer 1 demo)."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/CreditCard.csv")
y = (df.card == "yes").astype(int)
X = pd.get_dummies(df.drop(columns=["rownames", "card"]), drop_first=True, dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # LEAK: scaler fit on all rows
sel = SelectKBest(f_classif, k=5)
X_sel = sel.fit_transform(X_scaled, y)  # LEAK: supervised selection before the split
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.25, random_state=0, stratify=y)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(model.score(X_test, y_test))
