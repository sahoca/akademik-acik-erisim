import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
import xgboost as xgb

from train_and_shap import load_all_days, build_features, OUT_DIR

df = load_all_days()
X, y, feature_cols = build_features(df)
X["source_day"] = df["source_day"].values

DAY_ORDER = [
    "TestbedSatJun12Flows",
    "TestbedSunJun13Flows",
    "TestbedMonJun14Flows",
    "TestbedTueJun15Flows",
    "TestbedWedJun16Flows",
    "TestbedThuJun17Flows",
]

train_days = DAY_ORDER[:4]
test_days = DAY_ORDER[4:]

train_mask = X["source_day"].isin(train_days)
test_mask = X["source_day"].isin(test_days)

X_train = X.loc[train_mask, feature_cols]
y_train = y.loc[train_mask]
X_test = X.loc[test_mask, feature_cols]
y_test = y.loc[test_mask]

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42,
    eval_metric="logloss",
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]

row = {
    "split": "time_based_4train_2test",
    "accuracy": accuracy_score(y_test, pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
    "f1": f1_score(y_test, pred, zero_division=0),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba),
}

pd.DataFrame([row]).to_csv(OUT_DIR / "time_based_split_xgboost.csv", index=False)
