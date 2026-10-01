import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
import xgboost as xgb

from train_and_shap import load_all_days, process_time_of_day, OUT_DIR

df = load_all_days()

label_bin = (df["Label"].astype(str).str.strip().str.lower() != "normal").astype(int)

df["startTimeOfDay"] = df["startDateTime"].apply(process_time_of_day)
df["stopTimeOfDay"] = df["stopDateTime"].apply(process_time_of_day)

cat_cols = ["source", "destination", "appName", "direction",
            "sourceTCPFlagsDescription", "destinationTCPFlagsDescription", "protocolName"]
for c in cat_cols:
    df[c] = df[c].fillna("NA").astype(str)
    df[c] = LabelEncoder().fit_transform(df[c])

num_cols = ["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets",
            "totalSourcePackets", "sourcePort", "destinationPort",
            "startTimeOfDay", "stopTimeOfDay"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

feature_cols = cat_cols + num_cols
X = df[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))
y = label_bin

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


def evaluate(name, clf, X_te, y_te):
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred, zero_division=0),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, proba),
    }


results = []

dtree = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced")
dtree.fit(X_train, y_train)
results.append(evaluate("DecisionTree", dtree, X_test, y_test))

rforest = RandomForestClassifier(
    n_estimators=200, max_depth=15, n_jobs=-1, random_state=42, class_weight="balanced"
)
rforest.fit(X_train, y_train)
results.append(evaluate("RandomForest", rforest, X_test, y_test))

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42,
    eval_metric="logloss",
)
xgb_clf.fit(X_train, y_train)
results.append(evaluate("XGBoost", xgb_clf, X_test, y_test))

pd.DataFrame(results).to_csv(OUT_DIR / "ip_included_model_comparison.csv", index=False)
