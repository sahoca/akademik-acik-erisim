import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

BASE = Path(__file__).parent
CSV_DIR = BASE / "data" / "CSV"
OUT_DIR = BASE / "out"
OUT_DIR.mkdir(exist_ok=True)

COLS = ["generated", "appName", "totalSourceBytes", "totalDestinationBytes",
        "totalDestinationPackets", "totalSourcePackets", "sourcePayloadAsBase64",
        "sourcePayloadAsUTF", "destinationPayloadAsBase64", "destinationPayloadAsUTF",
        "direction", "sourceTCPFlagsDescription", "destinationTCPFlagsDescription",
        "source", "protocolName", "sourcePort", "destination", "destinationPort",
        "startDateTime", "stopDateTime", "Label"]

DROP_ALWAYS = [
    "generated",
    "source", "destination",
    "sourcePayloadAsBase64", "sourcePayloadAsUTF",
    "destinationPayloadAsBase64", "destinationPayloadAsUTF",
]


def process_time_of_day(t):
    try:
        hm = t.split(" ")[-1]
        h, m = hm.split(":")
        return 3600 * int(h) + 60 * int(m)
    except Exception:
        return np.nan


def load_all_days():
    frames = []
    files = sorted(glob.glob(str(CSV_DIR / "*.csv")))
    for f in files:
        df = pd.read_csv(f, names=COLS, skiprows=1, low_memory=False)
        df["source_day"] = Path(f).stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_features(df):
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns])

    df["startTimeOfDay"] = df["startDateTime"].apply(process_time_of_day)
    df["stopTimeOfDay"] = df["stopDateTime"].apply(process_time_of_day)
    df = df.drop(columns=["startDateTime", "stopDateTime"])

    label_bin = (df["Label"].astype(str).str.strip().str.lower() != "normal").astype(int)

    cat_cols = ["appName", "direction", "sourceTCPFlagsDescription",
                "destinationTCPFlagsDescription", "protocolName"]
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
    return X, y, feature_cols


def evaluate(name, clf, X_test, y_test):
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else pred
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    confusion_matrix(y_test, pred)
    return metrics


def main():
    t0 = time.time()
    df = load_all_days()
    X, y, feature_cols = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

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

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "no_leak_model_comparison.csv", index=False)

    best_row = res_df.sort_values("roc_auc", ascending=False).iloc[0]
    best_clf = {"DecisionTree": dtree, "RandomForest": rforest, "XGBoost": xgb_clf}[best_row["model"]]

    X.to_parquet(OUT_DIR / "iscx2012_features_full.parquet")
    X_test.to_parquet(OUT_DIR / "iscx2012_features_test.parquet")
    y_test.to_frame("label").to_parquet(OUT_DIR / "iscx2012_labels_test.parquet")

    import joblib
    joblib.dump(best_clf, OUT_DIR / f"best_model_{best_row['model']}.pkl")
    with open(OUT_DIR / "feature_columns.txt", "w") as f:
        f.write("\n".join(feature_cols))

    print(f"total_time_s={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
