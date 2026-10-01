import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
import xgboost as xgb

from train_and_shap import load_all_days, build_features, OUT_DIR

df = load_all_days()
X, y, feature_cols = build_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


def eval_model(name, clf, X_te, y_te):
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


scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_full = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42,
    eval_metric="logloss",
)
xgb_full.fit(X_train, y_train)
res_full = eval_model("XGBoost_full_13feat", xgb_full, X_test, y_test)

drop_cols = ["startTimeOfDay", "stopTimeOfDay"]
X_train_ab = X_train.drop(columns=drop_cols)
X_test_ab = X_test.drop(columns=drop_cols)

xgb_ab = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42,
    eval_metric="logloss",
)
xgb_ab.fit(X_train_ab, y_train)
res_ab = eval_model("XGBoost_no_timeofday_11feat", xgb_ab, X_test_ab, y_test)

pd.DataFrame([res_full, res_ab]).to_csv(OUT_DIR / "ablation_timeofday.csv", index=False)
