import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
import xgboost as xgb

from train_and_shap import load_all_days, build_features, OUT_DIR

df = load_all_days()
X, y, feature_cols = build_features(df)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    rows.append({
        "fold": fold,
        "accuracy": accuracy_score(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred, zero_division=0),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, proba),
    })

df_res = pd.DataFrame(rows)
summary = df_res.drop(columns="fold").agg(["mean", "std"])

df_res.to_csv(OUT_DIR / "cv_5fold_xgboost.csv", index=False)
summary.to_csv(OUT_DIR / "cv_5fold_xgboost_summary.csv")
