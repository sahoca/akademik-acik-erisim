"""ISCXIDS2012 revizyon deneyleri (CDEJ 2049182, Hakem-2 madde 1 ve 2).

A1) Akis duzeyi rastgele bolunme vs oturum duzeyi (5-tuple GroupShuffleSplit) bolunme
A2) Zaman ozelligi ablasyonu ve kronolojik ayrim: 5 tohum, ileri zincirleme splitler,
    bootstrap guven araligi, McNemar ve eslestirilmis t-testi

On isleme ve hiperparametreler makaledeki train_and_shap.py ile birebir aynidir.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

CSV_DIR = Path(os.environ.get("ISCX2012_CSV_DIR", "data/iscx2012/CSV"))
OUT_DIR = Path(__file__).parent / os.environ.get("RESULTS_DIR", "results")
OUT_DIR.mkdir(exist_ok=True)
SEEDS = [0, 42, 123, 1024, 2024]

COLS = ["generated", "appName", "totalSourceBytes", "totalDestinationBytes",
        "totalDestinationPackets", "totalSourcePackets", "sourcePayloadAsBase64",
        "sourcePayloadAsUTF", "destinationPayloadAsBase64", "destinationPayloadAsUTF",
        "direction", "sourceTCPFlagsDescription", "destinationTCPFlagsDescription",
        "source", "protocolName", "sourcePort", "destination", "destinationPort",
        "startDateTime", "stopDateTime", "Label"]
DAY_ORDER = ["TestbedSatJun12Flows", "TestbedSunJun13Flows", "TestbedMonJun14Flows",
             "TestbedTueJun15Flows", "TestbedWedJun16Flows", "TestbedThuJun17Flows"]
CAT_COLS = ["appName", "direction", "sourceTCPFlagsDescription",
            "destinationTCPFlagsDescription", "protocolName"]
NUM_COLS = ["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets",
            "totalSourcePackets", "sourcePort", "destinationPort",
            "startTimeOfDay", "stopTimeOfDay"]
FEATURES = CAT_COLS + NUM_COLS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_time_of_day(t):
    try:
        hm = t.split(" ")[-1]
        h, m = hm.split(":")
        return 3600 * int(h) + 60 * int(m)
    except Exception:
        return np.nan


def load_all_days():
    frames = []
    for f in sorted(glob.glob(str(CSV_DIR / "*.csv"))):
        df = pd.read_csv(f, names=COLS, skiprows=1, low_memory=False)
        df["source_day"] = Path(f).stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_features(df):
    df = df.copy()
    df["startTimeOfDay"] = df["startDateTime"].apply(process_time_of_day)
    df["stopTimeOfDay"] = df["stopDateTime"].apply(process_time_of_day)
    y = (df["Label"].astype(str).str.strip().str.lower() != "normal").astype(int)
    for c in CAT_COLS:
        df[c] = LabelEncoder().fit_transform(df[c].fillna("NA").astype(str))
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def session_ids(df):
    key = (df["source"].astype(str) + "|" + df["destination"].astype(str) + "|"
           + df["sourcePort"].astype(str) + "|" + df["destinationPort"].astype(str)
           + "|" + df["protocolName"].astype(str))
    return pd.factorize(key)[0]


def make_xgb(y_train, seed):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=spw, n_jobs=-1, random_state=seed,
                             eval_metric="logloss")


def make_models(y_train, seed):
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=seed,
                                               class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1,
                                               random_state=seed, class_weight="balanced"),
        "XGBoost": make_xgb(y_train, seed),
    }


def metrics(y_true, pred, proba):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def fit_eval(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    return metrics(yte, pred, proba), pred, proba


def summarize(df, keys):
    g = df.groupby(keys)
    m = g[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]]
    out = m.agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()


def bootstrap_ci(y_true, pred, proba, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); pred = np.asarray(pred); proba = np.asarray(proba)
    n = len(y_true)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pr, pb = y_true[idx], pred[idx], proba[idx]
        if yt.min() == yt.max():
            continue
        rows.append({"f1": f1_score(yt, pr, zero_division=0),
                     "balanced_accuracy": balanced_accuracy_score(yt, pr),
                     "roc_auc": roc_auc_score(yt, pb)})
    b = pd.DataFrame(rows)
    return {k: (float(b[k].quantile(0.025)), float(b[k].quantile(0.975))) for k in b.columns}


def mcnemar(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true); a = np.asarray(pred_a) == y_true; b = np.asarray(pred_b) == y_true
    n01 = int(((a) & (~b)).sum())   # A dogru, B yanlis
    n10 = int(((~a) & (b)).sum())   # A yanlis, B dogru
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    p = float(stats.chi2.sf(chi2, df=1))
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p": p}


def main():
    t0 = time.time()
    log("Veri yukleniyor")
    df = load_all_days()
    X, y = build_features(df)
    groups = session_ids(df)
    days = df["source_day"].values
    log(f"Akis={len(X):,} saldiri={int(y.sum()):,} normal={int((y==0).sum()):,} "
        f"benzersiz 5-tuple oturum={len(np.unique(groups)):,}")
    summary = {"n_flows": int(len(X)), "n_attack": int(y.sum()), "n_normal": int((y == 0).sum()),
               "n_sessions": int(len(np.unique(groups)))}

    # ---------------- A1: akis vs oturum duzeyi bolunme ----------------
    log("A1 basliyor: akis duzeyi vs oturum duzeyi bolunme (3 model x 5 tohum x 2)")
    rows = []
    overlap = []
    for seed in SEEDS:
        # akis duzeyi (makaledeki yontem)
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=seed)
        shared = np.isin(groups[te], np.unique(groups[tr]))
        overlap.append({"seed": seed, "split": "flow",
                        "test_flows_with_session_in_train": float(shared.mean()),
                        "test_attack_flows_with_session_in_train":
                            float(shared[y.values[te] == 1].mean())})
        for name, clf in make_models(y.iloc[tr], seed).items():
            m, _, _ = fit_eval(clf, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
            rows.append({"split": "flow", "seed": seed, "model": name, **m})
            log(f"  flow seed={seed} {name} f1={m['f1']:.4f}")
        # oturum duzeyi
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        overlap.append({"seed": seed, "split": "group",
                        "test_flows_with_session_in_train":
                            float(np.isin(groups[te], np.unique(groups[tr])).mean()),
                        "test_attack_flows_with_session_in_train": 0.0,
                        "test_attack_share": float(y.iloc[te].mean())})
        for name, clf in make_models(y.iloc[tr], seed).items():
            m, _, _ = fit_eval(clf, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
            rows.append({"split": "group", "seed": seed, "model": name, **m})
            log(f"  group seed={seed} {name} f1={m['f1']:.4f}")
    a1 = pd.DataFrame(rows)
    a1.to_csv(OUT_DIR / "A1_flow_vs_group_runs.csv", index=False)
    summarize(a1, ["split", "model"]).to_csv(OUT_DIR / "A1_flow_vs_group_summary.csv", index=False)
    pd.DataFrame(overlap).to_csv(OUT_DIR / "A1_session_overlap.csv", index=False)
    # eslestirilmis test (ayni tohumlar) XGBoost
    xf = a1[(a1.split == "flow") & (a1.model == "XGBoost")].sort_values("seed")["f1"].values
    xg = a1[(a1.split == "group") & (a1.model == "XGBoost")].sort_values("seed")["f1"].values
    tt = stats.ttest_rel(xf, xg)
    summary["A1_xgb_f1_flow_mean"] = float(xf.mean()); summary["A1_xgb_f1_group_mean"] = float(xg.mean())
    summary["A1_xgb_paired_t"] = float(tt.statistic); summary["A1_xgb_paired_p"] = float(tt.pvalue)
    log(f"A1 bitti. XGB F1 flow={xf.mean():.4f} group={xg.mean():.4f} p={tt.pvalue:.4f}")

    # ---------------- A2a: ablasyon 5 tohum ----------------
    log("A2a basliyor: zaman ozelligi ablasyonu, 5 tohum")
    rows = []
    preds42 = {}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=seed)
        for variant, cols in [("full_13", FEATURES),
                              ("no_timeofday_11", [c for c in FEATURES if c not in ("startTimeOfDay", "stopTimeOfDay")])]:
            m, pred, proba = fit_eval(make_xgb(y.iloc[tr], seed), X.iloc[tr][cols], y.iloc[tr],
                                      X.iloc[te][cols], y.iloc[te])
            rows.append({"experiment": "ablation", "variant": variant, "seed": seed, **m})
            log(f"  ablation seed={seed} {variant} f1={m['f1']:.4f}")
            if seed == 42:
                preds42[variant] = (pred, proba, te)
    a2a = pd.DataFrame(rows)
    a2a.to_csv(OUT_DIR / "A2_ablation_runs.csv", index=False)
    summarize(a2a, ["variant"]).to_csv(OUT_DIR / "A2_ablation_summary.csv", index=False)
    f_full = a2a[a2a.variant == "full_13"].sort_values("seed")["f1"].values
    f_abl = a2a[a2a.variant == "no_timeofday_11"].sort_values("seed")["f1"].values
    tt = stats.ttest_rel(f_full, f_abl)
    te42 = preds42["full_13"][2]
    mc = mcnemar(y.iloc[te42], preds42["full_13"][0], preds42["no_timeofday_11"][0])
    summary["A2_ablation_paired_t"] = float(tt.statistic); summary["A2_ablation_paired_p"] = float(tt.pvalue)
    summary["A2_ablation_mcnemar_seed42"] = mc
    summary["A2_ablation_f1_full_ci95_seed42"] = bootstrap_ci(y.iloc[te42], *preds42["full_13"][:2])
    summary["A2_ablation_f1_ablated_ci95_seed42"] = bootstrap_ci(y.iloc[te42], *preds42["no_timeofday_11"][:2])
    log(f"A2a bitti. F1 full={f_full.mean():.4f}+-{f_full.std(ddof=1):.4f} "
        f"ablated={f_abl.mean():.4f}+-{f_abl.std(ddof=1):.4f} p={tt.pvalue:.4f} McNemar p={mc['p']:.3g}")

    # ---------------- A2b: kronolojik ayrim ----------------
    log("A2b basliyor: kronolojik ayrim, 5 tohum + ileri zincirleme splitler")
    rows = []
    chrono_preds = None
    for k in (3, 4, 5):
        tr_days, te_days = DAY_ORDER[:k], DAY_ORDER[k:]
        trm, tem = np.isin(days, tr_days), np.isin(days, te_days)
        for seed in SEEDS:
            m, pred, proba = fit_eval(make_xgb(y[trm], seed), X[trm], y[trm], X[tem], y[tem])
            rows.append({"experiment": "chronological", "train_days": k, "test_days": 6 - k,
                         "seed": seed, "n_test": int(tem.sum()),
                         "test_attack_share": float(y[tem].mean()), **m})
            log(f"  chrono train={k}g seed={seed} f1={m['f1']:.4f}")
            if k == 4 and seed == 42:
                chrono_preds = (y[tem], pred, proba)
    a2b = pd.DataFrame(rows)
    a2b.to_csv(OUT_DIR / "A2_chronological_runs.csv", index=False)
    summarize(a2b, ["train_days"]).to_csv(OUT_DIR / "A2_chronological_summary.csv", index=False)
    summary["A2_chrono_4train_ci95_seed42"] = bootstrap_ci(*chrono_preds)
    # ablasyon x kronolojik (zaman ozelligi olmadan 4/2 ayrim), 5 tohum
    rows = []
    trm, tem = np.isin(days, DAY_ORDER[:4]), np.isin(days, DAY_ORDER[4:])
    cols = [c for c in FEATURES if c not in ("startTimeOfDay", "stopTimeOfDay")]
    for seed in SEEDS:
        m, _, _ = fit_eval(make_xgb(y[trm], seed), X[trm][cols], y[trm], X[tem][cols], y[tem])
        rows.append({"experiment": "chronological_no_timeofday", "seed": seed, **m})
        log(f"  chrono(4g) no_timeofday seed={seed} f1={m['f1']:.4f}")
    a2c = pd.DataFrame(rows)
    a2c.to_csv(OUT_DIR / "A2_chronological_no_timeofday_runs.csv", index=False)
    summarize(a2c.assign(variant="chrono_no_timeofday"), ["variant"]).to_csv(
        OUT_DIR / "A2_chronological_no_timeofday_summary.csv", index=False)

    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "iscx2012_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"TAMAMLANDI {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()
