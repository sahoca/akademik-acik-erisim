"""CIC-IDS2017 on dis dogrulama (CDEJ 2049182, Hakem-1 ve Hakem-2 dis dogrulama talebi).

Kapsam bilerek dar tutulmustur: yalnizca (1) kaynak/hedef IP adresi sizintisi ve
(2) zaman damgasindan turetilen gunun saati ozelliginin etkisi test edilir.
Port, TCP pencere boyutu veya SHAP analizi bu betigin kapsami disindadir.

On isleme ve hiperparametreler ISCXIDS2012 makalesindeki ip_included_baseline.py ve
ablation_no_timeofday.py ile ayni mantigi izler.
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

CSV_DIR = Path(os.environ.get("CICIDS2017_CSV_DIR", "data/cicids2017/TrafficLabelling"))
OUT_DIR = Path(__file__).parent / os.environ.get("RESULTS_DIR", "results")
OUT_DIR.mkdir(exist_ok=True)
SEEDS = [0, 42, 123, 1024, 2024]
META = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def time_of_day(ts):
    """Zaman damgasindan gunun saniyesi. Veri setinde saatler AM/PM isareti olmadan
    12 saatlik bicimde kaydedilmistir; yakalama 09:00-17:00 arasinda yapildigindan
    8'den kucuk saat degerleri ogleden sonrayi gosterir ve 12 saat eklenir."""
    try:
        hms = ts.strip().split(" ")[1].split(":")
        h = int(hms[0]); m = int(hms[1]); s = int(hms[2]) if len(hms) > 2 else 0
        if h < 8:
            h += 12
        return 3600 * h + 60 * m + s
    except Exception:
        return np.nan


def load():
    frames = []
    for f in sorted(glob.glob(str(CSV_DIR / "*.csv"))):
        name = Path(f).stem
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df[df["Label"].notna() & (df["Label"].astype(str).str.strip() != "Label")]
        df["timeOfDay"] = df["Timestamp"].astype(str).apply(time_of_day)
        df["source_file"] = name
        frames.append(df)
        log(f"  {name}: {len(df):,} akis")
    return pd.concat(frames, ignore_index=True)


def build(df):
    y = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
    num_cols = [c for c in df.columns if c not in META + ["timeOfDay", "source_file"]]
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X["timeOfDay"] = pd.to_numeric(df["timeOfDay"], errors="coerce")
    X["timeOfDay"] = X["timeOfDay"].fillna(X["timeOfDay"].median())
    X["srcIP"] = LabelEncoder().fit_transform(df["Source IP"].astype(str))
    X["dstIP"] = LabelEncoder().fit_transform(df["Destination IP"].astype(str))
    return X, y


SCENARIOS = {
    "ip_included": lambda cols: cols,
    "ip_excluded": lambda cols: [c for c in cols if c not in ("srcIP", "dstIP")],
    "ip_and_time_excluded": lambda cols: [c for c in cols if c not in ("srcIP", "dstIP", "timeOfDay")],
}


def make_models(y_train, seed):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=seed, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1,
                                               random_state=seed, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     scale_pos_weight=spw, n_jobs=-1, random_state=seed,
                                     eval_metric="logloss", tree_method="hist"),
    }


def metrics(y_true, pred, proba):
    return {"accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "f1": f1_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba)}


def summarize(df, keys):
    g = df.groupby(keys)
    out = g[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()


def main():
    t0 = time.time()
    log("CIC-IDS2017 yukleniyor")
    df = load()
    X, y = build(df)
    all_cols = list(X.columns)
    log(f"Akis={len(X):,} saldiri={int(y.sum()):,} normal={int((y==0).sum()):,} "
        f"ozellik(IP+zaman dahil)={len(all_cols)} benzersiz srcIP={X['srcIP'].nunique():,} "
        f"dstIP={X['dstIP'].nunique():,} timeOfDay NaN orani={df['timeOfDay'].isna().mean():.4f}")
    summary = {"n_flows": int(len(X)), "n_attack": int(y.sum()), "n_normal": int((y == 0).sum()),
               "n_features_with_ip_time": len(all_cols),
               "n_unique_src_ip": int(X["srcIP"].nunique()), "n_unique_dst_ip": int(X["dstIP"].nunique()),
               "time_of_day_nan_rate": float(df["timeOfDay"].isna().mean()),
               "label_counts": df["Label"].astype(str).str.strip().value_counts().to_dict()}

    # B1: tohum 42, uc model, uc senaryo (makaledeki Tablo 1 ile ayni duzen)
    log("B1: uc model x uc senaryo, tohum 42")
    tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=42)
    rows = []
    for scen, fn in SCENARIOS.items():
        cols = fn(all_cols)
        for name, clf in make_models(y.iloc[tr], 42).items():
            clf.fit(X.iloc[tr][cols], y.iloc[tr])
            pred = clf.predict(X.iloc[te][cols]); proba = clf.predict_proba(X.iloc[te][cols])[:, 1]
            m = metrics(y.iloc[te], pred, proba)
            rows.append({"scenario": scen, "model": name, "n_features": len(cols), "seed": 42, **m})
            log(f"  {scen} {name} f1={m['f1']:.4f} auc={m['roc_auc']:.5f}")
    b1 = pd.DataFrame(rows)
    b1.to_csv(OUT_DIR / "B1_cicids2017_scenarios_seed42.csv", index=False)

    # B2: XGBoost, 5 tohum, uc senaryo (ortalama +- std, eslestirilmis test)
    log("B2: XGBoost x uc senaryo x 5 tohum")
    rows = []
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=seed)
        for scen, fn in SCENARIOS.items():
            cols = fn(all_cols)
            clf = make_models(y.iloc[tr], seed)["XGBoost"]
            clf.fit(X.iloc[tr][cols], y.iloc[tr])
            pred = clf.predict(X.iloc[te][cols]); proba = clf.predict_proba(X.iloc[te][cols])[:, 1]
            m = metrics(y.iloc[te], pred, proba)
            rows.append({"scenario": scen, "model": "XGBoost", "seed": seed, **m})
            log(f"  seed={seed} {scen} f1={m['f1']:.4f}")
    b2 = pd.DataFrame(rows)
    b2.to_csv(OUT_DIR / "B2_cicids2017_xgb_5seed_runs.csv", index=False)
    summarize(b2, ["scenario"]).to_csv(OUT_DIR / "B2_cicids2017_xgb_5seed_summary.csv", index=False)
    f = {s: b2[b2.scenario == s].sort_values("seed")["f1"].values for s in SCENARIOS}
    t_ip = stats.ttest_rel(f["ip_included"], f["ip_excluded"])
    t_time = stats.ttest_rel(f["ip_excluded"], f["ip_and_time_excluded"])
    summary["B2_f1_means"] = {s: float(v.mean()) for s, v in f.items()}
    summary["B2_f1_stds"] = {s: float(v.std(ddof=1)) for s, v in f.items()}
    summary["B2_paired_t_ip"] = {"t": float(t_ip.statistic), "p": float(t_ip.pvalue)}
    summary["B2_paired_t_time"] = {"t": float(t_time.statistic), "p": float(t_time.pvalue)}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "cicids2017_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log(f"TAMAMLANDI {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()
