"""ISCXIDS2012 ek kronolojik deneyler: sabit gun bolunmesinde tohum etkisi olmadigindan
(tam-yigin XGBoost deterministik), her tohumda egitim gunlerinin akislarinin %90'i rastgele
alt orneklenerek bes bagimsiz egitim kumesi olusturulur; test gunleri sabittir.
Ayrica gun bazinda saldiri sayilari raporlanir."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from iscx2012_revizyon_deneyleri import (DAY_ORDER, FEATURES, SEEDS, build_features, fit_eval,
                                         load_all_days, log, make_xgb, summarize)

OUT_DIR = Path(__file__).parent / os.environ.get("RESULTS_DIR", "results")
OUT_DIR.mkdir(exist_ok=True)


def main():
    df = load_all_days()
    X, y = build_features(df)
    days = df["source_day"].values
    per_day = (pd.DataFrame({"day": days, "attack": y.values})
               .groupby("day")["attack"].agg(["size", "sum"]).reindex(DAY_ORDER))
    per_day.columns = ["n_flows", "n_attack"]
    per_day["attack_share"] = per_day["n_attack"] / per_day["n_flows"]
    per_day.to_csv(OUT_DIR / "A3_per_day_counts.csv")
    log(f"gun bazinda:\n{per_day}")

    no_time = [c for c in FEATURES if c not in ("startTimeOfDay", "stopTimeOfDay")]
    rows = []
    for k in (3, 4, 5):
        trm, tem = np.isin(days, DAY_ORDER[:k]), np.isin(days, DAY_ORDER[k:])
        tr_idx = np.where(trm)[0]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            sub = rng.choice(tr_idx, size=int(0.9 * len(tr_idx)), replace=False)
            for variant, cols in (("full_13", FEATURES), ("no_timeofday_11", no_time)):
                m, _, _ = fit_eval(make_xgb(y.iloc[sub], seed), X.iloc[sub][cols], y.iloc[sub], X[tem][cols], y[tem])
                rows.append({"train_days": k, "variant": variant, "seed": seed, **m})
                log(f"  chrono {k}g subsample seed={seed} {variant} f1={m['f1']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f}")
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_DIR / "A3_chronological_subsample_runs.csv", index=False)
    summarize(runs, ["train_days", "variant"]).to_csv(OUT_DIR / "A3_chronological_subsample_summary.csv", index=False)
    log("TAMAMLANDI")


if __name__ == "__main__":
    main()
