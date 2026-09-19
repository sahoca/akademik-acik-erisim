import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "out"

model = joblib.load(next(OUT_DIR.glob("best_model_*.pkl")))
X_test = pd.read_parquet(OUT_DIR / "iscx2012_features_test.parquet")
y_test = pd.read_parquet(OUT_DIR / "iscx2012_labels_test.parquet")["label"]

rng = np.random.RandomState(42)
n_total = 5000
n_attack = int(round(n_total * y_test.mean()))
n_normal = n_total - n_attack

attack_idx = y_test[y_test == 1].index
normal_idx = y_test[y_test == 0].index

sample_attack = rng.choice(attack_idx, size=min(n_attack, len(attack_idx)), replace=False)
sample_normal = rng.choice(normal_idx, size=min(n_normal, len(normal_idx)), replace=False)
sample_idx = np.concatenate([sample_attack, sample_normal])
rng.shuffle(sample_idx)

X_sample = X_test.loc[sample_idx]
y_sample = y_test.loc[sample_idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

plt.figure()
shap.plots.beeswarm(shap_values, show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_beeswarm_stratified.png", dpi=150)
plt.close()

plt.figure()
shap.plots.bar(shap_values, show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_bar_stratified.png", dpi=150)
plt.close()

mean_abs = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": mean_abs,
}).sort_values("mean_abs_shap", ascending=False)
importance_df.to_csv(OUT_DIR / "shap_feature_importance_stratified.csv", index=False)
