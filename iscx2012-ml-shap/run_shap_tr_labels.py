import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

shap.plots._labels.labels.update({
    "VALUE": "SHAP değeri (model çıktısına etkisi)",
    "GLOBAL_VALUE": "Ort. |SHAP değeri| (ortalama etki büyüklüğü)",
    "FEATURE_VALUE": "Özellik değeri",
    "FEATURE_VALUE_LOW": "Düşük",
    "FEATURE_VALUE_HIGH": "Yüksek",
    "FEATURE": "Özellik %s",
    "MODEL_OUTPUT": "Model çıktı değeri",
})

BASE = Path(__file__).parent
OUT_DIR = BASE / "out"

model = joblib.load(next(OUT_DIR.glob("best_model_*.pkl")))
X_test = pd.read_parquet(OUT_DIR / "iscx2012_features_test.parquet")
y_test = pd.read_parquet(OUT_DIR / "iscx2012_labels_test.parquet")["label"]

rng = np.random.RandomState(42)
sample_idx = rng.choice(X_test.index, size=min(5000, len(X_test)), replace=False)
X_sample = X_test.loc[sample_idx]
y_sample = y_test.loc[sample_idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

plt.figure()
shap.plots.beeswarm(shap_values, show=False, max_display=15, color_bar_label="Özellik değeri")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=150)
plt.close()

plt.figure()
shap.plots.bar(shap_values, show=False, max_display=15)
plt.gca().set_xlabel("Ort. |SHAP değeri|", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_bar.png", dpi=150)
plt.close()

attack_positions = np.where(y_sample.values == 1)[0]
if len(attack_positions) > 0:
    idx0 = attack_positions[0]
    plt.figure()
    shap.plots.waterfall(shap_values[idx0], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_waterfall_attack_example.png", dpi=150)
    plt.close()
