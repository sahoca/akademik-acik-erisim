#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC-IDS2017 Leave-One-Day-Out Validation-Calibrated Threshold vs Default & Oracle
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, precision_recall_curve
)

data_dir = '/Users/saho/Downloads/03_Veri_ve_Yazilim_Kaynaklari/MachineLearning_Veri_Setleri/TrafficLabelling '
files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))

day_map = {
    'monday': 'Monday',
    'tuesday': 'Tuesday',
    'wednesday': 'Wednesday',
    'thursday': 'Thursday',
    'friday': 'Friday'
}

dfs = []
day_labels = []

print("Veriler yükleniyor...")
for f in files:
    fname = os.path.basename(f)
    dname = 'Unknown'
    for k in day_map:
        if k in fname.lower():
            dname = day_map[k]
            break
    df_chunk = pd.read_csv(f, encoding='latin1', low_memory=False)
    df_chunk.columns = df_chunk.columns.str.strip()
    valid_mask = df_chunk['Label'].notna() & (df_chunk['Label'] != 'Label')
    df_chunk = df_chunk[valid_mask]
    dfs.append(df_chunk)
    day_labels.extend([dname] * len(df_chunk))

df = pd.concat(dfs, ignore_index=True)
days = np.array(day_labels)

labels_raw = df['Label'].astype(str).str.strip()
y = (labels_raw.str.upper() != 'BENIGN').astype(int).values

drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Source Port', 'Label']
existing_drop = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=existing_drop)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)
valid_idx = X.notna().all(axis=1).values
X = X[valid_idx].astype(np.float32).values
y = y[valid_idx]
days = days[valid_idx]
raw_attacks = labels_raw.values[valid_idx]

print(f"Toplam geçerli akış: {len(X):,}")

ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

results_table = []

for fold_idx, test_day in enumerate(ordered_days, 1):
    test_mask = (days == test_day)
    train_mask = ~test_mask
    
    X_train_full, X_test = X[train_mask], X[test_mask]
    y_train_full, y_test = y[train_mask], y[test_mask]
    
    n_test_benign = int((y_test == 0).sum())
    n_test_attack = int((y_test == 1).sum())
    test_total = len(y_test)
    attack_ratio = (n_test_attack / test_total) * 100
    
    if n_test_attack == 0:
        # Monday (sadece normal)
        scale_pos = (y_train_full == 0).sum() / max((y_train_full == 1).sum(), 1)
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist',
                              scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric='logloss')
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(X_test)
        fp = int((y_pred == 1).sum())
        tn = int((y_pred == 0).sum())
        
        row = {
            'Fold': fold_idx,
            'Test_Günü': test_day,
            'BENIGN_Sayı': f"{n_test_benign:,}",
            'Saldırı_Sayı': f"{n_test_attack:,} (%{attack_ratio:.2f})",
            'Prec_0.50': "N/A*",
            'Rec_0.50': "N/A*",
            'F1_0.50': f"FP: %{fp/test_total*100:.2f}",
            'Kalibre_Esik': "N/A*",
            'Val_F1': "N/A*",
            'Test_Prec_Kalibre': "N/A*",
            'Test_Rec_Kalibre': "N/A*",
            'Test_F1_Kalibre': "N/A*",
            'Oracle_F1': "N/A*",
            'Oracle_Esik': "N/A*",
            'ROC_AUC': "N/A*"
        }
        results_table.append(row)
        print(f"Fold {fold_idx} ({test_day}): Tamamlandı (Saldırı yok, FP={fp:,} %{fp/test_total*100:.2f})")
        continue
    
    # 1) Train/Val split (%85 Train, %15 Val - Stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=42
    )
    
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist',
                          scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric='logloss')
    model.fit(X_tr, y_tr)
    
    # 2) Optimum eşiği SADECE validation diliminde bul
    y_val_prob = model.predict_proba(X_val)[:, 1]
    p_val, r_val, t_val = precision_recall_curve(y_val, y_val_prob)
    f1_val_curve = 2 * (p_val * r_val) / np.maximum(p_val + r_val, 1e-8)
    best_val_idx = np.argmax(f1_val_curve)
    best_thresh = t_val[best_val_idx] if best_val_idx < len(t_val) else 0.5
    best_val_f1 = f1_val_curve[best_val_idx]
    
    # 3) Test setinde tahminler
    y_test_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_test_prob)
    
    # Varsayılan 0.50 eşik
    y_pred_05 = (y_test_prob >= 0.50).astype(int)
    p_05, r_05, f1_05, _ = precision_recall_fscore_support(y_test, y_pred_05, average='binary', zero_division=0)
    
    # Validasyonda kalibre edilmiş eşik (Test gününde)
    y_pred_cal = (y_test_prob >= best_thresh).astype(int)
    p_cal, r_cal, f1_cal, _ = precision_recall_fscore_support(y_test, y_pred_cal, average='binary', zero_division=0)
    
    # Eski Oracle (Test sızıntılı üst sınır)
    p_test, r_test, t_test = precision_recall_curve(y_test, y_test_prob)
    f1_test_curve = 2 * (p_test * r_test) / np.maximum(p_test + r_test, 1e-8)
    best_oracle_idx = np.argmax(f1_test_curve)
    oracle_thresh = t_test[best_oracle_idx] if best_oracle_idx < len(t_test) else 0.5
    oracle_f1 = f1_test_curve[best_oracle_idx]
    
    row = {
        'Fold': fold_idx,
        'Test_Günü': test_day,
        'BENIGN_Sayı': f"{n_test_benign:,}",
        'Saldırı_Sayı': f"{n_test_attack:,} (%{attack_ratio:.2f})",
        'Prec_0.50': f"{p_05:.4f}",
        'Rec_0.50': f"{r_05:.4f}",
        'F1_0.50': f"{f1_05:.4f}",
        'Kalibre_Esik': f"{best_thresh:.4f}",
        'Val_F1': f"{best_val_f1:.4f}",
        'Test_Prec_Kalibre': f"{p_cal:.4f}",
        'Test_Rec_Kalibre': f"{r_cal:.4f}",
        'Test_F1_Kalibre': f"{f1_cal:.4f}",
        'Oracle_F1': f"{oracle_f1:.4f}",
        'Oracle_Esik': f"{oracle_thresh:.4f}",
        'ROC_AUC': f"{auc:.4f}"
    }
    results_table.append(row)
    print(f"Fold {fold_idx} ({test_day}):")
    print(f"   Eşik 0.50       -> F1={f1_05:.4f} (Prec={p_05:.4f}, Rec={r_05:.4f})")
    print(f"   Val Kalibre Esik ({best_thresh:.4f}) -> Test F1={f1_cal:.4f} (Prec={p_cal:.4f}, Rec={r_cal:.4f}), ROC-AUC={auc:.4f}")
    print(f"   Eski Oracle     -> F1={oracle_f1:.4f} (Esik={oracle_thresh:.4f})")

df_res = pd.DataFrame(results_table)
df_res.to_csv('/Users/saho/Desktop/akademik/02_veri/calibrated_leave_one_day_out_results.csv', index=False)
print("\nTAMAMLANDI!")
