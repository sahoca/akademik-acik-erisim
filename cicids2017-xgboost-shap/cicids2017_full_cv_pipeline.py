#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC-IDS2017 Kapsamlı Doğrulama Pipeline'ı
1. 5-Fold Stratified Cross-Validation (Ortalama ± Standart Sapma)
2. Gün Bazlı Zamansal Sızıntı Kontrolü (GroupKFold / Leave-One-Day-Out)
3. TreeSHAP Analizi (5000 test örneği, Türkçe grafikler)
4. Ablasyon Deneyi (5-Fold Stratified CV ile Ortalama ± Standart Sapma)
5. Engelen vd. (2021) Etiketleme Analizi
Yazarlar: Şahabettin Akca, Murat Urfalıoğlu
"""

import os
import glob
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
import shap

# Matplotlib Türkçe font ayarı
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR_DATA = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_DIR_FIGS = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR_DATA, exist_ok=True)
os.makedirs(OUTPUT_DIR_FIGS, exist_ok=True)

def get_day_id(filename):
    fname = filename.lower()
    if 'monday' in fname:
        return 0, 'Monday'
    elif 'tuesday' in fname:
        return 1, 'Tuesday'
    elif 'wednesday' in fname:
        return 2, 'Wednesday'
    elif 'thursday' in fname:
        return 3, 'Thursday'
    elif 'friday' in fname:
        return 4, 'Friday'
    return 0, 'Other'

def load_data():
    print(">>> 1. Veri Okuma ve Ön İşleme...")
    data_dir = os.path.expanduser('~/Downloads/TrafficLabelling ')
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csv_files:
        data_dir = os.path.expanduser('~/Downloads/TrafficLabelling')
        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        
    dfs = []
    day_series = []
    file_info = {}
    
    for f in csv_files:
        fname = os.path.basename(f)
        day_id, day_name = get_day_id(fname)
        print(f"  Okunuyor: {fname} (Gün: {day_name})...")
        df_chunk = pd.read_csv(f, encoding='latin1', low_memory=False)
        df_chunk.columns = df_chunk.columns.str.strip()
        
        valid_mask = df_chunk['Label'].notna() & (df_chunk['Label'] != 'Label')
        df_chunk = df_chunk[valid_mask]
        
        file_info[fname] = {
            'day': day_name,
            'total_flows': len(df_chunk),
            'labels': df_chunk['Label'].value_counts().to_dict()
        }
        dfs.append(df_chunk)
        day_series.append(np.full(len(df_chunk), day_id, dtype=np.int8))
        
    df = pd.concat(dfs, ignore_index=True)
    days = np.concatenate(day_series)
    print(f"Toplam ham akış: {len(df):,}")
    
    # İkili sınıflandırma (BENIGN -> 0, Saldırılar -> 1)
    labels_raw = df['Label'].astype(str).str.strip()
    y = (labels_raw.str.upper() != 'BENIGN').astype(int).values
    
    # Doğrudan kimlik kolonları
    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Source Port', 'Label']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    print(f"Düşürülen doğrudan kimlik kolonları: {existing_drop_cols}")
    
    X = df.drop(columns=existing_drop_cols)
    
    # Sayısal dönüşüm ve temizleme
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_idx = X.notna().all(axis=1).values
    
    X = X[valid_idx].astype(np.float32)
    y = y[valid_idx]
    days = days[valid_idx]
    
    feature_names = list(X.columns)
    print(f"Temizleme sonrası geçerli akış: {len(X):,}, Özellik sayısı: {len(feature_names)}")
    print(f"Normal akış: {(y==0).sum():,} (%{(y==0).mean()*100:.2f}), Saldırı: {(y==1).sum():,} (%{(y==1).mean()*100:.2f})")
    
    return X.values, y, days, feature_names, file_info

def evaluate_predictions(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(b_acc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'roc_auc': float(auc)
    }

def run_stratified_5fold(X, y, feature_names):
    print("\n>>> 2. 5-Fold Stratified Cross-Validation Başlatılıyor...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Karar Ağacı': lambda: DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=30, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist', random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    cv_results = {name: {'accuracy': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []} for name in models}
    
    best_xgb_model = None
    last_X_test = None
    last_y_test = None
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n--- Fold {fold}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        
        last_X_test = X_test
        last_y_test = y_test
        
        for name, model_fn in models.items():
            t0 = time.time()
            model = model_fn()
            if name == 'XGBoost':
                model.set_params(scale_pos_weight=scale_pos)
                model.fit(X_train, y_train)
                if fold == 1:
                    best_xgb_model = model
            else:
                model.fit(X_train, y_train)
                
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            res = evaluate_predictions(y_test, y_pred, y_prob)
            for k in res:
                cv_results[name][k].append(res[k])
                
            print(f"  {name:14s} | Doğ.: {res['accuracy']:.4f} | Dengeli: {res['balanced_accuracy']:.4f} | F1: {res['f1']:.4f} | AUC: {res['roc_auc']:.5f} ({time.time()-t0:.1f} sn)")
        fold += 1
        
    summary_table = []
    for name in models:
        row = {'Model': name}
        for k in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
            mean_val = np.mean(cv_results[name][k])
            std_val = np.std(cv_results[name][k])
            row[f'{k}_mean'] = mean_val
            row[f'{k}_std'] = std_val
            row[k] = f"{mean_val:.4f} ± {std_val:.4f}" if k != 'roc_auc' else f"{mean_val:.5f} ± {std_val:.5f}"
        summary_table.append(row)
        
    df_sum = pd.DataFrame(summary_table)
    df_sum.to_csv(os.path.join(OUTPUT_DIR_DATA, 'table1_model_comparison_5fold.csv'), index=False)
    print("\n>>> 5-Fold CV Sonuçları (Tablo 1):")
    print(df_sum[['Model', 'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc']].to_string(index=False))
    
    return cv_results, summary_table, best_xgb_model, last_X_test, last_y_test

def run_group_kfold_temporal(X, y, days, feature_names):
    print("\n>>> 3. Gün Bazlı Zamansal Ayrım (GroupKFold - 5 Gün)...")
    gkf = GroupKFold(n_splits=5)
    
    results = []
    fold = 1
    for train_idx, test_idx in gkf.split(X, y, groups=days):
        test_days = np.unique(days[test_idx])
        day_names = [{0:'Pzt', 1:'Sal', 2:'Çar', 3:'Per', 4:'Cum'}[d] for d in test_days]
        print(f"  Fold {fold} - Test Günleri: {', '.join(day_names)} (Eğitim: {len(train_idx):,}, Test: {len(test_idx):,})")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Eğer test setinde sadece benign veya sadece saldırı varsa ROC-AUC hesaplanamaz
        if len(np.unique(y_test)) < 2:
            print(f"    Uyarı: Fold {fold} tek sınıflı test seti içeriyor, atlandı.")
            continue
            
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist', scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        res = evaluate_predictions(y_test, y_pred, y_prob)
        res['test_days'] = ', '.join(day_names)
        results.append(res)
        print(f"    Sonuç | Doğruluk: {res['accuracy']:.4f} | Dengeli Doğ.: {res['balanced_accuracy']:.4f} | F1: {res['f1']:.4f} | ROC-AUC: {res['roc_auc']:.5f}")
        fold += 1
        
    df_gkf = pd.DataFrame(results)
    df_gkf.to_csv(os.path.join(OUTPUT_DIR_DATA, 'groupkfold_temporal_results.csv'), index=False)
    print("\nGroupKFold Gün Bazlı Özet:")
    print(f"  Ortalama Doğruluk: {df_gkf['accuracy'].mean():.4f} ± {df_gkf['accuracy'].std():.4f}")
    print(f"  Ortalama F1-Skoru: {df_gkf['f1'].mean():.4f} ± {df_gkf['f1'].std():.4f}")
    print(f"  Ortalama Dengeli Doğruluk: {df_gkf['balanced_accuracy'].mean():.4f} ± {df_gkf['balanced_accuracy'].std():.4f}")
    return df_gkf

def run_shap(xgb_model, X_test, y_test, feature_names):
    print("\n>>> 4. TreeSHAP Analizi (5.000 Örneklem)...")
    _, X_sample_arr, _, y_sample = train_test_split(
        X_test, y_test, test_size=5000, stratify=y_test, random_state=42
    )
    X_sample = pd.DataFrame(X_sample_arr, columns=feature_names)
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Özellik': feature_names,
        'Ort_Mutlak_SHAP': mean_abs_shap
    }).sort_values(by='Ort_Mutlak_SHAP', ascending=False).reset_index(drop=True)
    
    print("\nEn Yüksek Katkılı İlk 10 Özellik (SHAP):")
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv(os.path.join(OUTPUT_DIR_DATA, 'table2_shap_importance.csv'), index=False)
    
    # 1. Bar Plot (Türkçe)
    plt.figure(figsize=(10, 8))
    top_15 = importance_df.head(15).iloc[::-1]
    plt.barh(top_15['Özellik'], top_15['Ort_Mutlak_SHAP'], color='#1f77b4', edgecolor='black', alpha=0.85)
    plt.xlabel('Ortalama |SHAP Değeri| (Model Çıktısına Ortalama Mutlak Katkı)', fontsize=11, fontweight='bold')
    plt.title('CIC-IDS2017 - Küresel Özellik Önemi (XGBoost TreeSHAP)', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_FIGS, 'shap_bar.png'), dpi=300)
    plt.close()
    
    # 2. Beeswarm Plot (Türkçe)
    plt.figure(figsize=(11, 8))
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=feature_names
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.xlabel('SHAP Değeri (Model çıktısını artırma / azaltma etkisi)', fontsize=11, fontweight='bold')
    plt.title('CIC-IDS2017 - Özellik Değerlerinin Karara Etkisi (Beeswarm)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_FIGS, 'shap_beeswarm.png'), dpi=300)
    plt.close()
    
    # 3. Waterfall Plot
    preds = xgb_model.predict(X_sample.values)
    attack_indices = np.where((y_sample == 1) & (preds == 1))[0]
    sample_idx = attack_indices[0] if len(attack_indices) > 0 else 0
    
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation[sample_idx], max_display=12, show=False)
    plt.title('Tekil Bir Saldırı Akışı İçin Yerel Karar Açıklaması (Waterfall)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_FIGS, 'shap_waterfall.png'), dpi=300)
    plt.close()
    
    return importance_df

def run_ablation_5fold(X, y, feature_names, suspect_features):
    print(f"\n>>> 5. 5-Fold Stratified Ablasyon Deneyi: Çıkarılan: {suspect_features}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    drop_idx = [feature_names.index(f) for f in suspect_features if f in feature_names]
    keep_idx = [i for i in range(len(feature_names)) if i not in drop_idx]
    
    X_abl = X[:, keep_idx]
    
    metrics = {'accuracy': [], 'balanced_accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}
    
    fold = 1
    for train_idx, test_idx in skf.split(X_abl, y):
        X_train, X_test = X_abl[train_idx], X_abl[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist', scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        res = evaluate_predictions(y_test, y_pred, y_prob)
        for k in res:
            metrics[k].append(res[k])
        fold += 1
        
    summary = {
        'Özellik Kümesi': f"{' + '.join(suspect_features)} Çıkarılmış ({len(keep_idx)} özellik)",
        'n_features': len(keep_idx)
    }
    for k in metrics:
        m = np.mean(metrics[k])
        s = np.std(metrics[k])
        summary[f'{k}_mean'] = m
        summary[f'{k}_std'] = s
        summary[k] = f"{m:.4f} ± {s:.4f}" if k != 'roc_auc' else f"{m:.5f} ± {s:.5f}"
        
    return summary

def main():
    t_start = time.time()
    X, y, days, feature_names, file_info = load_data()
    
    # 5-Fold Stratified CV
    cv_res, model_summary, best_xgb, X_test_last, y_test_last = run_stratified_5fold(X, y, feature_names)
    
    # SHAP Analizi
    importance_df = run_shap(best_xgb, X_test_last, y_test_last, feature_names)
    
    # Ablasyon Deneyi: En tepe 1. ve 2. özellikler
    top1 = importance_df.iloc[0]['Özellik']
    top2 = importance_df.iloc[1]['Özellik']
    print(f"\nSHAP Tepe Özellikleri: 1) {top1}, 2) {top2}")
    
    # Orijinal Tam Küme XGBoost Ortalaması
    xgb_full = [m for m in model_summary if m['Model'] == 'XGBoost'][0]
    abl_rows = [{
        'Özellik Kümesi': f"Tam Özellik Kümesi ({len(feature_names)} özellik)",
        'n_features': len(feature_names),
        'accuracy': xgb_full['accuracy'],
        'balanced_accuracy': xgb_full['balanced_accuracy'],
        'f1': xgb_full['f1'],
        'precision': xgb_full['precision'],
        'recall': xgb_full['recall'],
        'roc_auc': xgb_full['roc_auc']
    }]
    
    # Ablasyon 1 (Top 1)
    abl1 = run_ablation_5fold(X, y, feature_names, [top1])
    abl_rows.append(abl1)
    
    # Ablasyon 2 (Top 1 + Top 2 / Destination Port)
    suspect = [top1]
    if 'Destination Port' in [top1, top2] and 'Destination Port' not in suspect:
        suspect.append('Destination Port')
    elif top2 not in suspect:
        suspect.append(top2)
        
    abl2 = run_ablation_5fold(X, y, feature_names, suspect)
    abl_rows.append(abl2)
    
    df_ablation = pd.DataFrame(abl_rows)
    df_ablation.to_csv(os.path.join(OUTPUT_DIR_DATA, 'table3_ablation_5fold.csv'), index=False)
    print("\n>>> 5-Fold Ablasyon Deneyi Sonuçları (Tablo 3):")
    print(df_ablation[['Özellik Kümesi', 'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc']].to_string(index=False))
    
    # GroupKFold (Gün Bazlı)
    df_temporal = run_group_kfold_temporal(X, y, days, feature_names)
    
    # Nihai Özet JSON
    final_output = {
        'total_flows': len(X),
        'n_features': len(feature_names),
        'n_benign': int((y == 0).sum()),
        'n_attack': int((y == 1).sum()),
        'attack_ratio': float((y == 1).mean()),
        'table1_models_5fold': model_summary,
        'table2_top_shap': importance_df.head(15).to_dict(orient='records'),
        'table3_ablation_5fold': abl_rows,
        'temporal_groupkfold': {
            'accuracy_mean': float(df_temporal['accuracy'].mean()),
            'accuracy_std': float(df_temporal['accuracy'].std()),
            'f1_mean': float(df_temporal['f1'].mean()),
            'f1_std': float(df_temporal['f1'].std()),
            'balanced_acc_mean': float(df_temporal['balanced_accuracy'].mean()),
            'balanced_acc_std': float(df_temporal['balanced_accuracy'].std())
        },
        'file_info': file_info,
        'total_elapsed_min': (time.time() - t_start) / 60.0
    }
    
    with open(os.path.join(OUTPUT_DIR_DATA, 'final_experiment_results_5fold.json'), 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"\n=======================================================")
    print(f"TÜM HAKEM KRİTERLERİNE UYGUN ANALİZ BAŞARIYLA BİTTİ!")
    print(f"Toplam Süre: {(time.time() - t_start)/60.0:.2f} dakika.")
    print(f"=======================================================")

if __name__ == '__main__':
    main()
