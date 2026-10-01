# ISCXIDS2012: IP Sızıntısından Arındırılmış SHAP Analizi

ISCXIDS2012 veri seti üzerinde IP tabanlı özellik sızıntısını ölçen, IP'siz koşulda Karar Ağacı / Random Forest / XGBoost karşılaştırması yapan ve en iyi model için SHAP tabanlı açıklanabilirlik analizi üreten pipeline.

## Veri

Veri seti bu depoda yer almaz. `TestbedSatJun12Flows.csv` ... `TestbedThuJun17Flows.csv` (6 dosya) [unb.ca/cic/datasets/ids.html](https://www.unb.ca/cic/datasets/ids.html) adresinden indirilip `data/CSV/` altına konmalıdır.

## Akış

```mermaid
flowchart TD
    A[data/CSV/*.csv] --> B[train_and_shap.py]
    B -->|IP sütunları çıkarılır, 13 özellik| C[Karar Ağacı / Random Forest / XGBoost]
    C --> D[out/no_leak_model_comparison.csv]
    C --> E[out/best_model_*.pkl]
    E --> F[run_shap.py]
    E --> G[run_shap_stratified.py]
    E --> H[run_shap_tr_labels.py]
    F --> I[out/shap_feature_importance.csv]
    H --> J[out/shap_*.png]
    B -.-> K[cv_xgboost.py]
    B -.-> L[ablation_no_timeofday.py]
    A --> M[ip_included_baseline.py]
    M -->|IP sütunları dahil| N[out/ip_included_model_comparison.csv]
    A --> O[time_based_split.py]
    O -->|kronolojik 4g/2g ayrım| P[out/time_based_split_xgboost.csv]
```

## Çalıştırma sırası

```bash
pip install -r requirements.txt
python train_and_shap.py
python cv_xgboost.py
python ablation_no_timeofday.py
python ip_included_baseline.py
python time_based_split.py
python run_shap.py
python run_shap_stratified.py
python run_shap_tr_labels.py
```

## Dosyalar

| Dosya | Çıktı |
|---|---|
| `train_and_shap.py` | IP'siz 3-model karşılaştırması, en iyi model, parquet/pkl çıktıları |
| `cv_xgboost.py` | XGBoost için 5-katlı stratifiye çapraz doğrulama |
| `ablation_no_timeofday.py` | startTimeOfDay/stopTimeOfDay çıkarılınca performans |
| `ip_included_baseline.py` | Aynı ayrım/hiperparametrelerle IP dahil kontrollü karşılaştırma |
| `time_based_split.py` | Kronolojik (zamana dayalı) eğitim/test ayrımı |
| `run_shap.py` | SHAP değerleri, beeswarm/bar/waterfall grafikleri (rastgele örneklem) |
| `run_shap_stratified.py` | Aynı analiz, sınıf oranı korunan stratifiye örneklem |
| `run_shap_tr_labels.py` | Grafiklerin Türkçe eksen/lejant etiketleriyle üretimi |

`random_state=42` tüm bölünme ve modellerde sabittir.

## Atıf

Bu kod, aşağıdaki makalenin deneysel sonuçlarını üretmek için kullanılmıştır:

Akca, Ş., & Urfalıoğlu, M. (2026). ISCXIDS2012 Veri Setinde IP Sızıntısından Arındırılmış Açıklanabilir Saldırı Tespiti: XGBoost, Random Forest ve Karar Ağacı Modellerinin SHAP ile Karşılaştırılması. *Siber Güvenlik ve Dijital Ekonomi Dergisi*.

## Lisans

MIT (bkz. `LICENSE`).
