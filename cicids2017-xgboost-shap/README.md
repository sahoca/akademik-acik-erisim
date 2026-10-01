# CIC-IDS2017: XGBoost ve SHAP Analizi

CIC-IDS2017 veri seti üzerinde 5-katlı çapraz doğrulama, TreeSHAP öznitelik analizi, ablasyon deneyi ve gün bazlı değerlendirme kodları.

## Veri

Veri seti bu depoda yer almaz. CSV dosyaları [unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) adresinden temin edilebilir.

## Çalıştırma

```bash
pip install -r requirements.txt
python cicids2017_full_cv_pipeline.py
python diagnose_groupkfold.py
```

## Dosyalar

* `cicids2017_full_cv_pipeline.py`: 5-katlı çapraz doğrulama, SHAP ve ablasyon deneyi.
* `diagnose_groupkfold.py`: Gün bazlı analiz ve eşik değerlendirmesi.
* `figures/`: SHAP grafikleri.
* `results/`: Deney sonuç tabloları.

## Lisans

MIT
