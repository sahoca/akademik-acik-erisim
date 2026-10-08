# Revizyon Deneyleri

`iscx2012-ml-shap/` altındaki ana analizin hakem değerlendirmesi sonrasında eklenen sağlamlık denetimleri ve dış doğrulama kodları.

## İçerik

| Dosya | Açıklama |
|---|---|
| `iscx2012_revizyon_deneyleri.py` | ISCXIDS2012: akış düzeyi ve 5-tuple oturum düzeyi (GroupShuffleSplit) bölünme karşılaştırması; zaman özelliği ablasyonu ve kronolojik ayrımın beş tohumlu tekrarı; ileri zincirleme bölünmeler; bootstrap güven aralıkları, eşleştirilmiş t-testi ve McNemar testi |
| `cicids2017_dis_dogrulama.py` | CIC-IDS2017: IP'li, IP'siz ve IP'siz ve zamansız senaryolarda Karar Ağacı, Random Forest ve XGBoost karşılaştırması; XGBoost için beş tohumlu tekrar |
| `results/` | Deney çıktıları (CSV ve JSON) |

Ön işleme ve hiperparametreler ana analizdeki `train_and_shap.py` ile aynıdır.

## Veri

Veri setleri depoda yer almaz.

* ISCXIDS2012 akış CSV dosyaları (`TestbedSatJun12Flows.csv` ... `TestbedThuJun17Flows.csv`): [unb.ca/cic/datasets/ids.html](https://www.unb.ca/cic/datasets/ids.html). Varsayılan konum `data/iscx2012/CSV/`, `ISCX2012_CSV_DIR` ortam değişkeni ile değiştirilebilir.
* CIC-IDS2017 ham etiketli akış dosyaları (`TrafficLabelling`, sekiz CSV, kaynak/hedef IP sütunlarını içeren sürüm): [unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html). Varsayılan konum `data/cicids2017/TrafficLabelling/`, `CICIDS2017_CSV_DIR` ortam değişkeni ile değiştirilebilir.

CIC-IDS2017 dosyaları `latin1` kodlaması ile okunur; zaman damgaları 12 saatlik biçimde ve AM/PM işareti olmadan kaydedildiğinden 08:00 öncesi görünen saat değerlerine 12 saat eklenir.

## Çalıştırma

```bash
pip install -r ../iscx2012-ml-shap/requirements.txt scipy
python iscx2012_revizyon_deneyleri.py
python cicids2017_dis_dogrulama.py
```

Çıktılar `results/` altına yazılır (`RESULTS_DIR` ile değiştirilebilir).

## Lisans

MIT
