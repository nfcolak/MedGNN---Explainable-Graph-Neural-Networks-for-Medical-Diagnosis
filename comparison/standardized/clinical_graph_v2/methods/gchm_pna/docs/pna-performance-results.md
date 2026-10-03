# PNA tam veri performans karşılaştırması

59.607 train / 7.448 validation; test değerlendirilmedi. Seed1234, 30 epoch, batch128, Adam lr0.001 wd1e-5, CE; aynı star/common input.

| Model | Macro-F1 | Balanced accuracy | Accuracy | Best epoch | Parametre |
|---|---:|---:|---:|---:|---:|
| interaction | 0.470246 | 0.454446 | 0.574382 | 19 | 153950 |
| plain | 0.472846 | 0.464540 | 0.580559 | 7 | 145758 |

Interaction − plain: `{"accuracy": -0.006177000000000099, "balanced_acc": -0.010093999999999992, "macro_f1": -0.002599999999999991, "micro_f1": -0.006177000000000099, "top3_acc": 0.0021480000000000388, "top5_acc": -0.0004030000000000422}`.

Seçim: ilk maksimum shared macro-F1 (6 ondalık); 30 sınıfın tamamı validation içinde. Per-class, train-en-az 7 sınıf rare metrikleri, final epoch ve süreler hücre metrics/manifest dosyalarında. Seçilmiş checkpoint full validation üzerinde birebir yeniden üretildi.

Tek seed ve farklı parametre sayıları: yöntem üstünlüğü ispatı değil. Tarihsel GraphCare 0.481306 farklı sqrt-loss referansıdır, eş-loss kontrol değildir. Kaynak snapshot temporal_clean=false/raw_to_model_train_only=false; klinik erken-tahmin iddiası yok.

Artifact: `comparison/standardized/pna_experiments/full_ce_seed1234_v1`.
