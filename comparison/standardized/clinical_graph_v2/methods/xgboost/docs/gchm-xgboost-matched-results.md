# GCHM–XGBoost: eşleştirilmiş karşılaştırma sonucu

## Kapsam ve sonuç

Onaylanan 6 XGBoost koşulu tamamlandı: ağırlıksız/sqrt-inverse × seed 1234/1235/1236.
Mevcut 6 GCHM koşulu kaynak snapshot, checkpoint ve cohort kayıtları doğrulanarak
kullanıldı; yeni GCHM eğitimi yapılmadı. 59.607 eğitim, 7.448 validation kaydı;
**test_evaluated=false**, **temporal_clean=false**. Eski deneyler değiştirilmedi.

**Tek model performansının üç-seed ortalamasında XGBoost her iki ağırlıklandırmada
macro-F1, balanced accuracy ve accuracy'de önde.** Bu validation bulgusu klinik veya
bağımsız test üstünlüğü değildir.

| Model | Macro-F1 ort. ± örnek SS | Balanced accuracy ort. | Accuracy ort. |
|---|---:|---:|---:|
| GCHM / CE | 0.5656 ± 0.0015 | 0.5548 | 0.6310 |
| XGBoost / ağırlıksız | 0.5863 ± 0.0007 | 0.5756 | 0.6464 |
| GCHM / sqrt | 0.5593 ± 0.0058 | 0.5901 | 0.6203 |
| XGBoost / sqrt | 0.5801 ± 0.0008 | 0.5994 | 0.6399 |

Her iki tarafta checkpoint, sabit bütçe boyunca ilk maksimum validation macro-F1
(6 ondalık) ile seçildi. GCHM 30 epoch ve XGBoost 1200 boosting turu kullandığından
hesap bütçesi/checkpoint adedi eşit değildir. XGBoost erken durdurma kullanmadı.
Sqrt ağırlıkları yalnız tam eğitim sınıf sayımlarından hesaplandı; validation
üzerinde ağırlıklandırma veya yeniden örnekleme yoktur. Bu, aynı görevin ağırlık
politikasını eşler; iki optimizer'ın sayısal optimizasyon dinamiğini eşitlemez.

## Ensemble ayrı bir sonuçtur

| Üç-seed olasılık ortalaması | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| GCHM / sqrt | 0.5706 | 0.5991 | 0.6293 |
| XGBoost / sqrt | 0.5786 | 0.5975 | 0.6390 |

GCHM−XGBoost balanced accuracy farkı **+0.0016**, koşullu %95 bootstrap aralığı
**[-0.0095, +0.0123]**. Küçük sayısal fark, güvenilir genel üstünlük kanıtı değil.
Macro-F1 farkı -0.0080 için aralık [-0.0181, +0.0022]; bu ensemble kıyaslamasında
macro-F1 üstünlüğü de aralıkla kesinleştirilemez. Tek-seed/ortalama sonuçları ile
ensemble belirsizliğini birbirine karıştırmayın.

## Nadir sınıflardaki bulgu tamamen kaybolmadı

Eğitimde en az görülen 7 sınıfta ensemble ortalama recall:
**GCHM-sqrt 0.5098, XGBoost-sqrt 0.4705**. Buna karşılık ortalama precision
0.3988 / 0.4109, toplam yanlış pozitif 341 / 289. Bu grup destekleri eğitimden
belirlendi; eski validation n≤100 grubu yalnız keşifsel ek analiz olarak tutuldu.

Sepsis (40 vaka), aynı sqrt ensemble karşılaştırması:

| | GCHM | XGBoost |
|---|---:|---:|
| Recall | 0.525 | 0.375 |
| Precision | 0.420 | 0.341 |
| TP / FP / FN | 21 / 29 / 19 | 15 / 29 / 25 |

Sepsis recall farkı +0.150; noktasal koşullu %95 aralık [0.025, 0.300]. Bu aralık
çoklu karşılaştırma veya model seçim düzeltmesi içermez; 40 vakalık keşifsel
validation bulgusudur. Önceki 0.600 recall ve +0.275 farkı burada geçerli değildir:
o analiz balanced-accuracy seçimi ve ağırlıksız tek XGBoost kullanıyordu.

## Kanıtlar ve sınırlar

- Protokol: `comparison/standardized/matched_gchm_xgb_v1/protocol.json`
- Koşular: `comparison/standardized/matched_gchm_xgb_v1/runs/`
- Tam rapor: `comparison/standardized/matched_gchm_xgb_v1/reports/report.md`
- Makine-okunur bütün sınıflar, seed'ler, gruplar, aralıklar ve hash kayıtları:
  `comparison/standardized/matched_gchm_xgb_v1/reports/report.json`
- Altı XGBoost modeli kaydedilip yeniden yüklenerek validation olasılıkları birebir
  yeniden üretildi. Her koşul 1200 turunu tamamladı; seçilmiş ve son model saklandı.
- 2.000 tekrarlı, gerçek sınıfa göre tabakalı eşleştirilmiş hasta bootstrap'ı;
  seçilmiş checkpoint'lere ve gözlenen sınıf desteklerine koşulludur. Seed SS ayrı
  raporlanır; bootstrap eğitim veya model-seçim belirsizliğini kapsamaz.
- Klinik erken tahmin için tarihçe, laboratuvar ve vital alanlarının zaman güvenliği
  hâlâ çözülmemiştir. Bu tur eski girdiyi sessizce temizlemedi. Test fold'u açılmadı;
  test açmak zaman sızıntısını kendiliğinden gidermez. Klinik maliyet analizi, zaman
  güvenli veri sürümü ve harici değerlendirme ayrı çalışma gerektirir.
- Concept-dropout adayı ve diğer GNN'ler bu onaylı altı koşulun kapsamı dışındadır.

## Çalıştırma ve doğrulama

```bash
python3 -u -m comparison.standardized.matched_gchm_xgb_v1.runner --all --execute
# Yalnız tamamlanan koşulları hash/replay ile yeniden doğrula; yeniden eğitmez:
python3 -m comparison.standardized.matched_gchm_xgb_v1.runner --all --execute --resume
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -m comparison.standardized.matched_gchm_xgb_v1.report
```

Mevcut raporun üzerine yazmak için rapor komutunda açık `--overwrite` gerekir.
Eksik/başarısız eğitim klasörlerinin üzerine otomatik yazılmaz; `--resume` yalnız
biten koşulları doğrular. Bu kısa XGBoost koşulları için ara-tur optimizer/RNG resume
uygulanmadı; yeni çıktı dizininde yeniden başlatma ayrı ve açık bir işlemdir.

Birleşik doğrulama: **44 passed in 4.12s**. Komut:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -c 'import torch; torch.set_num_threads(2); import pytest; raise SystemExit(pytest.main(["-q", "-o", "faulthandler_timeout=20", "tests/test_matched_gchm_xgb_runner.py", "tests/test_matched_gchm_xgb_report.py", "tests/test_xgboost_native_baseline.py"]))'
```

İlk birleşik test komutları 180 saniyede zaman aşımına uğradı; ayrı testler geçti.
Aynı birleşik dosyalar, testlerden önce PyTorch thread sayısı 2 olarak açıkça
ayarlandığında tamamlandı. Bu bir gözlemdir; alttaki runtime kilitlenmesinin
kök nedeni kanıtlanmış değildir. Üretim eğitimleri bu test timeout'larından bağımsız
tamamlandı. `git diff --check` temiz.
