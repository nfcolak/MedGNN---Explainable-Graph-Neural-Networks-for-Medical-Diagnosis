# Temizlik planı — MedGNN (58 GB → ~28 GB)

Tarih: 2026-09-20. **Hiçbir şey silinmedi.** Her satır bağımlılık taramasıyla
doğrulandı; "yeniden üretilebilir" diyorsam komutu da yazdım.

## A. Güvenle silinebilir — 9,3 GB

Aktif kodun hiçbiri okumuyor (grep ile doğrulandı) veya tek komutla geri gelir.

| yol | boyut | gerekçe |
|---|---:|---|
| `data/graphs/` | 6,1 GB | v1 dönemi star/cooccur cache. `clinical_graph_v2/` ve `shared/` **kullanmıyor**. Sadece `zero_concept_verification/` ve eski `protgnn/graphcare` config'leri referans veriyor. |
| `data/processed_hetero_.../` | 2,6 GB | eski işlenmiş cache, aynı durum |
| `visualizer/dist/` | 431 MB | build çıktısı → `npm run build` |
| `visualizer/node_modules/` | 109 MB | → `npm install` |
| `comparison/standardized/graphxai_execution_2026*` | 12 KB | boş/iptal koşu |

## B. Rapor korunarak silinebilir — 295 MB

Bu dizinlerin **bulguları `docs/`'ta duruyor**; içerik ham ara çıktı.

| yol | boyut | raporu |
|---|---:|---|
| `common_input_20260913/` | 119 MB | `docs/common-input-improvement.md` |
| `data_quality_balance_20260914/` | 100 MB | `docs/data-quality-balance-improvement.md` |
| `performance_review_20260913/` | 34 MB | `docs/performance-improvement-review.md` |
| `graphxai_500_20260914T144701/` | 28 MB | `docs/graphxai-500-results.md` |
| `pna_experiments/` | 14 MB | `docs/pna-*.md` (3 dosya) |

Not: `zero_concept_verification/` (208 KB) küçük, bırakılabilir.

## C. Karar gerekir — 17,4 GB (en büyük kazanç)

### C1. `first_recorded_lab_all_visits_v2/graphs.jsonl` — 17 GB

**v1 event grafı.** Teşhisi yapıldı (`docs/new-input-diagnosis.md`): graf yapısı
tam girdinin üzerine **+0,009 macro-F1** katıyor, `structure_only` çoğunluk
tabanının altında. Terk edildi.

**Kritik:** v3 bu dosyaya bağlı **değil**. Manifest'i okudum, v3'ün ihtiyacı:
- `events.sqlite` (1,4 GB) ← **KALMALI**, 18 GB'lık `labevents.csv` taraması burada
- `cohort.csv` (20 MB) ← **KALMALI**

Yeniden üretim (gerekirse): `--resume-prepared` ile, `prepared_index.json` mevcut →
18 GB tarama tekrarlanmaz, sadece materialization (~15 dk).

**Risk:** v1↔v3 graf-seviyesi karşılaştırmasını tekrar yapamazsın. Ama sayısal
sonuçlar `docs/new-input-diagnosis.md`'de ve `new_input_diagnosis_v1/results.jsonl`'de
(20 hücre, olasılık hash'leriyle) duruyor.

### C2. `native_runs/` — 887 MB, `matched_gchm_xgb_v1/` — 555 MB

GCHM checkpoint'leri + GCHM↔XGBoost karşılaştırması. Raporlar:
`docs/gchm-xgboost-matched-results.md`, `docs/gchm-concept-dropout*.md`.

**Karar noktası:** checkpoint'ler silinirse GCHM sonuçları **yeniden üretilemez**
(yeniden eğitim gerekir). Raporlar ve `report.json` (tüm sınıf/seed/bootstrap
kayıtları) duruyor. Kod `gchm_analysis/` (52 KB) korunmalı.

Öneri: `report.json` + `protocol.json` sakla, `*.pt` sil → ~1,3 GB kazanç.

### C3. `event_training/` — 80 MB

v1 event grafı eğitimleri (macro-F1 0,084, yakınsamamış). v3 bunları geçersiz
kıldı. `representative_6000_fullval_v1/` (14 MB) rapor+manifest olarak saklanabilir,
60 MB'lık `local_first_lab_v2/` silinebilir.

## D. Asla silme

| yol | boyut | neden |
|---|---:|---|
| `data/Original CSVs/` | 18 GB | **yeniden indirilemez** (PhysioNet erişimi) |
| `event_inputs/clinical_graph_v3_full/` | 5,2 GB | güncel artefakt |
| `event_inputs/..._v2/events.sqlite` + `cohort.csv` | 1,4 GB | v3'ün kaynağı, 18 GB tarama |
| `clinical_runs_v3/` | 14 MB | güncel sonuçlar |
| tüm `*_analysis/`, `shared/`, `comparison/standardized/*/*.py` | ~80 MB | kod |
| `docs/` | 828 KB | tüm bulgular |
| `_targets_*`, `_features_v1` | 109 MB | etiket sidecar'ları, küçük |

## Özet

| aşama | kazanç | risk |
|---|---:|---|
| A | 9,3 GB | yok |
| B | 295 MB | yok (raporlar var) |
| C1 (v1 graphs.jsonl) | 17 GB | düşük — resume ile geri gelir |
| C2 (GCHM checkpoint) | 1,3 GB | orta — yeniden eğitim gerekir |
| C3 | 60 MB | yok |
| **toplam** | **~28 GB** | 58 GB → 30 GB |

A+B+C1 yapılırsa: **26,6 GB kazanç, geri dönülemez kayıp yok.**

## Komutlar (onay sonrası)

```bash
cd .

# A
rm -rf data/graphs data/processed_hetero_merged_ed_noLOS_prev10_pmi2_miss_disease
rm -rf visualizer/dist visualizer/node_modules
rm -rf comparison/standardized/graphxai_execution_2026*

# B
rm -rf comparison/standardized/{common_input_20260913,data_quality_balance_20260914}
rm -rf comparison/standardized/{performance_review_20260913,graphxai_500_20260914T144701}
rm -rf comparison/standardized/pna_experiments

# C1  (events.sqlite ve cohort.csv KALIR)
rm -f comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2/graphs.jsonl

# C3
rm -rf comparison/standardized/event_training/local_first_lab_v2
```

## Ayrı konu: git'te duran büyük dosyalar

`visualizer/public/graphs/` **161 dosya git'te izleniyor**, `manifest.json` tek başına
18 MB. Bunlar repo klonlayan herkese iniyor. Silmek istemezsen bile `.gitignore`'a
alıp `git rm --cached` yapmak repo boyutunu küçültür — ama bu geçmişi değiştirmez,
ayrı bir karar.
