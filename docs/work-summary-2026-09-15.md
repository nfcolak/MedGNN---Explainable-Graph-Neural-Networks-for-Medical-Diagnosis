# MedGNN Work Summary — 2026-09-15

## Kısa özet

- Repo temizlendi; kullanılmayan thesis/proposal/eski deney çıktıları ve legacy scriptler kaldırıldı.
- ProtGNN, GSAT, GraphCare, PNA ve PNA-interaction için tek standardized benchmark hattı kuruldu.
- Tüm yöntemler ProtGNN/GSAT referansındaki aynı native veriye bağlandı.
- Native ortak veri kontratı: 74,511 hasta; 59,607 train / 7,448 validation / 7,456 test; 30 sınıf; 331 native graph feature.
- PNA tabanlı yeni yöntem ve medication–complaint interaction bloğu kuruldu.
- Zengin hasta girdileri için ayrı opt-in enriched path kuruldu; ancak ana eşit karşılaştırmada native ProtGNN/GSAT verisi referans alındı.
- GraphXAI 500 hasta kapsamlı açıklama üretimi tamamlandı.
- XGBoost tabular baseline eklendi ve aynı native artifact üzerinden çalıştırıldı.
- Yeni hal GitHub’a gönderildi ve repo default branch’i yeni branch’e alındı.

## Ana sonuçlar — validation, test split kullanılmadı

| Yöntem | En iyi epoch/iter | Val macro-F1 | Accuracy | Balanced acc |
|---|---:|---:|---:|---:|
| XGBoost | 1198 | 0.581207 | 0.644468 | 0.568527 |
| GSAT | 22 | 0.563520 | 0.631176 | 0.563847 |
| PNA-interaction | 17 | 0.559393 | 0.626208 | 0.551905 |
| PNA | 7 | 0.557298 | 0.624060 | 0.545837 |
| GraphCare | 15 | 0.554693 | 0.627282 | 0.541750 |
| ProtGNN | 20 | 0.543735 | 0.612111 | 0.532598 |

## Önemli yorumlar

- XGBoost en yüksek validation skorunu verdi; fakat GNN değil, tabular baseline.
- PNA-interaction, plain PNA’dan biraz yüksek çıktı ama GSAT ve XGBoost’un altında kaldı.
- Eşit veri kullanımı artık hard-guard ile korunuyor; farklı split/input/contract mismatch olursa runner fail ediyor.
- Test split hâlâ korunuyor; final model seçimi için ayrıca test değerlendirmesi yapılmadı.
- Parametre optimizasyonu yapılmadı; XGBoost tek makul baseline ayarıyla çalıştı.

## Yeni ana komutlar

```bash
python3 -m comparison.standardized.train_identical --output comparison/standardized/native_runs/full_v1 --epochs 30 --batch-size 128 --seed 1234 --loss ce --execute
```

```bash
python3 -m comparison.standardized.xgboost_native_baseline --output comparison/standardized/native_runs/xgboost_v1 --seed 1234 --n-estimators 1200 --learning-rate 0.03 --max-depth 4 --early-stopping-rounds 50 --execute
```

## GitHub durumu

- Repo: https://github.com/nfcolak/MedGNN---Explainable-Graph-Neural-Networks-for-Medical-Diagnosis
- Branch: `new_repo_structure_and_new_node_to_node_edge_trials`
- Commit: `6c84fd41 Add standardized MedGNN benchmark pipeline`
- Default branch: `new_repo_structure_and_new_node_to_node_edge_trials`
- Main force-push edilmedi.

## Doğrulamalar

- Full test suite: 448 passed, 4 skipped.
- XGBoost + native focused tests: 22 passed, 2 skipped.
- Secret scan: 0 hit.
- `git diff --check`: temiz.
- Medikal data, native input artifact, checkpoint ve training run çıktıları Git’e eklenmedi; `.gitignore` ile dışarıda bırakıldı.

## Kalan mantıklı işler

1. XGBoost için küçük hyperparameter search yapılabilir.
2. En iyi validation model seçildikten sonra tek seferlik test evaluation yapılabilir.
3. XGBoost’u tezde GNN sonucu değil, güçlü tabular baseline olarak konumlamak gerekir.
4. PNA-interaction için neden GSAT/XGBoost’u geçemediği class-level error analiziyle incelenebilir.
5. Native input temporal/provenance sınırlamaları temiz veri iddiası yapılmadan raporda açık kalmalı.
