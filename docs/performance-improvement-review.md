# MedGNN: kanıta dayalı performans iyileştirme incelemesi

## Sonuç

**Altı yeni, izole eğitim deneyi ve dört lojistik regresyon koşulu gerçekten tamamlandı.** Yeni test değerlendirmesi yapılmadı; seçim yalnızca validation macro-F1 üzerinden yapıldı. Özgün 18 koşul, veri, split ve eski cache’ler korundu.

En önemli bulgu bir hiperparametre değil: **eşit düğüm/kenar topolojisi, eşit girdi demek değil.** ProtGNN/GSAT hasta hub’ında GraphCare’in görmediği 132 ek kanal taşıyor. Bunların arasında ziyaretin tamamından laboratuvar/vital özetleri ve zamana göre kesilmemiş toplam ziyaret sayısı var. Dolayısıyla mevcut üç-yöntem tablosu mimarilerin eşit bilgi altında üstünlüğünü kanıtlamıyor.

## 1. Özgün sonuçlar: değiştirilmeden yeniden hesaplandı

Her satır aynı **7.456 test hastası**, üç sabit seed (1234/1235/1236) ortalamasıdır. Eğitim/validation sayıları 59.607/7.448, sınıf sayısı 30. Ortalama metrikler doğrudan 18 `completed` run manifest’inin işaret ettiği JSON’lardan hesaplandı; aşağıdakiler yeni test çalıştırması değildir.

| Yöntem/topoloji | Test macro-F1 | Test accuracy | Test balanced accuracy | Test top-3 |
|---|---|---|---|---|
| graphcare/cooccur | 0.454953 | 0.566658 | 0.429087 | 0.762026 |
| graphcare/star | 0.454893 | 0.567373 | 0.429243 | 0.758763 |
| gsat/cooccur | 0.549610 | 0.621602 | 0.548676 | 0.818670 |
| gsat/star | 0.544321 | 0.618205 | 0.536402 | 0.816971 |
| protgnn/cooccur | 0.495493 | 0.547076 | 0.563328 | 0.742802 |
| protgnn/star | 0.495749 | 0.543723 | 0.567901 | 0.737348 |

Kanıt: `audit/original_means.json`; özgün dizin `comparison/standardized/results/`.

## 2. Girdi ve zaman uygunluğu denetimi

| Alan | PyG/ProtGNN/GSAT | GraphCare | Bilimsel yorum |
|---|---|---|---|
| İlaç + chief complaint | 104 ev-ilacı + 88 complaint, train-fit sözlük | Aynı üyelik | Ortak 192 bilgi sütunu; lojistik baseline bunları kullanıyor |
| Demografi | Hasta hub’ında 11 sütun | Yok | Girdi eşitliği yok |
| Hasta sayısal blok | Hub’da 121 sütun | Yok | 52 lab, 30 hx, 17 medclass, 18 vital-trend; ayrıca age/bmi/n_ed_visits/n_medications |
| Vital düğümleri | Yok | Yok | “Vital node yok” ifadesi hub’daki vital özetlerini dışlamıyor |
| disease/symptom/ICD/disposition/LOS | Standart model girdisinden dışlanmış | Dışlanmış | Doğrudan hedef sızıntısına karşı doğru koruma |
| ED-dispensed pyx | Mevcut kaynakta 0 sütun | 0 sütun | Bu dosyada pyx kaynaklı ek sinyal yok |

- `protgnn_analysis/load_dataset.py:593–639, 864–869` vital düğümlerini dışlarken sayısal hub bloğunu koruyor. `graphcare_analysis/adapter.py:175–226` yalnız concept presence ve sabit hub kullanıyor. Önceki graph-parity raporu üyelik/topoloji doğruluyor, sayısal özellik eşitliği değil.
- Cache preprocessing provenansı **train fold 0 / 59.607 kişi**. 121 sayısal sütunun ortalama ve std değerleri kaynak CSV’nin yalnız canonical train satırlarında yeniden hesaplandı: maksimum fark **0.0 / 0.0**. Model-fit aşamasında heldout ölçekleme bulunmadı.
- `shared/data_prep/merge_ed.py:29–32`: `n_ed_visits`, yorumdaki “prior” iddiasına rağmen bütün ziyaretlerin sayısı; hedef ziyaret zamanına kadar kesilmiyor. Canonical kohortta **41.290** kişinin toplam sayısı birden büyük. Bunların kaçında seçilen ziyaretin sonrasından bilgi geldiği ayrıca zaman kayıtlarıyla ölçülmedi; hepsinin sızıntılı olduğu iddia edilmiyor.
- `shared/data_prep/extract_ed_labs.py:5–13` lab değerlerini **[intime, outtime]** penceresinin tamamında ortalıyor. `merge_ed.py:74–103` vital min/max/std için bütün stay’i topluyor. Bu özellikler triage/erken tahmin için uygunluğu kanıtlanmış girdiler değil. “Tanı sırasında kullanılabilir” yorumu, ölçüm/result zamanı ile gerçek karar cutoff’unu doğrulamıyor.
- `merge_ed.py:303–343`: hx, önceki intake sırasıyla oluşturuluyor; eşit intime ve önceki ziyaretin tanısının henüz bilinmiyor olması için ayrıca availability guard yok. Üst-30 history sözlüğü de bütün kaynakta seçiliyor. BMI join’i backward olsa da gün düzeyinde chartdate, tam ölçüm saatini kanıtlamıyor.
- `merge_ed.py:615–629, 653–670`: top-300 ilaç seçimi, ilaç satır filtresi ve complaint prevalence filtresi **canonical split’ten önce bütün kaynak dağılımıyla** yapılıyor. Train-only cache provenansı bunu geri alamaz. Bu çalışma sabit kaynak-snapshot karşılaştırmasıdır; ham veriden son modele bütünüyle train-only/temporally-clean pipeline iddiası değildir.
- Ev ilaçları + triage complaint mevcut ortak, hedef-dışı bilgi kümesi olarak kullanıldı. Medrecon’un saat bazında triage’da hazır olduğu ayrıca doğrulanmadı; uygun kullanım çerçevesi ilaç öyküsünün alındığı aşamadır.

Kaynakta 77.697 satır var; sadece canonical **74.511** satır analiz/fit kapsamına alındı. İlk aggregate probe bütün kaynak satırlarının canonical olduğunu varsayan assertion’da durdu; son probe split üyeliğini uygulayarak düzeltildi. Kaynak/split değiştirilmedi. Ayrıntı: `input_audit.json`, `input_audit.py`.

## 3. Grafik bilgi kapsaması ve hata profili

Her iki topolojide 74.511 graph korundu. Ortalama concept sayısı **4.952**. **21,596** graph retained ilaçsız, **8,612** complaint’siz, **669** yalnız hub. Train/validation/test hub-only sayıları önceki audit’le uyumlu 552/59/58.

300 kaynak ilaç sütununun 104’ü train prevalence ≥%1 eşiğini geçiyor; canonical aktif ilaç üyeliklerinin **%78.88** kadarı korunuyor. **2038** kişi bu filtre nedeniyle retained ilaçsız hale geliyor. Eşiği düşürmenin yararı henüz deneyle gösterilmedi.

Cooccur, star’a göre yalnız **12,325/74,511 (%16.54)** graph’ı değiştiriyor; toplam **39,098** ek yönlü kenar var. Bu, topoloji farkının kapsamasını sınırlar; küçük metrik farklarının tek sebebinin bu olduğu ileri sürülmüyor.

Özgün **star/seed_1234** checkpoint’leri yeniden yüklenerek bütün **7.448 validation** hastasında gerçek prediction alındı. Seed/topoloji sonuçlara bakılarak seçilmedi; bütün yöntemler için aynı sabit star/1234 audit politikası kullanıldı.

| Yöntem | Validation macro-F1 | Accuracy | Balanced accuracy |
|---|---|---|---|
| protgnn | 0.490696 | 0.538131 | 0.567416 |
| gsat | 0.558853 | 0.625000 | 0.558748 |
| graphcare | 0.464285 | 0.578545 | 0.437630 |

En zayıf sınıflar (her yöntem için en düşük üç validation F1; destek bu sınıftaki validation hasta sayısı):

| Yöntem | Sınıf | Destek | Precision | Recall | F1 |
|---|---|---|---|---|---|
| protgnn | Dehydration | 128 | 0.168 | 0.227 | 0.193 |
| protgnn | Multiple fractures of ribs | 70 | 0.148 | 0.557 | 0.234 |
| protgnn | Hypotension | 48 | 0.255 | 0.292 | 0.272 |
| gsat | Acute upper respiratory infection | 82 | 0.295 | 0.220 | 0.252 |
| gsat | Dehydration | 128 | 0.305 | 0.227 | 0.260 |
| gsat | Multiple fractures of ribs | 70 | 0.356 | 0.300 | 0.326 |
| graphcare | Sepsis | 40 | 0.000 | 0.000 | 0.000 |
| graphcare | End stage renal disease | 43 | 0.143 | 0.023 | 0.040 |
| graphcare | Hypokalemia | 126 | 0.472 | 0.135 | 0.210 |

Dengesizlik salt sınıf sayısına indirgenmiyor: train en büyük/en küçük sınıf oranı **22.435**. ProtGNN’de rib-fracture recall yüksek ama precision düşük (yanlış pozitif aşırılığı); GraphCare’de sepsis recall sıfır, ESRD çok düşük. Karekök inverse-frequency ağırlıklarını deneme gerekçesi: ProtGNN’in tam ters-frekans ağırlığını yumuşatırken diğer iki yöntemin azınlık kaybını orta kuvvette artırmak. Aynı ilaç/complaint’in farklı hastalıklarda görülmesi de ayrımı kısıtlayabilir; bu nedensel olarak kanıtlanmış değil.

Öne çıkan confusion örnekleri: ProtGNN’de head injury → rib fractures **86**; GSAT’ta skin infection → limb injury/pain **90**; GraphCare’de aynı karışıklık **112**. Bütün 30 sınıfın precision/recall/F1/support ve 30×30 confusion matrisleri checkpoint audit klasörlerinde mevcut.

Altgrup accuracy (örtüşen gruplar; sınıf dağılımları farklı olduğu için nedensel etki değildir):

| Grup | Validation n | ProtGNN | GSAT | GraphCare |
|---|---|---|---|---|
| hub_only | 59 | 0.2034 | 0.2712 | 0.0847 |
| no_cc | 890 | 0.2843 | 0.3674 | 0.2629 |
| no_med | 2190 | 0.6548 | 0.7183 | 0.6749 |
| concepts_1_to_2 | 2799 | 0.6417 | 0.7078 | 0.6660 |
| concepts_3_plus | 4590 | 0.4793 | 0.5791 | 0.5316 |

Az concept her zaman daha kötü değil; 1–2 concept’li grubun accuracy’si daha yüksek. **Complaint yokluğu** daha güçlü bir hata-kapsama işareti. Hub-only grupta GraphCare’in gerçek klinik payload’ı yokken PyG sayısal/demografik bilgiyi koruyor. Bu gruplarda dahi aynı bilgi varsayımı yapılamaz. Grup hizalaması source/cache sırası ve prediction label sırası ile doğrulandı; yalnız sıralama hash’i kaydedildi, hasta kimlikleri yazdırılmadı.

Kanıt: `audit/coverage.json` (30 sınıfın tamamı), `class_findings.json`, `coverage_error_groups.json`, `checkpoint_audit/*/current/{class_report,confusion,state}.json`.

## 4. Gerçek lojistik regresyon baseline

- Canonical train **59.607**, validation **7.448**; test kullanılmadı.
- `LogisticRegression(C=1, solver=lbfgs, max_iter=150, tol=1e-4, random_state=1234)`.
- Concept view: aynı train-fitted 104 ilaç + 88 complaint multi-hot; graph yapısı olmadan aynı ortak bilgi. Ek fit/normalizasyon yok.
- Native diagnostic view: bu 192 sütun + cache’teki 132 hasta-hub sütunu. Hub ölçeklemesi zaten train-only; **timing uygunluğu ve GraphCare ile eşitlik iddiası yok**. Bu ikinci view yalnız model kapasitesi teşhisi.
- Her view’da mevcut unweighted CE ve train-only `sqrt(N/(30*n_c))` ağırlıkları; C/seed/test üzerinden arama yapılmadı.

| Girdi | Ağırlık | Validation macro-F1 | Accuracy | Balanced accuracy | Solver iter. |
|---|---|---|---|---|---|
| concepts | none | 0.472141 | 0.573845 | 0.451099 | 107 |
| concepts | sqrt_inverse | 0.467906 | 0.559076 | 0.489468 | 92 |
| native_diagnostic | none | 0.552493 | 0.624731 | 0.543565 | 150 |
| native_diagnostic | sqrt_inverse | 0.546886 | 0.608754 | 0.582459 | 150 |

Ortak-input, ağırlıksız lojistik baseline ile özgün GraphCare star/1234 validation farkı **+0.007856 macro-F1**. Native diagnostic lojistik regresyon, özgün GSAT star/1234’ten **-0.006360**, ProtGNN’den **+0.061797** farklı. Bunlar tek validation split’inde tanısal karşılaştırmalar; test kazanımı veya istatistiksel üstünlük değil.

İki concept model yakınsadı. İki native model 150 iterasyon sınırına ulaştı ve **ConvergenceWarning** verdi; uyarılar `baseline/results.json` içine kaydedildi. Sınır sonradan genişletilmedi. Dört modelin joblib checkpoint’i yeniden yüklenip validation prediction’ları tekrar hesaplandı; kaydedilen metriklerle eşleşti.

## 5. Eşit ve sınırlı validation deneyleri

**3 yöntem × 2 loss konfigürasyonu; her biri tam 3 epoch, seed 1234, star, batch 128, bütün canonical train/validation.** Her koşul **1398 optimizer update** bütçesine sahip. Duvar saati/parametre sayısı eşit değil. Test loader’ları mevcut factory tarafından oluşturulsa da test üzerinde forward/selection yapılmadı.

Her yöntemde kendi mevcut LR/weight decay/architecture sabit; değişken yalnız classification-loss ağırlığı. GSAT’ın kendi information loss’u, ProtGNN cluster/separation/L1 ve warm-up mantığı korunuyor. Bu ilk **loss/imbalance hiperparametre pilotu**; ayrıca LR/dropout araması yapılmadı. Ağır full 18 matrix başlatılmadı.

| Yöntem | Loss | Seçilen epoch | Val macro-F1 | Val accuracy | Val balanced acc | Parametre |
|---|---|---|---|---|---|---|
| protgnn | current | 3 | 0.391863 | 0.430317 | 0.467537 | 99946 |
| protgnn | sqrt_inverse | 3 | 0.469212 | 0.559748 | 0.477097 | 99946 |
| gsat | current | 3 | 0.494607 | 0.604995 | 0.490594 | 212258 |
| gsat | sqrt_inverse | 2 | 0.484011 | 0.567535 | 0.528859 | 212258 |
| graphcare | current | 3 | 0.426444 | 0.556794 | 0.405312 | 158120 |
| graphcare | sqrt_inverse | 3 | 0.458147 | 0.552632 | 0.471886 | 158120 |

Aynı yöntem/iç bütçe içindeki karekök ağırlık farkları:

| Yöntem | Δ macro-F1 | Δ accuracy | Δ balanced accuracy |
|---|---|---|---|
| protgnn | +0.077349 | +0.129431 | +0.009560 |
| gsat | -0.010596 | -0.037460 | +0.038265 |
| graphcare | +0.031703 | -0.004162 | +0.066574 |

Yorum ve sınırlar:
- ProtGNN ve GraphCare’de orta şiddetli sınıf ağırlıkları bu kısa bütçede macro-F1’i artırdı. GSAT’ta balanced accuracy artarken macro-F1/accuracy düştü; bu pilot GSAT’a aynı ağırlığı taşımayı desteklemiyor.
- ProtGNN 3 epoch’un tamamı özgün **10-epoch warm-up** içinde. Prototype projection epoch 20’de başlayacağı için hiç çalışmadı. Dolayısıyla sonuç **erken optimizasyon pilotu**, tam ProtGNN yaşam döngüsünün nihai HPO sonucu değil.
- GSAT’ın r-curriculum değişimi epoch 10’da; 3 epoch boyunca r=0.9. Uzun eğitimdeki trade-off test edilmedi.
- GraphCare’in orijinal batch’i 32; eşit update bütçesi için iki yeni koşulda da 128 kullanıldı. “current” burada **current loss**, özgün eğitim koşulunun tam kopyası değil. Loss etkisi iki batch-128 koşulu arasında izole; batch değişiminin tek başına etkisi ölçülmedi.
- Yöntemler hâlâ native girdilerini kullanıyor. Bu altı koşul, yeni bir **eşit-input yöntem sıralaması** oluşturmaz; yalnız aynı yöntemde sınırlı loss karşılaştırmasıdır.
- Beş koşulun seçilen checkpoint’i 3. epoch’ta; GSAT karekök koşulunda 2. epoch en iyi (3. epoch macro-F1’i düştü). Çoğu koşul bütçe sonunda hâlâ iyileşiyor. Sonuçlar doygun eğitim, güven aralığı, yeni test başarısı veya incumbent’in yenildiği iddiasını desteklemiyor.

Her koşul `state.json`, `history.json`, `best.pt`, `last.pt`, validation predictions, class report ve confusion matrix üretti. `last.pt` model+optimizer+epoch içerir; bu harness otomatik devam ettirme sunmaz. Tamamlanan dizinleri yeniden kullanmayı reddeder. Yeni koşullar yalnız review dizinindedir; benchmark runner’ın sabit/occupied results koruması gevşetilmedi.

## 6. Önerilen öncelik

1. **Önce bilimsel input/cutoff sözleşmesini düzeltin.** Aynı 192 ortak concept ile üç yöntemi karşılaştırın veya GraphCare’e de aynı zamanca uygun tabular hub bilgisini açıkça ekleyen ayrı deney tanımlayın. n_ed_visits için gerçek prior count; lab/vital/hx için karar anında availability şartı gerekir. Yeni ham-veri rebuild’i bu çalışmada yapılmadı.
2. **Ortak-input lojistik baseline’ı sonuç tablosunda tutun.** GraphCare’in daha karmaşık modelinin bu baseline’a üstünlüğü mevcut validation verisinde gösterilmedi. Native LR diagnostik olarak GSAT’a yakın; GNN kazancının açıklanması gerekli.
3. **Sonraki kontrollü uzun-budget adayları:** ProtGNN/GraphCare karekök ağırlık; GSAT mevcut CE. Önce input uygunluğu çözülmeli. ProtGNN için warm-up/projection’ı kapsayan önceden sabitlenen eşit bütçe gerekir. Seed seçimi/test geri-beslemesi yok.
4. Complaint eksikliğini kaynak/timing kalitesi ve sınıf kompozisyonuyla inceleyin; daha çok kenar veya daha çok nadir ilaç eklemeyi otomatik çözüm saymayın. Cooccur değişiminin kapsaması sınırlı; yeni yapı eklemeden önce train-only ablation gerekir.

## 7. Doğrulama ve koruma

- `verification.json`: **158 özgün dosyanın SHA-256 ve boyutu birebir aynı**; özgün sonuçlar/checkpoint’ler, tüm eski graph cache’leri, kaynak ve split dahil.
- Altı yeni `completed` trial, her birinde üç history satırı, loadable best checkpoint, 7.448 validation prediction. Kaydedilen prediction’lardan metrikler bağımsız yeniden hesaplandı ve eşleşti.
- Dört LR checkpoint yeniden yüklenip validation yeniden tahmin edildi. Üç özgün checkpoint audit’inin metrikleri de prediction dosyalarından yeniden hesaplandı.
- `focused_tests_final.log`: **7 passed**. Baseline’ın train-only fit/validation-only prediction davranışı, occupied-output reddi, bütçe sınırı, class weights, feature view, validation selection ve atomic JSON yazımı korunuyor. Neural pilotların train/validation-only kontrol akışı ayrıca kod ve ürettikleri fold/metric artefact’leri üzerinden denetlendi; bu baseline testi neural trainer’ların kapsamlı integration testi olarak sunulmuyor. Train-fit maskesi kaldırılan geçici mutant, regression test tarafından yakalandı (`removal_probe.json`); üretim dosyası değiştirilmedi.
- `maintained_tests.log`: **361 passed, 14 warnings, 95.16s**. `.venv-graphcare` gerçek model testleri: **8 passed, 1.90s** (`graphcare_tests.log`). Uyarılar mevcut LibreSSL/matplotlib/PyG deprecation uyarıları; convergence uyarıları ayrıca baseline’da açıkça raporlandı.
- İlk ek driver testi yerel test subclass’ı pickle edilemediği için başarısızdı; test gözlemleme yöntemi düzeltildi, son 7 test geçti. İlk dört pure helper testi eksik modülle kırmızı, implementasyon sonrası yeşildi. Tek-seferlik evidence script’leri için gerçek veri yürütmesi ve regression/removal probe uygulandı; her satır için test-first iddiası yok.
- Çalışma ağacındaki eski dirty işler korunarak yalnız yeni review kodu/test/artefact/report eklendi. Commit/push, cache rebuild veya kaynak düzenlemesi yapılmadı. Oturumdaki anahtar ML review dersi ayrıca `ml-experiment-review` skill’ine eklendi: node parity ile hub-payload/upstream-source eligibility ayrı denetlenmeli.

## 8. Tam çalıştırma komutları

Tüm komutların çalışma dizini:
`/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN`.
Ana interpreter `python3`; GraphCare `.venv-graphcare/bin/python3`. Başlangıçtaki SHA snapshot `preservation_before.json`; sondaki doğrulayıcı bunun birebir eşitliğini doğrular.

```bash
# İlk analiz; hedef mevcutsa tekrar çalıştırmaz.
PYTHONPATH=. python3 -u comparison/standardized/performance_review_20260913/evidence.py audit > comparison/standardized/performance_review_20260913/audit.log 2>&1

# Var olan checkpoint'ler, tüm validation fold; sırayla && ile yürütüldü.
PYTHONPATH=. python3 -u comparison/standardized/performance_review_20260913/evidence.py checkpoint --method protgnn > comparison/standardized/performance_review_20260913/checkpoint_protgnn.log 2>&1
PYTHONPATH=. python3 -u comparison/standardized/performance_review_20260913/evidence.py checkpoint --method gsat > comparison/standardized/performance_review_20260913/checkpoint_gsat.log 2>&1
PYTHONPATH=. .venv-graphcare/bin/python3 -u comparison/standardized/performance_review_20260913/evidence.py checkpoint --method graphcare > comparison/standardized/performance_review_20260913/checkpoint_graphcare.log 2>&1

OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=. python3 -u comparison/standardized/performance_review_20260913/evidence.py baseline > comparison/standardized/performance_review_20260913/baseline.log 2>&1

# Altı koşul; herhangi bir child başarısızsa zincir durur.
for method in protgnn gsat graphcare; do
  if [ "$method" = graphcare ]; then interpreter=.venv-graphcare/bin/python3; else interpreter=python3; fi
  for variant in current sqrt_inverse; do
    PYTHONPATH=. "$interpreter" -u comparison/standardized/performance_review_20260913/evidence.py train --method "$method" --variant "$variant" --epochs 3 > "comparison/standardized/performance_review_20260913/train_${method}_${variant}.log" 2>&1 || exit $?
  done
done

PYTHONPATH=. python3 -u comparison/standardized/performance_review_20260913/input_audit.py > comparison/standardized/performance_review_20260913/input_audit.log 2>&1
python3 -m pytest tests/test_performance_review.py tests/test_performance_evidence.py -q > comparison/standardized/performance_review_20260913/focused_tests_final.log 2>&1
python3 -m pytest tests/ -q > comparison/standardized/performance_review_20260913/maintained_tests.log 2>&1
.venv-graphcare/bin/python3 -m pytest graphcare_analysis/test_zero_concept_model.py -q > comparison/standardized/performance_review_20260913/graphcare_tests.log 2>&1
PYTHONPATH=. python3 -u comparison/standardized/performance_review_20260913/verify_results.py > comparison/standardized/performance_review_20260913/verification.log 2>&1
PYTHONPATH=. python3 comparison/standardized/performance_review_20260913/make_report.py
python3 -m compileall -q comparison/standardized/performance_review.py comparison/standardized/performance_review_20260913 tests/test_performance_review.py tests/test_performance_evidence.py
git diff --check
```

Uzun komutlar background+notify ile başlatıldı; runtime’ın `exit code None` bildirimleri tamamlanma sayılmadı. OS process, log, state, history, checkpoint ve yeniden hesaplanmış metrikler ayrı doğrulandı. Yeni eğitim için hedef dizinleri silmek/üzerine yazmak yerine ayrı, açıkça adlandırılmış yeni evidence root gerekir; bu tarihli root tarihsel kayıt olarak korunmalıdır.

## 9. Artefact haritası

Kök: `comparison/standardized/performance_review_20260913/`

- `audit/{original_means,coverage}.json`, `input_audit.json`: özgün sonuçlar, 30 sınıf kapsaması, train-only scaling doğrulaması.
- `audit/features_no_identifiers.npz`: cache’ten türetilmiş, kimlik sütunu olmayan yerel feature/label/fold matrisi; yine de klinik veri olduğundan paylaşım öncesi mahremiyet politikası geçerli.
- `checkpoint_audit/{protgnn,gsat,graphcare}/current/`: üç özgün checkpoint’in gerçek validation hata profili.
- `baseline/`: dört fit edilmiş lojistik model, convergence uyarıları ve validation metrikleri.
- `tuning/{protgnn,gsat,graphcare}/{current,sqrt_inverse}/`: altı yeni sınırlı eğitim; özgün results root’una yazılmadı.
- `experiment_plan_and_state.json`: bütçe, kapsam sınırlamaları, bütün tamamlanan koşullar ve checkpoint hash’leri.
- `preservation_{before,after}.json`, `verification.json`, test/log dosyaları: koruma ve gerçek yürütme kanıtı.
- Yeni yardımcı modül: `comparison/standardized/performance_review.py`; yeni testler: `tests/test_performance_review.py`, `tests/test_performance_evidence.py`.

**Kapsam sonu:** daha fazla seed/topoloji, yeni test değerlendirmesi, LR/dropout taraması, ham veri yeniden hazırlama veya tam epoch eğitimi yapılmadı. Yapılan çalışma, gerçek bounded loss pilotu ve input/per-class teşhisidir; nihai performans iyileşmesi/test üstünlüğü henüz kanıtlanmış değildir.
