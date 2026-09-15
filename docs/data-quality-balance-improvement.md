# MedGNN — data quality + class balance improvement

## Sonuç / karar

**Dört yeni GraphCare koşulu gerçekten tamamlandı; önceki en iyi sonucu aşan bir ilerleme doğrulanmadı.** Aynı **7.448 validation** hastasında mevcut 192-concept + sqrt-inverse hâlâ en iyi: **0.481306 macro-F1**. Dengeli sampler, CE kontrole göre macro-F1'i **-0.019041**, accuracy'yi **-0.068743** değiştirdi; balanced accuracy **+0.069764** yükseldi. Bu, daha fazla azınlık recall'ı uğruna precision/accuracy kaybıdır; önceden seçilen macro-F1 hedefine göre sampler reddedildi.

Yalnız medication prevalence eşiğini %1 → %0,5 düşürmek bilgi kaybını azalttı; CE ile best-validation macro-F1 **+0.002934** arttı, fakat epoch30 farkı **-0.003214** oldu. Bu küçük, best-epoch'e bağlı bulgu kalıcı veri-temizleme başarısı değildir. Yeni eşiği mevcut sqrt loss ile birleştiren deney yapılmadı; ayrı etkiler ölçülmeden birleşik kazanım uydurulmadı.

## Kapsam ve önceden dondurulan protokol

- Mevcut GraphCare backbone/adapter; yeni mimari yok. Güvenilir mevcut CPU harness ve hub-only forward/backward testleri nedeniyle seçildi. Model ailesi/optimizer mevcut `common_input_improvement.build_model` ile aynı.
- Canonical **74.511 kişi**: train **59.607**, validation **7.448**, test **7.456**; aynı 30 sınıf, sıra, subject üyeliği ve fold korunuyor. Hiç hasta silinmedi; majority downsampling/population deletion veya label değişikliği yok.
- Star, seed **1234**, **30 tamamlanan epoch**, batch **128**, her epoch **466**, her koşul **13.980 optimizer update**. CPU, Torch threads=4, Adam lr=0.001, weight_decay=0.00001. Early stop yok; strict first maximum validation macro-F1 best checkpoint'i seçer. Epoch30 ayrıca raporlanır.
- `plan.json` eğitimden önce yazıldı: baseline current CE, baseline sqrt-inverse, baseline balanced WeightedRandomSampler + CE, med005 current CE. **4 yeni neural run**, 0 reuse, 0 test inference, 0 full18 matrix, 0 yeni HPO. %1 ve tek önceden seçilmiş düşük %0,5 eşiği dışında arama yok. Sayım denetimi sonrası ayarlar sonuçlara bakılarak genişletilmedi.
- CE/sqrt karşılaştırması aynı baseline girdide; veri karşılaştırması aynı CE ile. Baseline CE ve sqrt yeniden eğitildi, önceki skorlar kopyalanmadı. Önceki iki koşulun tüm **60 epoch** validation probability dizileriyle fark **0.0**, argmax birebir aynı (`prior_control_reproduction.json`). Bu yeni seed kanıtı değil, yeniden üretim kanıtıdır.
- Sampler yalnız train label count'tan `1/n_class`, replacement=True, num_samples=59.607; unweighted CE. Sqrt loss yalnız train sayımlarıyla `sqrt(N/(30*n_class))`. Validation/test ne resample edildi ne ağırlık/filter fit'inde kullanıldı. Test loader hiç oluşturulmadı; test özellikleri yalnız koruma/parity aggregate audit kapsamındadır.

## Gerçek veri audit'i

### Concept normalization: değişiklik yapmamak da bir bulgudur

Kaynak `data/merged_ed.csv`: **77,697** satır; canonical altküme birebir korunuyor. **300** medication sütunu binary; nonbinary hücre **0**, yinelenen medication sütun adı **0**. Snapshot'taki **88** complaint değeri arasında case/whitespace collision grubu **0**. Mevcut standardizer'a yeniden verilince idempotent olmayan vocabulary değeri **1**; otomatik tekrar normalize edilmedi.

Mevcut brand→generic sözlüğünde hem alias hem generic slug'ı ayrı snapshot medication sütunu olarak bulunan çift **0**. Bu denetim tüm klinik synonym'lerin çözüldüğünü kanıtlamaz. **Yeni doğrulanmış alias ekleme/merge: 0.** LLM clinical synonym çıkarımı yapılmadı; mevcut el-yapımı dictionary, bağımsız klinik doğrulama yerine geçmez. İlaç kombinasyonlarını tek bileşene indirgeme, complaint laterality silme ve geniş sendrom eşlemeleri clinical review adayıdır. Örnek riskler doğrudan kaynak kodundadır: `med_standardizer.py:30–36` combination-brand simplification; `chiefcomplaint_standardizer.py:35–49,66,75,98` laterality/syndrome mappings. Bunların doğru/yanlış olduğu hasta düzeyinde adjudicate edilmedi.

### Missing complaint recovery: güvenli biçimde uygulanamadı

**8,612** canonical kişide snapshot complaint yok. Bu kişilerin **8,612** tanesinde herhangi bir raw ziyarette nonempty complaint string'i bulunuyor; bu sayım clinically informative token veya seçilmiş ziyarette zamanında kullanılabilir bilgi demek değildir. **3,437** kişide tek raw ED stay, **5,175** kişide birden fazla stay var.

Merged snapshot'ta `stay_id`, `intime` ve availability cutoff yok; raw triage'da `charttime` yok. Tek raw stay bulunması bile documentation/decision cutoff'u kanıtlamaz. Subject-only join farklı ziyaret bilgisi ekleyebilir. Bu nedenle **recovered complaint=0**; raw text hiç model girdisine eklenmedi, örnek hasta/kimlik/metin dışarı yazdırılmadı. Gerekli sonraki adım selected-stay lineage ve açık clinical event/documentation/result-availability cutoff'udur; yokken “temiz complaint recovery” iddiası yapılamaz.

### Train-only ilaç prevalence varyantı

En küçük uygulanabilir veri değişikliği: mevcut globally-filtered snapshot'taki `med_* > 0` binary semantics aynı; prevalence yalnız **59.607 canonical train** satırında fit, eşik %1→%0,5. Chief-complaint sözlüğü/üyeliği, label/fold, hub ve graph recipe değişmedi. Validation/test prevalence hesaplaması yok. Yeni matrisin eski concept'lere projection'ı baseline ile birebir aynı.

| Aggregate kapsam | Baseline %1 | med005 %0,5 |
|---|---:|---:|
| Retained ilaç / kaynak ilaç | 104/300 | 189/300 |
| Complaint concept | 88 | 88 |
| Toplam clinical concept | 192 | 277 |
| Retained ilaç üyeliği / kaynak üyelik, bütün canonical kohort | 274,951/348,559 (78.88%) | 321,474/348,559 (92.23%) |
| Filtre nedeniyle bütün ilaçlarını kaybeden kişi | 2,038 | 584 |
| Aynı sayım train / val / test | 1635 / 211 / 192 | 469 / 58 / 57 |
| Hub-only train / val / test | 552 / 59 / 58 | 504 / 50 / 56 |

**1,454** kişinin en az bir ilaç concept'i geri geldi; hub-only toplam **669→610**. Bu yeni hasta yaratmak veya sorunlu hastaları çıkarmak değil, mevcut snapshot sütunlarını daha az kayıpla temsil etmektir.

Her iki girdi için bütün **74.511 graph** doğrulandı: medication/complaint binary presence → aynı categorical node ID ve onun birebir one-hot açılımı; son düğüm tek sabit hub, hasta-specific numeric/demographic/label payload yok. Star bidirectional spokes, no KG expansion; GraphCare EHR mask'inde hub=0, visit mask'inde hub=1. PyG effective one-hot ve GraphCare node/edge üyeliği birebir kontrol edildi; başka GNN eğitilmedi. Sıfır clinical concept graph'ları tutuldu ve gerçek GraphCare singleton/mixed forward/backward testinden geçti.

**Önemli capacity sınırı:** vocabulary büyüyünce mevcut GraphCare embedding/readout boyutları ve başlangıç parametre eşlemesi de değişir: **158,120→249,410** parametre. Aynı backbone/hyperparameter, tam aynı kapasite veya initialization eşliği değildir. Bu yüzden küçük med005 farkı yalnız “daha iyi cleaning”in nedensel etkisi diye sunulmuyor.

### Label consistency / duplicate conflicts

- Source duplicate subject satırı **0**; canonical missing/invalid class **0**, 30 class sırası ve önceki `y` dizisi birebir eşleşti. Kanonik kişi sayısı 74.511 ve unique subject sayısı aynı; önceki fold vektörü birebir aynı.
- Train'de identity ve diagnosis/symptom-derived sütunlar hariç bütün snapshot feature'larının tam duplicate grubu **0**; conflicting-label duplicate grubu **0**. Bu klinik etiket doğruluğunu ispatlamaz ve o zengin/unsafe feature'lar eğitime sokulmadı.
- Train binary concept signature: baseline **1,349** multi-label grup / **19,391** kişi; med005 **1,153** grup / **17,352** kişi. Exact observed-input belirsizliği azalıyor, ama hâlâ var; bunun label error olduğu veya Bayes ceiling oluşturduğu söylenemez.
- Bütün raw diagnosis tablosu (**899,050** satır): exact duplicate **0**, aynı stay/priority'de farklı ICD code conflict grubu **0**. Bunlar seçilmiş canonical encounter istatistiği değildir.
- ICD crosswalk: **14,145** source code içinde **3,380** çok-target eşleme, **3,379** en iyi approximation düzeyinde tie. Mevcut `merge_ed.py:133–141` bunlardan ilkini tutuyor. Bu kaynak-audit sayımı etkilenen canonical hasta sayısı değildir; clinical review/target adjudication gerekir.
- Mevcut **37** `DISEASE_MERGES` rule'u **10** merged target adına gider; kod yorumları önceki confusion-driven grouping'i açıkça belirtir (`merge_ed.py:208–299`). Category'nin en sık title ile adlandırılması ve bu taxonomy bağımsız klinik inceleme adayıdır. **Yeni disease merge/relabel yapılmadı.**

## Sonuçlar — tamamlanmış gerçek koşullar

Her satır aynı **n=7.448 validation**, 30 epoch / 13.980 update. Best epoch 1-based. Final checkpoint metric'i ayrı; selection test verisine bakmadı.

| Girdi | Loss / sampling | Best epoch | Macro-F1 | Accuracy | Balanced acc | Epoch30 F1 | Parametre |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | current | 13 | 0.464117 | 0.578947 | 0.443460 | 0.459858 | 158120 |
| baseline | sqrt_inverse | 17 | 0.481306 | 0.573577 | 0.486248 | 0.477841 | 158120 |
| baseline | balanced_sampler | 21 | 0.445076 | 0.510204 | 0.513224 | 0.441988 | 158120 |
| med005 | current | 21 | 0.467051 | 0.583647 | 0.445580 | 0.456644 | 249410 |

| CE baseline’a göre | Δ best macro-F1 | Δ accuracy | Δ balanced acc |
|---|---:|---:|---:|
| baseline/sqrt_inverse | +0.017189 | -0.005370 | +0.042788 |
| baseline/balanced_sampler | -0.019041 | -0.068743 | +0.069764 |
| med005/current | +0.002934 | +0.004700 | +0.002120 |

### Bütün sınıflar: precision / recall

Aynı validation supports; oranlar 3 decimal, tam precision/recall/F1/support `class_metrics.csv` içinde. CE / sqrt / sampler baseline192; med005 unweighted CE. Hiç sınıf kaldırılmadı. Yalnız iyi örnekleri seçmek yerine 30 sınıfın tamamı aşağıda.

| Class | n | CE P/R | sqrt P/R | sampler P/R | med005 CE P/R |
|---|---:|---|---|---|---|
| Acute kidney failure | 128 | 0.384/0.219 | 0.362/0.195 | 0.346/0.211 | 0.345/0.156 |
| Acute pharyngitis | 168 | 0.757/0.762 | 0.707/0.762 | 0.752/0.720 | 0.774/0.756 |
| Acute upper respiratory infection | 82 | 0.314/0.195 | 0.283/0.341 | 0.215/0.451 | 0.294/0.183 |
| Alcohol abuse with intoxication | 344 | 0.907/0.765 | 0.776/0.785 | 0.916/0.759 | 0.904/0.767 |
| Anemia | 114 | 0.667/0.333 | 0.638/0.325 | 0.421/0.351 | 0.617/0.325 |
| Anxiety disorder | 138 | 0.733/0.536 | 0.591/0.587 | 0.503/0.630 | 0.685/0.536 |
| Back or spine pain | 855 | 0.763/0.754 | 0.784/0.745 | 0.835/0.677 | 0.778/0.749 |
| Cardiovascular risk factor | 748 | 0.423/0.658 | 0.482/0.567 | 0.602/0.259 | 0.451/0.628 |
| Dehydration | 128 | 0.270/0.188 | 0.184/0.352 | 0.148/0.344 | 0.253/0.195 |
| Diabetes mellitus | 424 | 0.570/0.594 | 0.580/0.575 | 0.612/0.432 | 0.571/0.604 |
| End stage renal disease | 43 | 0.200/0.023 | 0.128/0.116 | 0.065/0.349 | 0.333/0.140 |
| Epilepsy | 97 | 0.760/0.784 | 0.760/0.814 | 0.669/0.814 | 0.712/0.814 |
| GI bleed | 302 | 0.896/0.685 | 0.916/0.685 | 0.902/0.699 | 0.903/0.675 |
| Head injury | 502 | 0.589/0.665 | 0.621/0.610 | 0.699/0.508 | 0.560/0.695 |
| Heart failure | 85 | 0.358/0.282 | 0.408/0.471 | 0.324/0.659 | 0.476/0.235 |
| Hypokalemia | 126 | 0.415/0.135 | 0.324/0.183 | 0.287/0.198 | 0.531/0.135 |
| Hypotension | 48 | 0.455/0.208 | 0.457/0.333 | 0.179/0.396 | 0.433/0.271 |
| Hypothyroidism | 64 | 0.411/0.469 | 0.341/0.484 | 0.211/0.578 | 0.397/0.359 |
| Laceration w/o fb of l idx fngr w/o damage to nail | 121 | 0.657/0.537 | 0.500/0.752 | 0.429/0.793 | 0.628/0.587 |
| Limb injury or pain | 903 | 0.574/0.721 | 0.676/0.638 | 0.752/0.549 | 0.594/0.715 |
| Lower respiratory disease | 369 | 0.409/0.496 | 0.445/0.458 | 0.539/0.225 | 0.404/0.547 |
| Major depressive disorder | 268 | 0.766/0.795 | 0.709/0.817 | 0.728/0.799 | 0.716/0.847 |
| Multiple fractures of ribs | 70 | 0.529/0.129 | 0.367/0.257 | 0.161/0.514 | 0.588/0.143 |
| Non-ST elevation (NSTEMI) myocardial infarction | 72 | 0.222/0.139 | 0.205/0.431 | 0.118/0.611 | 0.667/0.111 |
| Sepsis | 40 | 0.333/0.025 | 0.200/0.050 | 0.069/0.325 | 0.500/0.025 |
| Skin or soft-tissue infection | 389 | 0.622/0.419 | 0.527/0.478 | 0.508/0.483 | 0.649/0.442 |
| UTI or pyelonephritis | 412 | 0.444/0.442 | 0.458/0.420 | 0.568/0.325 | 0.410/0.512 |
| Unsp intestnl obst | 118 | 0.454/0.500 | 0.470/0.466 | 0.283/0.746 | 0.418/0.500 |
| Unspecified asthma with (acute) exacerbation | 112 | 0.553/0.509 | 0.556/0.491 | 0.409/0.580 | 0.638/0.393 |
| Unspecified atrial fibrillation | 178 | 0.444/0.337 | 0.441/0.399 | 0.384/0.410 | 0.452/0.320 |

Per-class tabloda sampler'ın recall/precision değişimi görünür; tek başına balanced accuracy ile aday seçilmedi. `validation_coverage.json` aynı hastalarda hub-only, no-complaint, medication-restored ve added-med-information altgruplarını ve label-count'larını saklar; farklı sınıf karışımlarındaki coverage farkları nedensel fayda diye yorumlanmaz.

## Checkpoint / resume ve doğrulama

- Her koşul `state.json=completed`; `run_manifest.json`, `best.pt`, `last.pt`, 30 history satırı ve 30 validation prediction dosyası yüklenebilir. **120 epoch** prediction dosyasından confusion-count tabanlı bağımsız NumPy metrics yeniden hesaplandı; shared metric helper sonuçlarıyla eşleşti. Bütün 120 class-report satırı ayrıca yeniden hesaplandı.
- **8 checkpoint** (her koşul best+last) yeniden yüklendi ve aynı 7.448 validation hastasında forward tekrarlandı; argmax tam aynı, en büyük probability farkı **0**. `checkpoint_replay.json`.
- Sampler gerçek eğitimde epoch1 / 466 update sonunda durduruldu; explicit `--resume` ile model/optimizer/Python/NumPy/Torch RNG ve dedicated sampler generator state restore edilerek 30 epoch tamamlandı. Epoch içi partial-batch resume değil, epoch-atomic resume. Son dedicated RNG, seed1234'ten 30 gerçek-boyutlu sampling epoch'unun yeniden üretilmesiyle tam eşleşti. `events.jsonl`, `sampler_exposure.json`. Her epoch draw/class count ve unique train subject sayısı aggregate saklandı; sampling, o epoch'ta bazı kişilerin seçilmemesi anlamına gelebilir, canonical popülasyondan silinmesi anlamına gelmez.
- Yeni train-fit ve sampler behavior regression'ları önce eksik işlev için başarısız oldu, sonra geçti. Train-only preprocessing, heldout-only concept exclusion, bütün satırların tutulması, sampler train scope/epoch length/CE policy, no-test runner scope, RNG + optimizer update resume eşliği ve binding testleri çalıştırıldı. Tek-seferlik audit/render script'lerinin her satırı için TDD iddiası yok.
- GraphCare interpreter focused suite: **24 passed**. Ana maintained suite: **370 passed, 2 skipped, 14 warnings**; skip'ler ana ortamda olmayan pyhealth, GraphCare ortamında gerçek hub-only/mixed testleri geçti. Yeni quality testleri ayrıca ana ortamda çalıştırıldı; nihai log `main_env_quality_tests_final.log`. Uyarılar mevcut LibreSSL/matplotlib deprecation. Compileall ve git diff --check geçti.
- **650 önceki dosya** size/SHA-256 başlangıçla aynı: source/split, özgün cache/results, önceki review/common-input/diagnosis artifact'leri ve snapshot'a alınan dirty/untracked çalışma. Ayrıca **5 raw dosya** supplementary-audit öncesi/sonrası hash'leri aynı. `preservation_after.json`, `raw_preservation_after.json`.
- İlk preservation verifier, yeni deneyin kendi test/log dosyalarını yanlışlıkla “önceki korunan dosya” saydığı için durdu: git relative path ile absolute `__file__` kökü karşılaştırılmıştı. Yeni regression kırmızı→yeşil; yalnız bu izole kökün resolved descendants'ı kapsamdan çıkarıldı. Özgün snapshot ve başarısız log korunuyor, istisnalar `preservation_scope_correction.json` içinde. **Hiçbir eski dosya istisna edilmedi; model/input/loss/training code değiştirilmedi, hiçbir koşul yeniden başlatılmadı.**
- Background notify talep edildi; runtime bazen `exit_code=None/exited` dediği halde OS PID ve son koşul hâlâ çalışıyordu. Bildirime güvenilmedi: state/history/checkpoint ve gerçek PID kontrol edildi, mevcut işin tamamlanması beklendi; duplicate neural run açılmadı.

## Zaman uygunluğu ve kalan bilimsel sınır

**temporal_clean=false**, **early-triage uygunluğu kanıtlanmış değil**, **raw-to-model pipeline bütünüyle train-only değil**. Snapshot'taki upstream whole-source top300 medication seçimi, medication row eligibility, complaint prevalence ve label/cohort seçimi burada geri alınmadı. Train-only eşiği, önceki upstream global filtering'i temizleyemez.

`merge_ed.py:390–400` bütün stay medication kayıtlarını cutoff olmadan toplar. Önceki korunmuş raw timing audit'inde 2.987.342 medication satırının 2.987.308'i intake sonrasında chart edilmişti; median fark 4.440 saniye. Bu bütün raw medication tablosudur, selected canonical visit availability kanıtı değildir; charttime da tek başına klinik bilgi mevcudiyeti değildir. Triaged complaint adı timestamp garantisi sayılmadı. Unsafe lab/vital/history/demographic payload yeni girdide kullanılmadı.

Bu tek seed ve validation üzerinde epoch/policy seçimi içeren source-snapshot tanı deneyidir. Multi-seed superiority, confidence interval, test kazanımı, clinical validity veya deployment önerisi değildir. Aynı koşulların birebir yeniden üretimi bağımsız seed sonucu sayılmaz. med005 genişleyen vocabulary/capacity taşır; best/last trade-off'u ve selection optimism nedeniyle kesin kazanım iddiası desteklenmez.

## Sonraki karar

1. Önceden tanımlı validation macro-F1 hedefine göre mevcut **baseline192 + sqrt-inverse** korunmalı; sampler bu bütçede tercih edilmemeli. Accuracy öncelikli ayrı kullanım kararı için trade-off ayrıca tanımlanmalı.
2. %0,5 threshold bilgi kapsamasını iyileştiriyor ama skor faydası zayıf ve epoch30'da tersine dönüyor. Otomatik production adoption veya sqrt ile birleşik kazanım ilanı yok; gerekirse sonraki ayrı onaylı matched-seed çalışmasında vocabulary-capacity etkisi de kontrol edilmeli.
3. Asıl data-quality önceliği: selected-stay lineage + açık availability cutoff, upstream train-only fit ve clinical-review destekli medication/complaint/ICD-label eşlemeleri. Bunun için hasta düşürme, disease relabel/merge veya semantik uydurma yapılmamalı.

## Artefact / komutlar

Yeni kod/veri/sonuç kökü: `comparison/standardized/data_quality_balance_20260914/`. Bu rapor dışında repo yazıları yalnız yeni izole kökte; eski dirty tree korundu. Commit/push yok.

- `audit.py`, `supplemental_audit.py`, `quality.py`, `run.py`, `verify.py`, `exposure.py`, `render_report.py`, `test_quality.py`.
- `plan.json`, `audit.json`, `supplemental_audit.json`, `baseline.npz`, `med005.npz`, `*_contract.json`.
- `trials/{baseline/{current,sqrt_inverse,balanced_sampler},med005/current}/` her koşulun checkpoint/history/prediction/class report/confusion/state/manifest dosyalarını içerir.
- `results_verified.json`, `deltas.json`, `class_metrics.csv`, `checkpoint_replay.json`, `sampler_exposure.json`, `validation_coverage.json`, `verification.json`.
- Privacy: NPZ'lerde yalnız presence/y/folds var, patient identifier yok; klinik türetilmiş veri olduğu için bu paylaşım izni anlamına gelmez. Raw text/row veya patient-map key rapor/console'a çıkarılmadı.

Repo kökünden tarihsel komutlar (occupied output'ta tekrar yeni eğitim başlatmayın):

```bash
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/audit.py
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/run.py --method graphcare --input baseline --variant balanced_sampler --execute --stop-after 1
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/run.py --method graphcare --input baseline --variant balanced_sampler --execute --resume
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/run.py --method graphcare --input baseline --variant current --execute
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/run.py --method graphcare --input baseline --variant sqrt_inverse --execute
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/run.py --method graphcare --input med005 --variant current --execute
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/supplemental_audit.py
.venv-graphcare/bin/python3 -m pytest comparison/standardized/data_quality_balance_20260914/test_quality.py tests/test_common_input_improvement.py graphcare_analysis/test_zero_concept_model.py -q
PYTHONPATH=. .venv-graphcare/bin/python3 -m comparison.standardized.data_quality_balance_20260914.verify
PYTHONPATH=. .venv-graphcare/bin/python3 comparison/standardized/data_quality_balance_20260914/exposure.py
python3 -m pytest tests/ -q
python3 -m pytest comparison/standardized/data_quality_balance_20260914/test_quality.py -q
python3 -m compileall -q comparison/standardized/data_quality_balance_20260914
git diff --check
python3 comparison/standardized/data_quality_balance_20260914/render_report.py
```
