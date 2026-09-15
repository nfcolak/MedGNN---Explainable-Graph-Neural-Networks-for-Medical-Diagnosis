# Common-input improvement: medication-history-available source-snapshot diagnostic

## Kapsam ve bilimsel sınır

Bu aşama **early-triage veya temporally-clean bir deney değildir**. Aynı sabit kaynak-snapshot üzerinde ortak bilgi ile kontrollü validation-only tanı deneyidir. Kullanıcının onayladığı kapsam: star, seed 1234, önceden sabitlenmiş 30 epoch; ProtGNN current/sqrt_inverse, GraphCare current/sqrt_inverse, GSAT current kontrol. Yeni test değerlendirmesi ve tam 18 koşul matrisi yok. Özgün kaynak/split/cache/results ve önceki review çıktıları değiştirilmiyor.

## Ortak girdi sözleşmesi ve gerçek audit

Kaynak `data/merged_ed.csv`; canonical split `comparison/canonical_split.json`. Canonical train 59.607, validation 7.448, test 7.456; 30 sınıf. Yalnız source-snapshot'ın canonical train satırlarında yeniden fit edilmiş 104 ilaç + 88 complaint concept'i. Önceki PyG concept matrisi ile yeniden oluşturulan matris, vocabulary sırası eşleştirilerek birebir karşılaştırıldı. Train-only burada **kaynak dosyasından sonraki fit aşaması** demektir; ham veriden tüm pipeline train-only değildir.

- PyG: düğüm kimliğinin 193 boyutlu one-hot kodlaması.
- GraphCare: aynı sıralı categorical node ID; one-hot açılımı PyG `x` ile tam aynı. Öğrenilmiş embedding değerleri eşitlenmiyor: bunlar yöntem mimarisinin parçası.
- Son kanal yalnız sabit patient hub kimliği; bütün hastalarda `[0,...,0,1]`. 132 native demografik/sayısal hub kanalı tamamen yok. Ayrı node-type/scalar ekleri de yok.
- Aynı star hub-spoke kenarları; hasta dışı KG expansion, lab/vital/hx/visit-count/medclass/diagnosis/symptom/disposition girdisi yok.
- GraphCare visit membership hub'ı içerir, direct-EHR membership hub'ı içermez; bu sinyaller ortak graph'tan deterministik türetilir, ek hasta bilgisi değildir. Direct-EHR pooling ve learned encoders gibi gerçek yöntem mekanizmaları korunur.
- Bütün **74.511** graph için efektif encoder girişi ve kenarlar gerçekten karşılaştırıldı; tüm canonical hastalar korunuyor. Hub-only train/validation/test: **552/59/58**. Bu hastalar düşürülmedi, sahte clinical concept eklenmedi.
- İzole veri: `comparison/standardized/common_input_20260913/inputs_no_identifiers.npz`. Kimlik sütunu yoktur; yine de klinik türetilmiş veridir, paylaşım izni anlamına gelmez.
- Denetim/provenans/hash: `input_contract_and_audit.json` aynı kökte.

## Neden zaman uygunluğu hâlâ bloke?

`shared/data_prep/merge_ed.py:390–400` medrecon'u stay boyunca topluyor; `615–629` bütün kaynak üzerinde top300 ilaç seçimi ve satır filtresi, `653–670` bütün kaynak complaint prevalence filtresi uyguluyor. Bunları sonraki train-fit sözlüğü geri alamaz. Aynı script'in lab/vital/visit-count gibi bilinen riskli native alanları bu yeni girdi setinde hiç kullanılmıyor.

Ham dosyalar yerelde bulundu ve yalnız gerekli schema/aggregate zaman denetimi yapıldı:
- `data/Original CSVs/medrecon.csv`: `charttime` var. **2.987.342** medication satırının tamamı ED intake ile eşleşti; **2.987.308** satırın charttime'ı intake sonrasında, **34** satır intake anında/öncesinde. Median fark **4.440 saniye**. Bu sayılar bütün raw medication satırlarına aittir, canonical seçilmiş ziyaretlere değil; charttime klinik kullanılabilirliğin kesin kanıtı da değildir.
- Raw triage'da complaint var fakat charttime yok.
- Merged snapshot'ta stay_id/intime/prediction cutoff yok. Yalnız subject join yaparak seçilmiş stay veya medication availability geri yüklenemez; aynı kişide farklı ziyaretleri birleştirmek yeni leakage yaratır.
- Hastaya ait tanı karar anı ve veri availability cutoff'u ayrıca kanıtlanmadı. Dolayısıyla medication-history-available ifadesi bir kullanım çerçevesidir, her satırın o anda hazır olduğunun doğrulandığı iddiası değildir.

Timestamp-safe upstream rebuild yapılmadı. Güvenli bir sonraki çalışma, seçilmiş stay lineage'ını koruyan ham-veri dönüşümü, açık karar cutoff'u ve train-only upstream filtreleri gerektirir; eski canonical kohortu sessizce yeniden tanımlamak veya hastaları düşürmek kabul edilmedi.

## Eğitim protokolü

`plan.json` gerçek eğitimden önce kaydedildi. Her koşul 30 epoch, batch128; aynı epoch/update bütçesi, eşit süre veya model kapasitesi iddiası yok. GraphCare native batch32 ->128; iki loss koşuluna aynı değişiklik. Diğer yöntemler batch128. LR/weight decay/architecture native config; değişken yalnız belirtilen classification loss. Current: ProtGNN inverse-frequency, diğerleri CE. Sqrt inverse: yalnız train class count.

ProtGNN: ilk 10 epoch last layer frozen, sonra joint; zero-based epoch20'de native MCTS projection, her prototype için yalnız train'den 10 class-matched graph; cluster/separation/L1 ve gradient clipping korunur. GSAT: bilgi kaybı, stochastic attention ve r-curriculum korunur. Early stopping yok; 30 epoch tamamlanır. Best checkpoint yalnız validation macro-F1; ayrıca son epoch sonuçları raporlanır. Test loader oluşturulmaz.

`last.pt` model, optimizer, Python/NumPy/Torch/MPS RNG, next epoch, update sayısı, history ve best checkpoint kopyasını atomik içerir. Explicit `--resume` zorunlu; input/split/harness/adapter binding farklıysa reddedilir. File lock aynı koşulun eşzamanlı ikinci eğitimini engeller. Kesintiye uğramış kısmi epoch son tamamlanmış epoch'tan tekrar oynatılır; epoch içi minibatch resume iddiası yok. `--stop-after 1` GSAT'ta gerçek checkpoint/resume smoke için kullanıldı; toplam bütçe yine 30.

## Gerçek sonuçlar — beş koşul tamamlandı

Her satır aynı **7.448 validation hastası**, tek sabit seed 1234, star ve 30 tamamlanmış epoch. Her koşul **13.980 optimizer update**; test inference yok. Best checkpoint yalnız validation macro-F1 ile seçildi. Epoch sütunu 1-based; checkpoint içindeki epoch 0-based.

| Yöntem | Loss | Best epoch | Best val macro-F1 | Accuracy | Balanced accuracy | Epoch30 macro-F1 | Parametre |
|---|---|---:|---:|---:|---:|---:|---:|
| protgnn | current | 15 | 0.440559 | 0.506445 | 0.502145 | 0.439272 | 82282 |
| protgnn | sqrt_inverse | 11 | 0.473967 | 0.558673 | 0.491681 | 0.469846 | 82282 |
| graphcare | current | 13 | 0.464117 | 0.578947 | 0.443460 | 0.459858 | 158120 |
| graphcare | sqrt_inverse | 17 | 0.481306 | 0.573577 | 0.486248 | 0.477841 | 158120 |
| gsat | current | 25 | 0.471834 | 0.578545 | 0.452618 | 0.463407 | 194594 |

| Loss karşılaştırması (sqrt − current) | Δ best macro-F1 | Δ best accuracy | Δ best balanced accuracy | Δ epoch30 macro-F1 |
|---|---:|---:|---:|---:|
| protgnn | +0.033408 | +0.052228 | -0.010464 | +0.030574 |
| graphcare | +0.017189 | -0.005370 | +0.042788 | +0.017983 |

**Sonuç:** Karekök ağırlık iki yöntemde de hem best-validation hem epoch30 macro-F1'i artırdı. Ancak ProtGNN balanced accuracy **0.010464** düştü; GraphCare accuracy **0.005370** düştü. Dolayısıyla bütün metriklerde kazanım yok. Bu ortak-input koşulunda en yüksek seçilmiş macro-F1 GraphCare sqrt_inverse'da **0.481306**; tek-seed validation sıralaması olarak okunmalı.

Ortak 192-concept, ağırlıksız lojistik baseline: validation macro-F1 **0.472141**, n=7.448 (önceki korunmuş gerçek fit; yeni girdi matrisinin aynı concept bilgisi taşıdığı audit ile doğrulandı). Bu baseline da source-snapshot sınırlarını taşır.

### Mekanizmalar ve checkpoint durumu

- ProtGNN current: epoch20 (0-based) gerçek MCTS projection, 90 prototype × 10 train candidate; **87/90** prototype pozitif similarity ile değiştirildi. Seçilen checkpoint **projection öncesi**; son checkpoint projection sonrasıdır.
- ProtGNN sqrt_inverse: epoch20 (0-based) gerçek MCTS projection, 90 prototype × 10 train candidate; **87/90** prototype pozitif similarity ile değiştirildi. Seçilen checkpoint **projection öncesi**; son checkpoint projection sonrasıdır.
- GSAT r=0.9/0.8/0.7 curriculum dönemleri kaydedildi; information loss ve stochastic attention aktiftir.
- Her `last.pt`: next_epoch=30, updates=13.980, yüklenebilir optimizer ve RNG. Her `best.pt`: seçilen gerçek epoch ve model state. Tamamlanma `state.json=completed` ile doğrulandı.
- Learning curves: `learning_curves.csv`, `validation_learning_curves.png`, `validation_learning_curves.svg`; bütün 150 epoch prediction dosyası ve beş history korunuyor.

### Yorumun sınırı

Bu tek-seed validation tanı karşılaştırmasıdır; güven aralığı, multi-seed üstünlük, test kazanımı veya klinik kullanılabilirlik kanıtı değildir. Loss farkları ortak girdi ve sabit bütçe içinde yorumlanabilir. Önceki native-input sonuçlarla fark yalnız input etkisine bağlanamaz; girdi kodlaması ve eğitim bütçesi de farklıdır. Validation üzerinde epoch/loss seçimi validation iyimserliği taşır; test açılmadı. MPS/PyG ve CPU/GraphCare farklı Torch ortamları kullandı; sürümler `runtime_versions.json` içinde, süre/capacity eşitliği iddiası yok.

### Doğrulama ve kurtarma

- **281 korunmuş dosyanın** SHA-256 ve boyutu başlangıçla aynı: özgün 18 koşul, eski caches, kaynak/split, önceki review ve snapshot'a alınan dirty tracked dosyalar. `preservation_before.json`, `preservation_after.json`, `verification.json`.
- Bütün **150 validation prediction** dosyasından confusion-count tabanlı bağımsız NumPy accuracy/recall/F1 ve top-k yeniden hesaplandı; shared metrics helper çağrılmadı, kaydedilen altı metrik ile eşleşti.
- **10 best/last checkpoint** yeniden yüklenip 7.448 validation hastasında tekrar forward yapıldı; argmax birebir eşleşti. Maksimum probability farkı **3.03983688e-06**. `checkpoint_replay_*.json`.
- Gerçek GSAT checkpoint smoke epoch1/466 update noktasında durdu; Python/NumPy/Torch/MPS RNG ve optimizer ile epoch2’den devam edip 30 epoch tamamladı. `events.jsonl` restored olayı ve önceki state kaydı mevcut.
- İlk GraphCare zinciri legacy Torch `torch.mps` API eksikliğiyle **ilk checkpoint/ilk update öncesinde** code1 verdi; sqrt koşulu o zincirde hiç başlamadı. Aynı checkpoint regression legacy ortamda kırmızıydı; ardından legacy `weights_only` API farkı da aynı testte yakalandı. Capability detection düzeltildi ve her iki interpreter’da test edildi. Başarısız deneme/log silinmedi: `failed_attempts/graphcare_current_precheckpoint/`, `graphcare_current_failed_precheckpoint.log`.
- Compatibility migration, eski source hash’lerini ters dönüşümle doğruladı; girdi/model/loss değişikliği yok. GSAT epoch1 checkpoint hash’i migration öncesi/sonrası aynıydı. Eski/yeni code binding `compatibility_migration.json` içinde. GraphCare sıfır update’ten açıkça yeniden başladı; bu optimizer resume diye sunulmuyor. Sonraki iki GraphCare koşulu gerçekten completed ve checkpoint replay ile doğrulandı.
- `exit None` bildirimleri sonuç sayılmadı; OS PID/log/state/history/checkpoint ayrı okundu. Tamamlanmış koşulun üstüne yeni eğitim yazılmadı.
- Son testler: ana ortam **370 passed, 2 skipped, 14 warnings**; focused **9 passed, 2 skipped**; GraphCare ortamı **19 passed**. İki skip yalnız ana ortamda olmayan `pyhealth` nedeniyle GraphCare singleton/mixed testleridir; aynı iki test GraphCare ortamında geçti. Uyarılar mevcut LibreSSL/matplotlib deprecation bildirimleridir. `compileall` ve `git diff --check` geçti. `train_fit_removal_probe.json`, fit maskesi çıkarılınca heldout concept'lerin sözlüğe girdiğini ve regression'ın bunu yakaladığını kaydeder.
- Test çıktıları: `focused_tests_final.log`, `graphcare_tests_final.log`, `maintained_tests_final.log`. Test-first dilimler: feature/hub contract, train-fit, checkpoint/RNG restore, attempt binding ve warm-up/projection schedule. Gerçek singleton hub-only ve mixed-batch forward/backward her yöntemde test edildi. Tek-seferlik audit/runner/verifier için gerçek veri yürütmesi yapıldı; her evidence-script satırının TDD ile yazıldığı iddia edilmiyor.
- Commit/push yapılmadı, production runner koruması ve eski caches değiştirilmedi. Tek seferlik yeni input materyalizasyonu yalnız izole köke yazıldı.

## Komutlar ve artefact'ler

Çalışma dizini `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN`.

```bash
PYTHONPATH=. python3 -u comparison/standardized/prepare_common_input.py
python3 -m pytest tests/test_common_input_improvement.py -q
.venv-graphcare/bin/python3 -m pytest tests/test_common_input_improvement.py -q -k graphcare
PYTHONPATH=. python3 comparison/standardized/run_common_input.py --method gsat --variant current
PYTHONPATH=. python3 -u comparison/standardized/run_common_input.py --method gsat --variant current --execute --stop-after 1 > comparison/standardized/common_input_20260913/gsat_resume_smoke.log 2>&1
PYTHONPATH=. .venv-graphcare/bin/python3 -u comparison/standardized/run_common_input.py --method graphcare --variant current --execute > comparison/standardized/common_input_20260913/graphcare_current.log 2>&1
PYTHONPATH=. .venv-graphcare/bin/python3 -u comparison/standardized/run_common_input.py --method graphcare --variant sqrt_inverse --execute > comparison/standardized/common_input_20260913/graphcare_sqrt_inverse.log 2>&1

# İlk pre-checkpoint GraphCare hatasından sonra yalnız capability migration:
PYTHONPATH=. python3 comparison/standardized/migrate_common_checkpoint_compatibility.py
# GraphCare'in yukarıdaki iki komutu arşivleme sonrası aynı sırada yeniden çalıştı.
# GSAT gerçek optimizer/RNG resume, ardından iki ProtGNN koşulu && ile:
PYTHONPATH=. python3 -u comparison/standardized/run_common_input.py --method gsat --variant current --execute --resume > comparison/standardized/common_input_20260913/gsat_current.log 2>&1
PYTHONPATH=. python3 -u comparison/standardized/run_common_input.py --method protgnn --variant current --execute > comparison/standardized/common_input_20260913/protgnn_current.log 2>&1
PYTHONPATH=. python3 -u comparison/standardized/run_common_input.py --method protgnn --variant sqrt_inverse --execute > comparison/standardized/common_input_20260913/protgnn_sqrt_inverse.log 2>&1

PYTHONPATH=. .venv-graphcare/bin/python3 -u comparison/standardized/verify_common_input.py --replay graphcare > comparison/standardized/common_input_20260913/checkpoint_replay_graphcare.log 2>&1
PYTHONPATH=. python3 -u comparison/standardized/verify_common_input.py --replay gsat > comparison/standardized/common_input_20260913/checkpoint_replay_gsat.log 2>&1
python3 -m pytest tests/ -q > comparison/standardized/common_input_20260913/maintained_tests_final.log 2>&1
python3 -m pytest tests/test_common_input_improvement.py -q > comparison/standardized/common_input_20260913/focused_tests_final.log 2>&1
.venv-graphcare/bin/python3 -m pytest tests/test_common_input_improvement.py graphcare_analysis/test_zero_concept_model.py -q > comparison/standardized/common_input_20260913/graphcare_tests_final.log 2>&1
PYTHONPATH=. python3 -u comparison/standardized/verify_common_input.py --replay protgnn > comparison/standardized/common_input_20260913/checkpoint_replay_protgnn.log 2>&1
PYTHONPATH=. python3 -u comparison/standardized/verify_common_input.py > comparison/standardized/common_input_20260913/verification.log 2>&1
PYTHONPATH=. python3 comparison/standardized/render_common_input_report.py
python3 -m compileall -q comparison/standardized/common_input_improvement.py comparison/standardized/run_common_input.py comparison/standardized/prepare_common_input.py comparison/standardized/migrate_common_checkpoint_compatibility.py comparison/standardized/verify_common_input.py comparison/standardized/render_common_input_report.py tests/test_common_input_improvement.py
git diff --check
```

Komutlar tarihsel yürütme kaydıdır; occupied output'a tekrar eğitim başlatmayın. İlk GraphCare current hatasında `&&` nedeniyle sqrt koşulu çalışmadı; iki koşul yalnız arşivlenmiş hata ve compatibility düzeltmesi sonrasında gerçekten tamamlandı. Background `notify=true` kullanıldı; None bildirimleri OS/state doğrulaması olmadan sonuç sayılmadı.

Yeni modül `comparison/standardized/common_input_improvement.py`; tek-seferlik audit/runner `prepare_common_input.py`, `run_common_input.py`; compatibility kaydı `migrate_common_checkpoint_compatibility.py`; verifier/render `verify_common_input.py`, `render_common_input_report.py`; test `tests/test_common_input_improvement.py`. Tüm deney artefact'leri `comparison/standardized/common_input_20260913/` içinde. Özgün standardized runner korumalarına dokunulmadı. Başlangıç koruma snapshot'ı `preservation_before.json`.

Her tamamlanan hücrede doğrulanmış state/checkpoint/metrics'ten türetilmiş `run_manifest.json` var; şeması `common-input-source-snapshot-diagnostic-v1`, production benchmark şeması değildir. Konsolide sonuç `results_verified.json`, öğrenme eğrileri `learning_curves.csv` ve PNG/SVG, bağımsız doğrulama `verification.json`, kod/sürüm bağları manifest'ler ve `runtime_versions.json` içindedir. Uzun koşul/checkpoint uyumluluk dersleri varsayılan profildeki `checkpointed-ml-experiments` skill'ine eklendi.
