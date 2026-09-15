# MedGNN kullanılabilirlik ve doğrulama raporu

## Sonuç ve sınır

Çalışma kökü: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN`.

- Son tam bakım testi: **342 passed, 14 warnings in 86.50s**; çıkış kodu 0.
- **28 ayrı argparse giriş dosyasının tamamında `--help` başarılı.** GSAT açıklayıcısının ikinci, standardized parser dalı ayrıca çalıştırıldı. Üretilen üç eğitim ve üç açıklama komutu da kendi gerçek yorumlayıcısında `--help` eklenerek doğrulandı.
- Orkestrasyon artık gerçek cache oluşturucuyu çağırıyor, ardından parity denetliyor; 18 eğitimi ayrı iş olarak izliyor ve altı topology/seed açıklama setini gerçekten çalıştıracak komutlar üretiyor. Başarısız iş sonraki adımları durduruyor.
- Run-level yeniden başlatma, tamamlanmış çıktıları doğrulayarak yeniden kullanıyor. Tamamlanmamış bir run/set yalnız açık `--archive-incomplete` ile korunarak başka bir attempt dizinine taşınabiliyor; bu seçenek bu çalışmada **yalnız geçici fixture dizinlerinde** kullanıldı. Epoch/optimizer resume uygulanmadı ve böyle bir iddia yok.
- **Gerçek veri üzerinde cache üretimi, eğitim veya açıklama üretimi yapılmadı.** Her iki standardized topology için PyG cache, metadata ve GraphCare KG dosyaları yok. Üretim audit komutu beklendiği gibi eksik `star` cache nedeniyle çıkış 1 verdi. Üretim parity hash'i veya benchmark metriği iddia edilmiyor.
- Test sayısı bütün fonksiyonların kanıtlandığı anlamına gelmez. Bazı testler pahalı eğitim sınırını stub/mock ile izole eder; aşağıda gerçek model, gerçek alt süreç, fixture ve yalnız-parser kanıtları ayrılmıştır.

## Başlangıç incelemesi ve koruma

Önce `pwd`, `git status --short`, `git diff --stat`, kök dizin ve talimat dosyası keşfi yapıldı. Üst dizindeki `AI-Workplace/AGENTS.md` kuralları uygulandı. Bakımı yapılan kökte ek `AGENTS.md` bulunmadı; bulunan `visualizer/node_modules/cytoscape/AGENTS.md` üçüncü taraf bağımlılığına aitti ve o bağımlılık değiştirilmedi. `README.md`, `STRUCTURE.md`, `requirements.txt`, `requirements-lock.txt`, `environment.yml`, standardized config/runbook, gerçek runner/parser ve manifest API'leri, viewer `package.json`/README ile ham veri runbook'u okundu. Başlangıç kapsamı salt kurulum incelemesi değildi; yetkili minimal uygulama ve test çalışmasıydı. Paket kurulmadı.

Başlangıçta zaten değişmiş izlenen dosyalar:

```text
README.md
graphcare_analysis/adapter.py
graphcare_analysis/build_kg.py
graphcare_analysis/config.py
graphcare_analysis/run.py
protgnn_analysis/config.py
protgnn_analysis/load_dataset.py
protgnn_analysis/train.py
shared/lib/config_base.py
shared/lib/graph_structures.py
shared/lib/metrics.py
```

Başlangıçta ayrıca `comparison/standardized/`, `docs/superpowers/`, `graphcare_analysis/explainability/`, `gsat_analysis/`, `protgnn_analysis/explainability/explain_standardized.py`, `shared/lib/{benchmark_contract,canonical_graph,explanation_contract,fidelity,run_manifest}.py` ve `tests/` izlenmeyen mevcut çalışmalar içeriyordu. Bunlar yeni baştan yazılmış veya temizlenmiş sayılmadı. Bu geçişte bu alanlardan yalnız aşağıdaki açık dosya listesi düzenlendi; önceki bilimsel uygulama korunuyor. `git reset`, checkout/revert, commit ve push yapılmadı.

CLI/model doğrulama taramasından önce `data/`, üç yöntemin `outputs/` dizinleri, standardized `results/` ve `explanations/` altında **304 giriş** için dosya boyutu/mtime, dizin varlığı ve symlink hedefi kaydedildi. Sonraki karşılaştırma: **değişen 0, eklenen 0**. Bu bir tüm-repo kriptografik başlangıç snapshot'ı değil; belirtilen korunan ağaçların metadata/symlink karşılaştırmasıdır. Python `__pycache__` ve pytest'in kendi cache/temporary dizinleri test çalışması sırasında oluşabilir.

Korunan mevcut veri symlink'i:

```text
data/graphs/cooccur/protgnn/hetero_merged_ed_noLOS_prev10_pmi2_miss_disease
 -> /Users/necatifurkancolak/Projects/MP---Self-Explainable-Graph-Neural-Networks-via-Prototype-Learning-for-Medical-Diagnosis/data/processed_hetero_merged_ed_noLOS_prev10_pmi2_miss_disease
```

Bu eski hedef metni özellikle değiştirilmedi. Kök `comparison/standardized/checkpoint/state.json` doğrulama sonunda hâlâ yoktu; dry-run üretim checkpoint state'i oluşturmadı.

## Kök dizin envanteri ve organizasyon

İncelenen kök girişler:

```text
.DS_Store  .claude/  .git/  .gitignore  .pytest_cache/  .superpowers/
.venv-graphcare/
README.md  STRUCTURE.md  TODO.txt
requirements.txt  requirements-lock.txt  environment.yml
protgnn_analysis/  gsat_analysis/  graphcare_analysis/  shared/
comparison/  baselines/  tests/  external/  data/  visualizer/
docs/  docs-vault/  thesis/  thesis_proposals/
data.zip  Master Project_Template..zip  Master Project_Template./
job_postings_2026-07-11.xlsx
```

[STRUCTURE.md](../STRUCTURE.md) artık üç yöntemi, current/legacy ayrımını, veriyi, harici bağımlılıkları, viewer'ı, tez/vault belgelerini ve çıktı sahipliğini aynı yerde gösteriyor. README'de güvenli başlangıç komutları ve navigasyon var. Bilimsel paketler taşınmadı; root'taki arşivler, unrelated workbook ve sistem dosyaları da silinmedi. Git diff istatistiğinin büyük görünmesi önceki standardized çalışma değişikliklerini de içerir; tüm diff bu geçişte üretilmiş değildir.

## Doğrulanan kök nedenler ve değişiklikler

| Sorun | Kaynak kanıtı | Düzeltme / regresyon kanıtı |
|---|---|---|
| Cache fazı üretim yapmıyordu | Eski `command_plan()['cache']` yalnız `audit_caches` çağırıyordu; audit'in kendisi cache üretmediğini açıkça söylüyor. | `build_caches.py` gerçek `get_dataset` ve `build_global_kg` API'lerini kullanıyor; default/dry-run yazmıyor, execute eksikleri üretiyor ve geçerli cache'i tekrar kullanıyor. Plan builder ardından audit çağırıyor. Altı satırlık geçici veriyle iki topology gerçekten üretildi, tekrar kullanımda byte'lar korundu ve gerçek audit her topology'de 6 subject için geçti. Bu fixture'ın küçük sınıf/fold sözleşmesi testte geçici olarak küçültüldü; üretim kanıtı değildir. |
| Açıklama komutu eksikti ve execute bile dry-run idi | Eski argv `run_explanations --dry-run` idi; gerçek parser `--topology` ve `--seed` istiyor. | Plan artık iki topology × üç seed için altı gerçek execute komutu oluşturuyor. Her sette üç yöntemin standardized açıklayıcısı var. Plan testi ve gerçek generated-child parser kontrolleri geçti. |
| Bir faz tekrarlandığında tamamlanan run'ların dizinleri doluydu | Runner `_reject_occupied_outputs` mevcut cell'i reddediyor, önceki orchestrator bütün matrisi tek train işi sayıyordu. | Eğitim her cell için ayrı argv/kalıcı state; tamamlanmış çıktılar manifest, input hash, metric sözleşmesi ve checkpoint varlığıyla doğrulanarak atlanıyor. Parent öldükten sonra tamamlanmış child'ın kurtarılması gerçek geçici manifest/artifact ile test edildi. Kalan/incomplete cell'i açık archive-retry ayrı attempt olarak koruyor. |
| Önceki failed faz atlanıp downstream'e geçiliyordu | `Checkpoint.start` failed için False döndürüyordu; ana döngü koşulsuz `continue` yapıyordu. | Failed iş artık pipeline'ı nonzero ile bloke eder. Sentinelli gerçek Python alt süreç testi, fail sonrasında açıklamanın başlamadığını ve explicit retry'da önceki başarılı işin tekrarlanmadığını gösteriyor. |
| İki legacy CLI import'ta çöküyordu | `confusion_analysis` olmayan `disease_baseline` root import'unu; `explain_checkpoint` kaldırılmış `train_and_explain` modülünü kullanıyordu. | Gerçek `baselines.disease_baseline` ve `protgnn_analysis.train` import'ları; confusion CSV varsayılanı ortak `DATA_DIR`. İki başarısız help reproducer, yamadan sonra geçti. |
| HPO help/import çıktı dizini oluşturuyordu | Modül düzeyinde `os.makedirs(RESULTS_DIR)` vardı. | Dizin oluşturma başarılı argument parsing sonrasına alındı; import yan etkisi regresyonu geçti. |
| Legacy multi-seed yanlış trainer ve eski metric dosyasını kullanıyordu | Olmayan `scripts/train_and_explain.py` çağrısı ve sabit `outputs/results/test_metrics.json`; multiclass metric adları aggregate listesinde yoktu. | `-m protgnn_analysis.train`, doğru repo cwd, `latest_run.txt` üzerinden son run metric'i ve güncel multiclass metric anahtarları. Launch fixture'ı ve aggregate regresyonu geçti; gerçek multi-seed eğitimi çalıştırılmadı. |

`run_all` state'i her argv'yi kaydeder; aynı iş anahtarı için komut değişmişse ayrı state/events çifti ister. `--dry-run` ve `--execute` çelişkisi parser'da reddedilir. `--archive-incomplete`, `--resume --retry-failed` olmadan kabul edilmez. Kesilme `BaseException` dahil failed olarak kaydedilir. Açıklama resume, cohort/subject/schema/top-k sözleşmesini ve mevcut checkpoint hash'lerini tekrar doğrular; üç yöntem × 50 sentetik açıklama kaydıyla bu kurtarma yolu ve checkpoint hash değişimi testi geçti. Sentetik kayıtlar gerçek model sonuçları olarak raporlanmadı.

### Bu geçişte oluşturulan dosyalar

- `comparison/standardized/build_caches.py`
- `tests/test_build_caches.py`
- `tests/test_pipeline_recovery.py`
- `tests/test_cli_usability.py`
- `tests/test_protgnn_model_smoke.py`
- `docs/usability-verification.md`

### Bu geçişte düzenlenen mevcut dosyalar

- `comparison/standardized/run_all.py`
- `tests/test_run_all.py`
- `protgnn_analysis/scripts/confusion_analysis.py`
- `protgnn_analysis/scripts/explain_checkpoint.py`
- `protgnn_analysis/scripts/hpo_disease.py`
- `protgnn_analysis/scripts/run_multi_seed.py`
- `README.md`
- `STRUCTURE.md`
- `comparison/standardized/README.md`
- `docs/RUN_WITH_OWN_DATA.md`
- `visualizer/README.md`

## CLI envanteri — parser kanıtı

Aşağıdaki **28 dosyanın tamamı** doğrudan dosya formunda çalıştırıldı:

```bash
PYTHONPATH=.:external/GraphXAI-main:external/GraphCare python3 <dosya> --help
# graphcare_analysis/ altındaki iki argparse CLI için:
PYTHONPATH=.:external/GraphXAI-main:external/GraphCare .venv-graphcare/bin/python3 <dosya> --help
```

Tablo source `add_argument` envanteri ile gerçek help çıktılarını birleştirir; aliases korunmuştur. `--help` başarısı yalnız import/parser erişilebilirliğini gösterir, her seçeneğin pahalı çalışma yolunu değil. GSAT açıklayıcısında legacy ve standardized bayrak kümeleri iki farklı parser'a aittir; tüm bayraklar aynı anda kullanılmaz.

| Dosya | Doğrulanan uzun bayraklar (`--help` ayrıca tümünde var) | Sonuç |
|---|---|---|
| `comparison/error_analysis.py` | `--figure`, `--no_figure`, `--run`, `--top_pairs` | help: 0 |
| `comparison/explain_canonical.py` | `--n` | help: 0 |
| `comparison/standardized/audit_caches.py` | `--canonical-split`, `--dataset`, `--structures` | help: 0 |
| `comparison/standardized/build_caches.py` | `--canonical-split`, `--dataset-dir`, `--dry-run`, `--execute`, `--structures` | help: 0 |
| `comparison/standardized/build_explanation_cohort.py` | `--dataset`, `--n`, `--output`, `--seed`, `--split` | help: 0 |
| `comparison/standardized/run_all.py` | `--archive-incomplete`, `--dry-run`, `--events`, `--execute`, `--resume`, `--retry-failed`, `--state` | help: 0 |
| `comparison/standardized/run_benchmark.py` | `--config`, `--continue-on-error`, `--dry-run`, `--limit`, `--max-epochs`, `--methods`, `--seeds`, `--structures` | help: 0 |
| `comparison/standardized/run_explanations.py` | `--cohort`, `--dataset`, `--dry-run`, `--output-root`, `--seed`, `--split`, `--structure`, `--topology` | help: 0 |
| `comparison/standardized/summarize.py` | `--results-dir`, `--summary-aggregate-csv`, `--summary-csv`, `--summary-md` | help: 0 |
| `protgnn_analysis/explainability/explain_standardized.py` | `--canonical-split`, `--canonical_split`, `--checkpoint`, `--cohort`, `--dataset`, `--graph-structure`, `--graph_structure`, `--out-dir`, `--out_dir`, `--seed` | help: 0 |
| `protgnn_analysis/scripts/archive_results.py` | `--command`, `--overwrite`, `--results-dir`, `--runs-dir`, `--tag`, `--timestamp` | help: 0 |
| `protgnn_analysis/scripts/confusion_analysis.py` | `--csv`, `--min_class`, `--top_pairs` | help: 0 |
| `protgnn_analysis/scripts/eval_checkpoint.py` | `--dataset`, `--which` | help: 0 |
| `protgnn_analysis/scripts/explain_checkpoint.py` | `--clinical_limit`, `--clst`, `--dataset`, `--explain_n`, `--log`, `--sep`, `--skip_clinical`, `--which` | help: 0 |
| `protgnn_analysis/scripts/generate_clinical_explanations.py` | `--csv`, `--explainer`, `--limit`, `--results_dir` | help: 0 |
| `protgnn_analysis/scripts/hpo_disease.py` | `--dataset`, `--max_epochs`, `--n_trials`, `--seed`, `--subsample` | help: 0 |
| `protgnn_analysis/scripts/run_multi_seed.py` | `--seeds`, `--tag` | help: 0 |
| `protgnn_analysis/scripts/summarize_all_test.py` | `--dataset`, `--limit`, `--threshold`, `--top_nodes`, `--which` | help: 0 |
| `protgnn_analysis/scripts/summarize_graphxai.py` | `--dataset`, `--explanations_dir`, `--top_nodes` | help: 0 |
| `protgnn_analysis/scripts/summarize_hetero_explanations.py` | `--dataset`, `--explainer`, `--top_nodes` | help: 0 |
| `protgnn_analysis/train.py` | `--archive_tag`, `--canonical-split`, `--canonical_split`, `--clst`, `--dataset`, `--explain_n`, `--graph`, `--graph-structure`, `--graph_structure`, `--limit`, `--margin`, `--max-epochs`, `--max_epochs`, `--no_archive`, `--no_prot`, `--out_dir`, `--output-dir`, `--seed`, `--sep` | help: 0 |
| `gsat_analysis/explainability/explain_gsat.py` | `--canonical-split`, `--canonical_split`, `--checkpoint`, `--ckpt`, `--cohort`, `--dataset`, `--graph`, `--graph-structure`, `--graph_structure`, `--match_subjects`, `--max_graphs`, `--out-dir`, `--out_dir`, `--seed`, `--top_nodes` | help: 0 |
| `gsat_analysis/explainability/summarize_gsat.py` | `--explanations_dir`, `--graph`, `--graph_structure`, `--top_nodes` | help: 0 |
| `gsat_analysis/train.py` | `--canonical_split`, `--graph`, `--graph_structure`, `--limit`, `--max_epochs`, `--out_dir`, `--seed` | help: 0 |
| `graphcare_analysis/explainability/explain_graphcare.py` | `--canonical-split`, `--canonical_split`, `--checkpoint`, `--cohort`, `--dataset`, `--graph-structure`, `--graph_structure`, `--out-dir`, `--out_dir`, `--seed` | help: 0 |
| `graphcare_analysis/run.py` | `--canonical_split`, `--graph`, `--graph_structure`, `--limit`, `--max_epochs`, `--out_dir`, `--patience`, `--seed` | help: 0 |
| `baselines/disease_baseline.py` | `--full`, `--model`, `--no-pyxis` | help: 0 |
| `visualizer/scripts/export_protgnn_graphs.py` | `--dataset-name`, `--keep-existing`, `--limit`, `--out-dir`, `--shard-size` | help: 0 |

GSAT standardized dalının bağımsız gerçek kontrolü:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 -m gsat_analysis.explainability.explain_gsat --checkpoint fixture-not-loaded.pt --help
```

Çıkış 0; fixture checkpoint yüklenmedi. Ayrıca `command_for(BenchmarkSpec(method, 'star', 1234), RunnerOptions(load_config()))` ile üretilen üç eğitim argv'si ve `build_commands(topology='star', seed=1234, require_checkpoints=False)` ile üretilen üç açıklama argv'sine `--help` eklendi. Altı alt süreç de config'teki gerçek interpreter ile çıkış 0 verdi. Böylece yalnız elle yazılmış örnekler değil, pipeline'ın gerçek ürettiği bayraklar da parser'da denendi.

### Parser'ı olmayan / legacy script entry point'leri

Bunlar kaynak üzerinden incelendi; `--help` güvenli varsayılmadı:

| Dosya(lar) | Durum |
|---|---|
| `comparison/build_split.py` | Canonical split üretir; çalıştırılmadı, mevcut split korunuyor. |
| `comparison/compare.py` | Eski karşılaştırma raporunu yazar; çalıştırılmadı. |
| `comparison/plain_gcn_canonical.py`, `comparison/protgnn_canonical.py`, `comparison/tabular_baseline_canonical.py` | Eski eğitim/ablation entry point'leri; çalıştırılmadı. |
| `protgnn_analysis/explainability/graphxai_integration.py` | Legacy örnek main/path varsayımları; standardized açıklama API'si yerine geçmez, çalıştırılmadı. |
| `protgnn_analysis/models/{GCN,GAT,GIN}.py`, `protgnn_analysis/my_mcts.py` | Embedded model/demo entry point'leri; doğrudan main çalıştırılmadı. Standardized GCN model API'si ayrıca smoke test edildi; GAT/GIN/MCTS'nin her yolu test edilmiş değildir. |
| `protgnn_analysis/scripts/check_environment.py` | Gerçek import check çalıştırıldı, çıkış 0. |
| `graphcare_analysis/build_kg.py` | Doğrudan çalıştırmak legacy cache üretebilir; sadece yeni split-aware builder üzerinden geçici fixture ile API test edildi. |
| `graphcare_analysis/test_model_smoke.py` | İzole GraphCare yorumlayıcısında gerçek sentetik forward/backward çalıştırıldı. |
| `shared/data_prep/chiefcomplaint_standardizer.py` | Embedded örnekler; source inspection, yeni real-data işlem yok. |
| `shared/data_prep/extract_ed_labs.py` | Büyük lab batch işlemi; çalıştırılmadı. |
| `shared/data_prep/merge_ed.py` | Main guard yok, import'ta veri okur/yazar; import/`--help` çalıştırılmadı. |

Shared library dosyaları normal import API'leridir, CLI değildir. `external/` içindeki upstream script ve testler, `visualizer/node_modules/`, sanal ortamlar, tez dosyaları ve notebook'lar maintained Python suite'e dahil edilmedi; korunup yerinde bırakıldı. Viewer npm girişleri `dev`, `build`, `preview` olarak manifestten okundu; npm test script'i yok.

## Çalıştırılan komutlar ve gerçek sonuçlar

Tüm Python komutlarının cwd'si repo kökü; viewer build adımlarının cwd'si `visualizer/`.

| Komut / kontrol | Sonuç |
|---|---|
| `python3 -m pytest tests/test_run_all.py -q` başlangıç | 4 passed; eski yüzeysel testlerin hataları yakalamadığı doğrulandı. |
| Yeni hedefli RED testleri | Failed fazdan sonra child launch, cache yerine audit, faz tekrarı, eksik recovery API, iki bozuk import, HPO import mkdir, yanlış multi-seed trainer/metric ve eksik multiclass aggregate beklenen şekilde kırmızı görüldü. |
| `python3 -m pytest tests -q` ilk uygulama geçişi | 332 passed, 14 warnings; ardından kapsam eklenip yeniden çalıştırıldı. |
| `python3 -m pytest tests -q` son geçiş | **342 passed, 14 warnings in 86.50s**, exit 0. |
| `python3 -m compileall -q comparison protgnn_analysis gsat_analysis graphcare_analysis shared baselines tests visualizer/scripts` | exit 0. Third-party/venv/data taranmadı. |
| `git diff --check` | exit 0; bu kontrol izlenmeyen dosyaları kapsamaz. Yeni Python dosyaları ayrıca compile/test edildi. |
| `python3 -m comparison.standardized.run_all --dry-run` | exit 0; faz komut sayıları: preflight 1, cache 2, train 18, explain 6, summarize 1. |
| `python3 -m comparison.standardized.build_caches --dry-run` | exit 0; execute false, üretim cache yok. |
| `python3 -m comparison.standardized.run_benchmark --dry-run` | exit 0; JSON'da programatik olarak **18 benzersiz method/topology/seed** doğrulandı. |
| `python3 -m comparison.standardized.run_explanations --topology <star/cooccur> --seed <1234/1235/1236> --dry-run` | Altı kombinasyon da exit 0; her JSON tam üç method argv'si içeriyor. |
| `python3 -m comparison.standardized.audit_caches --structures star cooccur` | **exit 1**, eksik production `star` PyG cache. Fail-closed gate doğru; full-data parity kanıtı değil. |
| `python3 -m pytest tests/test_protgnn_model_smoke.py tests/test_gsat_smoke.py::test_gsat_synthetic_forward_backward tests/test_gsat_smoke.py::test_graphxai_wrapper_is_deterministically_faithful -q` | 4 passed in 3.18s: ProtGNN GCN prototipli/prototipsiz ve GSAT gerçek sentetik forward/backward; finite loss/gradient, GSAT wrapper deterministik eşitlik. Optimizer step/eğitim döngüsü yok. |
| `PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3 graphcare_analysis/test_model_smoke.py` | `OK: logits (2, 30), loss 3.925, backward ran.` Sentetik rastgele smoke loss'u, benchmark metriği değil. |
| `PYTHONPATH=.:external/GraphXAI-main python3 -m protgnn_analysis.scripts.check_environment` | Core/optional/project/üç GraphXAI explainer import'u OK; Environment check passed, exit 0. |
| `./node_modules/.bin/tsc --noEmit` | exit 0. |
| `./node_modules/.bin/vite build --outDir /tmp/medgnn-visualizer-usability-build` | Başarılı build; 38 module transformed, 1.81s. Repo viewer çıktıları değiştirilmedi. Browser render/end-to-end kullanıcı akışı test edilmedi. |

Ek gerçek entegrasyonlar son tam suite içinde: küçük CSV'den split-aware PyG/KG üretimi → yeniden kullanım → record parity audit; sentinelli gerçek subprocess başarısızlık/retry akışı; geçici completed/incomplete manifest ve artifact kurtarma; sentetik üç-method açıklama output validator; summarizer'ın geçici manifestlerden CSV/aggregate CSV/Markdown üretmesi. Eğitim runner sözleşme testlerinde pahalı method execution sınırı gerektiğinde mock/stub olarak kalır.

Üretim audit'in verdiği eksik dosya:

```text
data/graphs/star/protgnn/hetero_merged_ed_noLOS_prev10_pmi2_miss_disease_std_ds95b055d53c5f_split33f6cd71399a_recipe669c4a8ae0c8/data.pt
```

Son salt-okunur kontrol: `star` ve `cooccur` için exact-recipe PyG data, metadata ve `graphcare/kg.pt` yollarının **hepsi mevcut değil**. Eski cooccur symlink'i current standardized cache değildir ve bu eksikliği kapatmış sayılmadı.

### Ortam ve uyarılar

- Ana Python: `/Library/Developer/CommandLineTools/usr/bin/python3`, 3.9.6; torch 2.8.0, PyG 2.6.1, NumPy 1.26.2, pandas 2.3.3, scikit-learn 1.6.1. Ana import check geçti.
- GraphCare: `.venv-graphcare/bin/python3`, Python 3.9.6, torch 1.12.0, PyG 2.3.0; ayrı tutuldu.
- Ana lock/env NumPy 2.0.2 pinli; mevcut test edilen NumPy 1.26.2. Bu oturum fresh lock/conda install doğrulaması değildir. Yeni kurulum için dokümanın önerdiği Python 3.10/3.11 ayrı bir ortamda doğrulanmalı.
- Ortamda urllib3/LibreSSL `NotOpenSSLWarning`; suite içinde 14 Matplotlib/pyparsing deprecation uyarısı var. Uyarıları susturmak için bağımlılık değiştirilmedi.
- Viewer Node v25.9.0 / npm 11.12.1 / Vite 8.1.1. Build dış `/tmp` outDir'i temizlemediğini ve >500 kB chunk uyarısını verdi. Bunlar başarısız build değil; bundling optimizasyonu yapılmadı.

## Kalan işler ve onay gerektiren yollar

1. **Üretim cache ve parity**: gerçek standardized cache generation ayrı onay ister. Ardından audit geçmeden full eğitim başlamamalı; mevcut durumda gate bunu engelliyor.
2. **Gerçek ML sonuçları**: 18 full run, üretim checkpoint yükleme/kalite ölçümü, gerçek fixed-cohort attribution ve bilimsel karşılaştırma metrikleri çalıştırılmadı. Bounded bir-epoch real-data smoke da yapılmadı; bu görev eğitime izin vermiyordu.
3. **Resume sınırlaması**: optimizer/scheduler/RNG/son epoch state restore yok. `--archive-incomplete` eski run/set'i korur ve onu baştan çalıştırır. Tek orchestrator çalıştırılmalı; stale-running işin gerçekten durmuş olduğu operatörce doğrulanmalı. Preflight veri/split/matris denetimidir; explanation cohort mevcut input'tur, otomatik yeniden oluşturulmaz.
4. **Ham veri girişleri**: `merge_ed.py` import-time çalışır ve `extract_ed_labs.py` ile birlikte kök yerine `shared/data` hesaplar. Dosyalar taşınmadı; docs artık bu riski açıkça işaret ediyor. Sonraki iş: küçük raw-table fixture'larıyla main guard, gerçek argparse/help/dry-run, doğru root path ve yazma hedefi sözleşmesi. Tam MIMIC merge/lab extraction bu test sayısıyla doğrulanmış sayılmaz.
5. **Legacy script sınırı**: eval/explain/clinical summary/HPO/confusion/export/archive/multi-seed işlerinin tamamı gerçek veri ve checkpoint'lerle uçtan uca çalıştırılmadı. HPO ve confusion eğitim yapar; viewer exporter dataset/manifest yazar; archive script'inin overwrite seçeneği ayrı risk taşır. Legacy timestamped output düzeni standardized cell isolation güvencesi değildir. Yeni karşılaştırmalar için standardized runner tercih edilmeli.
6. **İskelet/alternatif model yolları**: örneğin `GnnBase.save_state_dict`/config `process_args` gibi eski placeholder'lar ile tüm GAT/GIN/MCTS ve GraphXAI algoritma dalları bu geçişte tamamlanmış/test edilmiş değildir. Coverage oranı ölçülmedi; “her fonksiyon çalışıyor” iddiası yok.
7. **Fiziksel organizasyon**: `src/` toplu geçişi veya scientific module rename yapılmamalı. Önce güncel navigation/CLI konsolide edilmeli; tekrarlı helper'lar call-site testleriyle ele alınmalı. `docs-vault/` proje içinde kalmalı. `data.zip`, eski template kopyaları ve unrelated workbook ancak sahiplik/retention onayı sonrası tek tek arşivlenmeli; bu geçişte yerlerinde kaldılar.
8. **Harici kapsam**: third-party notebook/test koleksiyonları, yeni bağımlılık kurulumu, güvenlik taraması, uzaktaki hizmetler, deployment, commit/push bu çalışmaya dahil edilmedi.

Geçici yardımcı kanıt dosyaları (kalıcı teslim yerine tekrar üretilebilir ekler): `/tmp/medgnn-help-results.json`, `/tmp/medgnn-cli-flags.json`, `/tmp/medgnn-dryrun-results.json`, `/tmp/medgnn-production-audit.txt`, `/tmp/medgnn-protected-check.json`. Kalıcı, kök navigasyondan bağlı teslim bu rapordur.
