# İlaç–şikâyet etkileşimli PNA

## Kapsam

`pna_analysis/` çalışan, **opt-in deneysel GNN** paketidir. Mevcut üç modelin
varsayılanları, standardized method registry'si ve 18-hücre matrisi değişmez.
Bu paket o matrisin tam entegrasyonu değildir; kendi sınırlı train/eval/checkpoint
ve read-only replay CLI'sı vardır. Smoke sonuçları başarı karşılaştırması değildir.

Girdi mevcut `comparison/standardized/common_input_20260913/` içindeki
`inputs_no_identifiers.npz` ve `input_contract_and_audit.json` dosyalarından gelir:
104 ilaç + 88 şikâyet = 192 binary concept; 193-boyutlu one-hot içinde sabit hub.
59.607 train / 7.448 validation / 7.456 test ve 30 sınıf sırası korunur.
Yeni klinik bilgi, dış KG, hasta kimliği, etiket özelliği, node veya edge eklenmez.
Paylaşılan `pyg_graph` aynen kullanılır; sadece **sabit vocabulary metadata'sından**
`node_type` (med=0, complaint=1, hub=2) eklenir. Pertürbe edilen `x.argmax()` tür
belirlemek için hiçbir zaman kullanılmaz. `x` doğrudan trainable linear embedding'e
girer; one-hot üzerinde Grad/IG türev yolu açıktır.

Loader input/source/split SHA-256 değerlerini korunmuş audit ile, sınıf sırasını
canonical split ile, fold sayılarını NPZ/split ile ve bütün mevcut common-input
run manifest binding'lerini birbiriyle doğrular. NPZ hash'i mevcut all-row audit'e
bağlı row-order/label/fold eşliğini korur; ham preprocessing yeniden yapılmaz.
**`temporal_clean=false`, `raw_to_model_train_only=false`**: upstream kaynak
snapshot'ın whole-source filtrelerini veya zaman sorunlarını bu mimari gidermez.
Erken triyaj/klinik kullanım iddiası yoktur.

## Mesajın içeriğinde gerçekten ne değişiyor?

`InteractionPNAConv`, PyG'nin gerçek `PNAConv` / `MessagePassing` alt sınıfıdır.
Standart pre-MLP, aggregator, degree scaler, post-MLP ve tower output kullanılır.
Sadece **aggregation öncesindeki gerçek incoming mesaj içeriği** genişletilir.
Bir hub'ın ilaçları M, şikâyetleri C ve gerçek incoming-edge gate'leri a olsun:

```text
q(m,c) = tanh((A h_m) ⊙ (B h_c))           # R-boyutlu low-rank bilinear içerik
Δ_m    = U_med [Σ_c a_(c→hub) q(m,c) / |C|]
Δ_c    = U_cc  [Σ_m a_(m→hub) q(m,c) / |M|]
message(m→hub) = a_(m→hub) [PNA_pre(h_hub,h_m) + Δ_m]
message(c→hub) = a_(c→hub) [PNA_pre(h_hub,h_c) + Δ_c]
```

A/B/U bias'sız öğrenilen matrislerdir. İki yön ayrı content projection kullanır;
bu yalnız scalar attention skoru veya ilaç/şikâyet havuzlarının concatenation'ı
değildir. Sabit ilaç ve hub için şikâyet değişince **ilk katmandaki aynı ilaç
mesajı** değişir; `cross_pairs=False` kontrolünde değişmez. Çiftler sadece aynı
hedef hub'a gelen gerçek edge'lerden kurulur; batch genelinde eşleştirme yapılmaz.
Node ve edge permutation'ları equivariant; readout graph başına invariant'tır.
Pair intermediate maliyeti hasta başına O(|M| |C| R); bu başlangıç sürümü CPU ve
küçük patient-star graph'ları için yazılmıştır, yüksek dereceli graph'larda hız
üstünlüğü iddiası yoktur.

Ardından standart PNA: `mean/min/max/std` ×
`identity/amplification/attenuation`, tek tower, pre/post layers=1.
İki katman varsayılanı `h ← h + ReLU(PNA(h))`; readout **yalnız hub**, linear class
head. Batch normalization, global concept pooling veya maskeyi aşan bir predictor
kolu yoktur. M veya C boşsa Δ tam sıfırdır: standart mesaj yolu **birebir** korunur.
Hub-only graph finite forward/backward verir; sahte concept oluşturulmaz.

Degree histogram yalnız **tam canonical train** yıldızlarının in-degree'lerinden
fit edilir: k concept'li graph, k adet degree-1 yaprak ve degree-k hub içerir.
Histogram detached ve checkpoint'e kaydedilir. Tamamen edgeless bir training-set
olursa histogramı değiştirmeden average-log-degree paydasına 1e-6 sayısal alt sınır
konur; gerçek bu veride devreye girmez. Validation/test degree fit edilmez.

Standart çok-katmanlı PNA da etkileşimleri dolaylı öğrenebilir. Buradaki katkı,
aynı hub'ın karşı-tip concept çiftlerine **açık content-level inductive bias**;
üstünlük veya bilimsel yenilik henüz ispatlanmadı.

## Gerçek edge-mask sözleşmesi / GraphXAI

`edge_mask` gerçek **yönlü** edge sırasıyla [0,1] gate değerleri taşır. PyG
`set_masks` ile verilen maskeler de desteklenir; GNNExplainer logit maskeleri
PyG'nin `_apply_sigmoid` politikasına göre bir kez sigmoid'den geçirilir.

- Maskeler **pair context kurulmadan önce** okunur. Karşı concept'in incoming
  edge gate'i sıfırsa o concept'in content katkısı sıfırdır.
- Birincil mesajın tamamı kendi gerçek edge gate'iyle **bir kez** çarpılır.
- Pair ortalaması orijinal karşı-tip edge sayısına bölünür; maskelenen kütleyi
  yeniden bire normalize ederek maskeyi etkisizleştirmez.
- `explain_message` özellikle identity'dir. PyG'nin varsayılan final maskesini
  tekrar uygulamak gate'leri karesine yükseltirdi. Half-mask testi standart
  PyG PNA'nın bir kez maskelenmiş **tam çıktısıyla** eşitlik sınar.
- PNA degree/min/max semantics standart masked-message semantics olarak kalır;
  soft edge masking ile edge'i graph'tan fiziksel silmek aynı işlem değildir.
- All-ones logits tam aynı; all-zero real-edge gate'lerinde concept özelliklerini
  değiştirmek hub logits'ini değiştirmez. Hub'ın sabit identity prior'ı kalır.

`GraphXAIWrapper(model, original_graph.node_type)` orijinal node sırasını bağlar.
Node sırası/sayısı değişince yeni wrapper oluşturun; feature perturbation veya edge
subset'leri metadata'yı değiştirmez. Paylaşılan
`shared.lib.graphxai_standardized.explain_algorithms` ile gerçek vendored
**GradExplainer, IntegratedGradExplainer ve GNNExplainer** testleri hem çiftli hem
hub-only sentetik graph'ta çalıştırılır. Bu, native üç-model 500-kişilik explanation
cohort'u ile karşılaştırma veya full standardized explanation-schema kaydı değildir.

## Kontroller ve kapasite

Varsayılan 193-input / 30-class / 2-layer / rank16 için **gerçek model parametreleri**:

| CLI architecture | Width | Parametre |
|---|---:|---:|
| `plain` | 64 | 145.758 |
| `interaction` | 64 | 153.950 |
| `plain_wide` | 66 | 154.536 |

Interaction branch 8.192 parametre. `plain_wide`, metrik görmeden yalnız analitik
parametre sayısına en yakın daha geniş integer width ile seçilir; sayılar actual
model üzerinden tekrar ölçülür. Gap %0,380643; ilan edilmiş tolerans %2.
**Tam kapasite eşliği değil**, yakın parametre kontrolüdür; compute eşliği değildir.
Özel width/rank için rapor `within_tolerance=false` olabilir, bu gizlenmez.
`--disable-cross-pairs` aynı parametreli interaction ablation;
`--no-messages` tüm mesajları sıfırlayan negatif kontroldür. Plain modellerde pair
parametresi yoktur; standart PNA ile operatör eşliği test edilir.

## Çalıştırma

Repo kökünden, mevcut ana interpreter ile (bu makinede `python3`: Torch 2.8.0,
PyG 2.6.1). `.venv-graphcare` ayrı eski GraphCare ortamıdır; bu paketi oraya
kurmak veya bağımlılık değiştirmek gerekmez. `-B` Python bytecode yazımını da kapatır.

```bash
# Eğitim ve output yaratmaz; mevcut input hash/binding doğrulaması yapar.
python3 -B -m pna_analysis.run --help
python3 -B -m pna_analysis.run --dry-run
python3 -B -m pna_analysis.run --architecture plain --dry-run
python3 -B -m pna_analysis.run --architecture plain_wide --dry-run

# İzinli bounded smoke: train'den 256 graph, 8 update, küçük 64-val readout.
# Her çalıştırmada YENİ bir output adı kullanın; mevcut dizin ASLA overwrite edilmez.
python3 -B -m pna_analysis.run --execute --steps 8 --train-limit 256 \
  --val-limit 64 --batch-size 32 --architecture interaction --loss ce \
  --output-dir comparison/standardized/pna_experiments/implementation_smoke_v1

# Salt okuma: kaydedilmiş config/input/degree/code binding ile logits + metrics replay.
python3 -B -m pna_analysis.run \
  --replay comparison/standardized/pna_experiments/implementation_smoke_v1

# Reproducible sentetik, fixture ve gerçek GraphXAI compatibility testleri.
python3 -m pytest tests/test_pna_interaction.py tests/test_pna_runner.py -q
# Bütün maintained regression suite.
python3 -m pytest tests -q
```

İkinci loss `--loss sqrt_inverse`; ağırlıklar yalnız **tam train label count**'larından
ortak `class_weights` fonksiyonuyla hesaplanır, smoke subset'ten değil. CE de aynı
fonksiyonun `none` politikasını kullanır. `--val-limit 0`, sadece seçilen train
smoke subset'ini değerlendirir. Test evaluation seçeneği yoktur.

CLI hard sınırları: 1–100 update, 1–4096 train graph, 0–1024 validation graph,
batch 1–128, layers 1–3. Bu sürüm full benchmark runner değildir.
Seed1234, CPU deterministik ops, Adam lr=0.001/wd=1e-5; subset seçimi seeded train
permutation, validation orijinal NPZ sırasının ilk N satırıdır. **Selection: fixed-step
last checkpoint; validation seçimi yok.** Primary metric macro-F1, diğer metrikler
aynı `shared.lib.metrics.multiclass_metrics` implementasyonundan gelir. Küçük smoke
fold'unda bazı sınıflar bulunmayabilir; skorları performans kanıtı gibi yorumlamayın.

Output: `config.json`, `run_manifest.json`, her adımda güncellenen `history.json`
(loss ve her pair-parametre gradient L1), `checkpoint.pt` (model/optimizer/RNG,
config, vocabulary/classes, train degree/weights, row-index binding),
`predictions.npz`, `metrics.json`, input/source/split/önceki manifest SHA-256
before/after kayıtları ve git/runtime provenance. Tamamlanma ancak reload logits,
probabilities, labels, predictions ve metrikleri birebir yeniden üretince kaydedilir.
`--replay` mevcut dosyalara yazmaz; input veya executable-source hash'i değişmişse
reddeder. Checkpoint config'i immutable'dır; replay yeni CLI mimari/loss ayarlarını
uygulamaz. **Optimizer resume yok**; kesilen smoke'u koruyup yeni adla başlatın.

## Uygulama doğrulaması

TDD tracer'ları gerçekten RED→GREEN çalıştırıldı: eksik operator, mask-context
bypass, eksik end-to-end predictor, GraphXAI wrapper, train-degree fit, all-hub
NaN, kapasite kontrolü, input-binding loader, CLI, train/reload, read-only replay,
failed-step provenance. Ek removal probe'ları sqrt-loss'un CE'ye dönmesini ve
PyG'nin ikinci maskeyi uygulamasını yakaladı. Güncel sayısal execution kanıtı
`comparison/standardized/pna_experiments/` altında ayrı smoke/verification
artifact'leridir; eski sonuçları veya kaynak dosyaları değiştirmez.

### Gerçek execution kaydı

- `implementation_smoke_v1/`: train exit **0**, **8 update / 256 train graph**,
  yalnız **64 validation graph** readout; tam test değerlendirilmedi.
- Reload: logits max-abs fark **0.0**; bütün prediction dizileri ve ortak
  metrikler birebir eşit. Checkpoint SHA-256:
  `3a88782a8427b4bc15eecd242963c5ac24554f116c93e9660749f2834984d4e0`.
- Her pair-parametre tensöründe her adımda gradient L1 > 0;
  minimum **0.0002904093998949975**. Pair parametreleri gerçekten güncellendi.
- Eğitilmiş checkpoint üzerinde sentetik first-layer ilaç-message content değişimi
  L1 **0.015436183661222458**; cross-pairs kapalıyken **0.0**.
  All-ones/all-zero/maskelemiş karşı concept/no-message ihlal farkları **0.0**;
  mixed-batch max fark **5.960464477539063e-08**, permutation fark **0.0**.
- Maintained suite: **412 passed, 2 skipped, 14 warnings**; başlangıç baseline'ı
  **400 passed, 2 skipped, aynı 14 warnings**. Atlanan iki GraphCare koşulunun
  ayrı ortamındaki ilgili dosya ayrıca **8 passed**. `git diff --check` temiz.
- `implementation_verification_v1/regression.log`, `regression.xml`,
  `graphcare_regression.log`, `verification_summary.json` ve
  `model_contract_and_graphxai.json`: gerçek sonuçlar. Son dosyada pair/hub-only
  graph'larında üç named GraphXAI algoritmasının başarılı ham çıktıları bulunur.

Bu sayılar yalnız mekanizma, izolasyon, optimizasyon ve replay doğrulamasıdır;
validation macro-F1 üstünlüğü veya klinik genelleme kanıtı değildir.
