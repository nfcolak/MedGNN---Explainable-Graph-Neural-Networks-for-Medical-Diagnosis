# MedGNN: dördüncü GNN için kanıta dayalı araştırma

## Karar

**İlk aday: sığ residual GIN + çok-ölçekli Jumping Knowledge readout. İkinci aday: aynı kontrollü iskelette PNA.** Önce mevcut GSAT'ın kapısız, bilgi-cezasız GIN predictor'ı güçlü kontrol olarak kurulmalı; residual/çok-ölçekli sürüm ancak bu kontrolü de geçerse tutulmalı. Bu öneri gerçekten mesaj-geçiren bir GNN'dir; tabular ensemble, GSAT+GraphCare birleşimi veya LLM ile özellik zenginleştirme değildir.

**Bu kombinasyon yeni bir bilimsel yöntem iddiası değil, literatüre dayanan güçlü bir mühendislik baseline'ıdır.** GIN zaten GSAT backbone'unda var. Sadece gate'i kaldırıp adını değiştirmek özgün dördüncü yöntem sayılmaz; benchmark'a ayrı bir predictive-GNN satırı ekler. Dördüncü satırın ayrıca farklı, yayımlanmış bir mesaj-geçirme operatörünü temsil etmesi şartsa **PNA** daha temiz seçimdir. GIN'in çok-katmanlı readout'u da özgün makalesinde bulunur.[1][2][10]

**“Üç modeli ciddi farkla geçecek” garantisi için kanıt yok.** Ulaşılabilir kazanım henüz ölçülmedi. Literatür adayların mekanizmasını destekliyor; MedGNN'in bu 30-sınıflı, sabit girdili görevinde beklenen skorlarını vermiyor. En savunulabilir yol, karmaşıklığı artırmadan önce mevcut bilginin daha iyi kullanılıp kullanılamadığını sınamak.

| Aday | Mevcut göreve uyum | Deneme önceliği | Başlıca sınır |
|---|---|---|---|
| **Residual GIN + JK**; GINE yalnız edge-semantics ablation'ında | Küçük categorical graph, sabit star, graph classification | **1**; önce sade GIN kontrolü | GSAT'tan bağımsız yeni operatör değil; kazanç ve özgünlük henüz yok |
| **PNA** | Hub'da farklı concept mesajlarını ve dereceyi birlikte özetleme | **2**; farklı yayımlanmış GNN | Degree-1 yapraklarda çoklu istatistikler tekrarlı; ek maliyet karşılıksız kalabilir |
| **R-GCN** | İlaç/şikâyet/hub ayrımına relation-specific dönüşüm | 3; mevcut relation çeşitliliği sınırlı | Bugünkü star'ın bütün spokes'ları aynı relation ID; yeni klinik ilişki yok |
| **GraphGPS** | Gerçek local GNN + global attention | İlk ucuz taramaya alma | Star'da yollar zaten kısa; global bypass açıklama karşılaştırmasını zorlaştırır |
| **MUSE** | Sağlık alanı güçlü; fakat farklı veri/graph problemi | Mevcut sözleşmede ele | Patient–modality batch graph, çoklu modaliteler ve contrastive training gerektiriyor |

Aşağıdaki uygunluk ve maliyet sıralaması **bu repo için mühendislik değerlendirmesidir**, makalelerden aktarılmış bir MedGNN başarı sıralaması değildir.

## 1. Gerçekte hangi baseline'ı geçmemiz gerekiyor?

### 1.1 Eski native-input test matrisi: üç seed'in tamamı

Korunmuş `comparison/standardized/benchmark_results_summary.tsv` içindeki **18 completed hücrenin** işaret ettiği gerçek metric JSON'ları okundu. ProtGNN için üst düzey metrikler, GSAT/GraphCare için `test` alt sözlüğü kullanıldı. TSV ile sayısal eşleşme kontrol edildi. Her topology ayrı tutuldu; **1234, 1235, 1236** üzerinden aritmetik ortalama ve **örnek standart sapma, ddof=1** hesaplandı. Test-best seed seçilmedi.

Her satır aynı canonical **n=7.456 test**, 30 sınıf; seed değişir, split değişmez. Değerler 0–1 ölçeğinde, altı ondalığa yuvarlandı.

| Native-input model | Topology | Macro-F1 ort. ± std | Accuracy ort. ± std | Balanced accuracy ort. ± std |
|---|---|---:|---:|---:|
| ProtGNN | star | 0.495749 ± 0.005455 | 0.543723 ± 0.012679 | 0.567901 ± 0.005245 |
| ProtGNN | cooccur | 0.495493 ± 0.005122 | 0.547076 ± 0.005758 | 0.563328 ± 0.005827 |
| GSAT | star | 0.544321 ± 0.003016 | 0.618205 ± 0.002664 | 0.536402 ± 0.002096 |
| GSAT | cooccur | 0.549610 ± 0.001981 | 0.621602 ± 0.001709 | 0.548676 ± 0.001592 |
| GraphCare | star | 0.454893 ± 0.003172 | 0.567373 ± 0.002769 | 0.429243 ± 0.000904 |
| GraphCare | cooccur | 0.454953 ± 0.003731 | 0.566658 ± 0.001361 | 0.429087 ± 0.002008 |

**Bunlar aynı split/topology/label ekseninde tarihsel olarak karşılaştırılabilir; tam eşit-input mimari yarışı değildir.** Native ProtGNN/GSAT hub'ında 132 ek demografik/sayısal kanal bulunurken GraphCare yalnız concept bilgisini kullanıyor. Training budget ve loss da tam eşit değil. GSAT'ın test F1 liderliği ile ProtGNN'in balanced-accuracy liderliği farklı soruları cevaplar. Bu tabloyu yeni common-input validation skorlarıyla çıkarıp “şu kadar iyileştik” demek geçersizdir.

Manifest kontrolü: bütün hücrelerde completed, aynı 30 sınıf sırası ve `59.607/7.448/7.456` efektif fold sayımı; topology başına tek policy fingerprint bulundu. Policy fingerprint gerçek bütün-graf eşitliğinin tek başına kanıtı değildir; common-input için aşağıdaki ayrı materyalizasyon audit'i esastır. Bu araştırmada eski graph cache'leri yeniden oluşturulmadı.

### 1.2 Bugünkü gerçek eşit-input hedefi: henüz tek seed

`common_input_20260913/results_verified.json` ve **beş** `trials/*/*/run_manifest.json` aynı input hash'ini, star, seed1234, **30 tamamlanan epoch / 13.980 update / batch128** koşulunu doğruluyor. Checkpoint seçimi validation macro-F1 ile. Aşağıdakiler **n=7.448 validation**, test değil:

| Model / loss | Seçilmiş val macro-F1 | Accuracy | Balanced accuracy | Epoch30 F1 |
|---|---:|---:|---:|---:|
| ProtGNN / current inverse | 0.440559 | 0.506445 | 0.502145 | 0.439272 |
| ProtGNN / sqrt-inverse | 0.473967 | 0.558673 | 0.491681 | 0.469846 |
| GraphCare / CE | 0.464117 | 0.578947 | 0.443460 | 0.459858 |
| **GraphCare / sqrt-inverse** | **0.481306** | 0.573577 | 0.486248 | 0.477841 |
| GSAT / CE | 0.471834 | 0.578545 | 0.452618 | 0.463407 |

**Eşit 192-concept girdide çok-seed mean/std henüz yok.** Tek seed'in std'si “0” değildir; hesaplanabilir bağımsız tekrar sayısı yoktur. `data_quality_balance_20260914` içindeki CE/sqrt tekrarları aynı seed'de önceki 60 epoch'un probability dizilerini birebir üretmiş (`prior_control_reproduction.json`); bunları yeni seed gibi ortalamak yanlış olur.

Yeni veri/sampler çalışması da lideri değiştirmedi: aynı validation'da balanced sampler F1 **0.445076**, med005+CE **0.467051**. İkincisi **277 concept** ve daha büyük kapasite taşıyor; bu dördüncü mimari deneyine sessizce eklenmemeli. Aynı concept bilgili lojistik tanısal kontrol **0.472141** F1: GNN yerine önerilmiyor, ancak karmaşık mimarilerin henüz güçlü predictive avantaj göstermediğini hatırlatıyor. Bunların kaynağı `docs/data-quality-balance-improvement.md` ve `docs/model-performance-diagnosis.md`.

**500 kişi ayrı bir açıklama kohortudur.** `docs/graphxai-500-results.md` içindeki 500 kişi × üç model × üç gerçek GraphXAI algoritması, native star/seed1234 checkpoint'lerine aittir. Bu kohortun accuracy/fidelity sayıları ne tam test F1'ı ne de common-input çok-seed başarı kanıtıdır. Yeni adayın karşılaştırması native 500-kohort sonuçlarına değil, aynı common-input incumbent checkpoint'lerinin yeni açıklamalarına dayanmalıdır.

## 2. Mimari kararını değiştiren veri ve kod bulguları

### Sabit sözleşme

- Kaynak: `data/merged_ed.csv`; canonical split: `comparison/canonical_split.json`.
- **59.607 train / 7.448 validation / 7.456 test**, 30 sınıf. Aynı hasta üyeliği, sınıf sırası ve fold korunmalı.
- **104 ilaç + 88 complaint = 192 clinical concept**, binary presence. GNN girişindeki one-hot boyutu **193**: son kanal her hastada aynı sabit hub kimliği. “192 hasta-specific numeric kanal” gibi yorumlanmamalı.
- Hasta dışı node/KG genişletme yok; hub–concept çift yönlü star. İlaç/şikâyet/hub semantik olarak farklı düğümler olsa da ana PyG graph'ı tek categorical feature uzayında kodlanıyor; zengin bir typed EHR knowledge graph değil.
- `graphcare_analysis/adapter.py:61–81`: bütün star spokes'ları her iki yönde `patient_relation_id=2`. GraphCare modelinin `num_rels=3` ile kurulması, bu koşulda üç aktif ilişki olduğu anlamına gelmiyor. `common_input_improvement.py:128–132` PyG'ye yalnız one-hot `x`, `edge_index`, `y` aktarıyor.
- `input_contract_and_audit.json`: **74.511 graph** için efektif feature/edge parity audit'i kayıtlı; hub-only train/validation/test **552/59/58**. Bu kişileri düşürmek yasak; sahte clinical concept eklenmemeli.

Bu araştırmada source, split ve common input dosyasının SHA-256'sı yeniden hesaplanıp korunmuş audit/manifest değerleriyle eşleştirildi:

| Artifact | SHA-256 |
|---|---|
| Source CSV | `95b055d53c5f879e5a5784b9d44cd3e6f8b0a073798fadc6bb25e5a228eaaf8b` |
| Canonical split | `33f6cd71399ab2605948d66f80902030cc0b36ae1017bdde551f4504e4bf82b0` |
| Common input NPZ | `7f90c512655d188187aaccc8ae52523141286f53c264ccf4e9ae2ad0b9c85d99` |

### Bilgi sınırı ile optimizasyon sınırını karıştırmayın

1. **Exact-input çakışması gerçek:** `performance_diagnosis/signature_ambiguity.json` train'de **1.349** farklı-label signature grubunda **19.391** kişi; validation'da **4.632/7.448** kişi train'de görülmeyen concept kombinasyonunda. Aynı gözlenebilir graph'a aynı deterministic model farklı hasta kimliğine göre farklı bilgi atayamaz. Daha güçlü GNN, olmayan doz/süre/şiddet bilgisi yaratmaz. Ancak bu sayımlardan validation/test Bayes tavanı hesaplanamaz; train signature-majority accuracy'si de hedef skor değildir.
2. **Küçük graph önemli:** `target_and_small_graphs.json`, train'de **22.669** graph'ın hub dahil en fazla üç node içerdiğini gösteriyor. Star'da iki farklı concept arasındaki yol iki hop; çok derin ağ için uzun-range gerekçesi zayıf. Üçüncü katman yeni node erişimi değil, ek nonlinear yeniden işleme getirir. Bu geometrik çıkarım, over-smoothing tanısı değildir.
3. **GraphCare temporal mekanizması bu veride daralıyor:** gerçek builder `max_visit=1`; `model.py:51–73` visit-axis softmax. Korunmuş `synthetic_graphcare.json` alpha=1, alpha prediction-gradient=0 ve graph-içi edge attention range=0 gösteriyor. Doğrudan EHR kolu mean embedding üzerinde eval'da affine/linear katkı; graph kolu yine nonlinear. Daha iyi readout hipotezi makul, fakat neden-sonuç eğitim ablation'ı yok.
4. **ProtGNN kusuru var; eski best sonucunu onunla açıklayamıyoruz:** `synthetic_pyg.json` 1/2/3-node MCTS root'unun pozitif gerçek similarity yerine 0 döndürdüğünü doğruluyor. Mevcut common best checkpoint'ler projection öncesi; bu hata best F1'ın düşük olmasının kanıtlanmış nedeni değil. Yeni adayın üstünlüğü bu kusuru sömürerek sunulmamalı. Düzeltme onaylanırsa regression ve izole projection kontrolü sonrası güçlü incumbent yeniden ölçülmeli; bu araştırma kodu düzeltmedi.
5. **Loss önemli ama tek başına çözüm değil:** train sınıf sayısı en büyük **7.224**, en küçük **322**. Sqrt-weighting iki eski GNN'de F1'ı artırmış; sampler minority recall'ı artırırken F1'ı düşürmüş. Yeni modele özel class balancing verip incumbent'ları eski loss'ta bırakmak mimari kazancı şişirir.

**Klinik sınır:** `temporal_clean=false`, `raw_to_model_train_only=false`. Kaynak snapshot'ta stay lineage/cutoff yok; whole-source medication/complaint filtreleri ve stay-wide medication toplama var. Yeni loss veya GNN bunları temizlemez. Bu çalışma yalnız *medication-history-available source-snapshot diagnostic* kapsamındadır; erken triyajda güvenli klinik tanı önerisi değildir.

## 3. Adayların derin değerlendirmesi

### A. Sığ residual GIN + çok-ölçekli JK — önce bunu deneyin

**Birincil literatür.** GIN, komşu multiset'ini sum + MLP ile işleyen bir message-passing GNN'dir. Teorik sonucu belirli injectivity varsayımlarında WL-ayrıştırma gücüyle ilgilidir; klinik genelleme garantisi değildir. Makale MUTAG/PTC/NCI1/PROTEINS ve sosyal graph-classification görevlerini inceler; MedGNN'in dengesiz 30-tanı problemi yoktur.[1][10]

JK, farklı katmanların komşuluk ölçeklerini birleştirir; özgün çalışma sosyal, bioinformatics ve citation ağlarında değerlendirilmiştir. GIN makalesinin graph readout'u da bütün katmanların graph temsillerini birleştirir. Dolayısıyla “GIN'e JK ekledik, yeni yöntem icat ettik” dememeliyiz.[3][10]

**Bu veriye uyarlama hipotezi.** Aynı 193-one-hot encoder, başlangıç için mevcut GSAT predictor'ının üç GIN katmanı ve trainable epsilon ayarı korunur. Önce deterministic, kapısız predictor; sonra tek tek sum readout, katmanlar arası residual ve katman 0/1/2/3 temsillerinin JK-concat readout'u denenir. Graph temsili üzerindeki classifier bir GNN head'idir; ayrı tabular tahminci veya ensemble kolu değildir. İlk sürümde temporal attention, prototype loss, stochastic bottleneck, yeni pretraining veya bilgi ekleyen feature kolu yok.

- İlk hop hub'a concept kümesini taşır; ikinci hop concept'lere diğer concept'lerin bağlamını geri iletir. Bu yol, binary-concept ilişkilerini nonlinear öğrenmek için yeterli bir başlangıçtır.
- Residual/JK, son katmanın erken concept bilgisini yeniden yazmasına karşı alternatif erişim sağlar. Bu bir optimizasyon/temsil hipotezidir; mevcut modellerin over-smoothing yaşadığı ispatlanmış değildir.
- Mean yerine sum, normalize edilmeyen concept yükünü readout'a daha doğrudan taşır. **Ancak burada benzersiz concept kimlikleri ve sabit hub var: mean pooling'in cardinality'yi bütünüyle sildiği söylenemez.** GIN makalesi bile bazı farklı-feature rejimlerinde mean'in güçlü olabileceğini ve bazı veri kümelerinde mean readout kullandığını açıklar. Sum üstünlüğünü teoriden peşinen çıkaramayız.[10]
- Hub-only graph'ta GIN self-term'i ve classifier bias'ı finite bir prior üretebilir; hastayı atmak gerekmiyor. Bu rejimde komşu mesajı bulunmadığını açıkça işaretlemek gerekir.

**Neden otomatik GINE değil?** GINE komşu mesajına edge feature ekler; PyG'de `MessagePassing` tabanlı, `edge_dim` destekli operatör var.[11] Fakat bugünkü star'da aktif ilişki tek tip. Dört deterministik yön/tür etiketi (ilaç→hub, hub→ilaç, complaint→hub, hub→complaint) mevcut node ID'lerinden türetilebilir; bu yeni klinik bilgi değil, yeni inductive bias olur. Bu nedenle GINE/typed mesaj ancak **ayrı ablation** ve bütün adaylara açık aynı türetim sözleşmesiyle denenmeli; ilaç–hastalık ontolojisi veya yeni concept–concept kenarı eklenmemeli.

**Uygulama/maliyet.** GIN'in resmi PyTorch kodu açık; README eski Torch sürümlerini ve orijinal CV seçiminin MedGNN'den farklı olduğunu belirtiyor. Onun eğitim protokolünü kopyalamak yerine mevcut PyG predictor ve resmi JK/GINE operator arayüzleri kullanılabilir.[2][11][12] Küçük graph'ta sparse message passing yeterli; GSAT'ın extractor + gated predictor maliyetinin ortadan kalkması muhtemel avantajdır, süre kazanımı ölçülmedi. İlk incelemede GIN/GINE/JK için paket kurulmadı ve forward çalıştırılmadı.

**Neden kazanamayabilir?** Aynı bilginin LR düzeyinde yeterince kullanılıyor olması, görev için sum'ın gereksiz olması, JK head kapasitesi nedeniyle overfit, ya da GSAT regularizasyonunun gerçekten yararlı olması. B0 kapısız GIN mevcut GSAT'ı geçmezse bottleneck'i suçlamak destek bulmaz.

### B. PNA — farklı yayımlanmış GNN için en iyi ikinci seçenek

**Birincil literatür.** PNA aynı node'a gelen mesajları mean/max/min/std gibi tamamlayıcı aggregators ve degree-scalers ile birlikte özetler. Makalede graph-theory görevlerinin yanında **ZINC/MolHIV** ve **CIFAR10/MNIST graph** benchmark'ları var. Bu kanıt mesaj dağılımını zenginleştirmeyi destekliyor; tanı macro-F1'ına sayısal aktarım yapmaz.[4]

**MedGNN hipotezi.** Hub'ın aldığı concept embedding'lerinin ortalaması yanında çeşitlilik ve uç değerleri korumak, aynı ortalamaya yakın ama farklı klinik kombinasyonları ayırabilir. Degree scaler, az-concept/çok-concept girdilerini farklı işlemesine izin verir. Bilgi yine aynı hasta graph'ından gelir; başka hastalardan inference-time mesaj veya dış KG yok.

**Önemli karşı-kanıt/riski:** star yapraklarının tek komşusu hub'dır. Degree-1 komşulukta mean/min/max aynı mesajdır; dispersion bilgi taşımaz veya sayısal epsilon tabanına iner. PNA'nın ek aggregation gücü büyük ölçüde hub'da yoğunlaşır. Üstelik binary kimlikleri GIN zaten güçlü biçimde özetleyebilir. Bu nedenle ilk tercihten önce pahalı geniş PNA taraması haklı değil.

**Kontrollü deneme.** GIN ile aynı encoder/readout/katman sayısında tek PNA bloğu ailesi; `mean,min,max,std`, `identity,amplification,attenuation`, tek tower başlangıç paketi. Degree histogram **yalnız train graph'larından**; PyG API'si de `deg` girdisini training-set histogramı olarak tanımlar.[13] “Tüm graph'ları zaten biliyoruz” diyerek test degree dağılımıyla scaler fit edilmemeli. Hub-only degree0, empty aggregation ve std'nin finite gradient'i ayrıca sınanmalı.

**Maliyet/erişim.** Yazar kodu PyTorch/DGL/PyG implementasyonlarını sağlıyor; PyG `PNAConv` hazır.[5][13] Ancak aggregators×scalers intermediate tensörleri ve pre/post MLP'leri sade GIN'den daha pahalı olabilir. Aynı hidden-width aynı parametre/compute demek değildir. Ölçülmüş parameter count, update zamanı ve peak memory olmadan “eşit bütçe” iddiası kurulamaz.

### C. R-GCN — heterojenlik var, fakat aktif relation az

R-GCN farklı relation türleri için ayrı lineer mesaj dönüşümleri ve normalize komşu toplamı kullanır; basis/block decomposition çok-relation parametre yükünü azaltır. Özgün hedefi knowledge-base **entity classification ve link prediction**, örneğin AIFB/BGS/AM ve FB15k-237'dir; bağımsız hasta-graph sınıflandırması değildir.[6][17]

MedGNN'e uyarlamak için mevcut concept türü/yönünden relation tipi türetip graph-level readout eklemek gerekir. İlaç mesajı ile complaint mesajını farklı parametreyle hub'a taşımak makul bir bias'tır. Fakat mevcut kimlik encoder'ı bu ayrımı zaten öğrenebilir; tek relation'lı star üzerinde “heterogeneous GNN” etiketi tek başına ek sinyal getirmez. Çok az relation için basis compression ana ihtiyaç değildir. GIN'in nonlinear multiset dönüşümünü bırakıp relation-specific lineer mesajlara geçmek de otomatik iyileşme değildir.

Referans kod Keras/Theano tabanlı, çok eski sürüm beklentileri açıkça belirtilmiş; onu mevcut ortama kurmak önerilmiyor.[7] Modern PyG uyarlaması yapılabilir fakat bu görevde graph head ve tür türetimi ilk kez sınanacaktır. Edge-mask relation filtering/sıralaması ayrıca denetlenmeli; yalnız `MessagePassing` sınıfından türemek tam açıklama uyumu garantisi sayılmaz. **Önce GIN/GINE typed-versus-untyped ablation'ı daha ucuz ve daha tanısal.**

### D. GraphGPS — mantıklı genel mimari, burada ilk yatırım değil

GraphGPS üç bileşeni birleştirir: positional/structural encoding, gerçek kenarlarda local MPNN, global attention. Bu nedenle tabular ensemble değildir; ancak saf local message-passing'den daha geniş bir **GNN+Transformer** modelidir.[8][18] Makale moleküler graph'lar, görsel graph'lar, kod/protein ve büyük graph benchmark'ları dahil farklı alanlarda değerlendirilmiştir; küçük graph desteği var, “yalnız büyük graph'a yarar” demek yanlış olur.[18]

Buradaki asıl sınırlama büyüklük değil **ihtiyaç**: star'da her concept diğerine iki hop uzakta. Global attention hub sıkıştırmasını aşarak concept çiftlerini doğrudan karşılaştırabilir, fakat aynı etkinin sığ nonlinear GIN/PNA ile yeterince elde edilip edilemediği bilinmiyor. Aynı star boyutunda yaprakların yapısal simetrisi nedeniyle PE/RWSE yeni hasta bilgisi yaratmaz; rastgele eigenvector basis'ini hasta ayrımı gibi kullanmak istenmez.

Standart dense attention quadratic, lineer maliyet iddiası seçilen Performer/BigBird gibi efficient attention mekanizmasına bağlıdır; her GraphGPS konfigürasyonu otomatik lineer değildir.[18] Resmi repo PyG/GraphGym framework'ü ve belirli eski dependency kurulumu istiyor; moleküler büyük-model sürelerini MedGNN süre tahmini diye kullanmıyorum.[9]

En önemli açıklama riski: real-edge mask yalnız local MPNN yolunu kısabilir; global attention graph'taki diğer düğümlere hâlâ erişir. Yüksek accuracy elde edilse bile mevcut edge explanation metriği tüm predictor'ı açıklamaz. Bu aday ancak daha basit modellerde umut verici fakat sınırlı kazanım sonrasında, global-on/off ablation ve açık açıklama protokolüyle düşünülmeli.

### E. MUSE ve GRAPE adı: sağlık literatürü neden doğrudan çözüm değil?

**İlgili MUSE**, Wu ve arkadaşlarının ICLR 2024 *Multimodal Patient Representation Learning with Missing Modalities and Labels* çalışmasıdır; başka alanlardaki aynı kısaltmalı modeller değil. MUSE hasta ve modalite düğümleri, modality-feature edge'leri ve edge-dropout'lu iki GNN görünümü arasında unsupervised/supervised contrastive loss kullanır. Makale edge-attribute GraphSAGE omurgasını ve her minibatch için bipartite graph oluşturmayı açıklar.[14]

Klinik alan örtüşmesi gerçek: MIMIC-IV/eICU mortalite ve readmission, ADNI progression. Ama ICU görevleri **binary**, metrikler AUC-ROC/AUC-PR; girdiler diagnosis/procedure/lab/notes/vitals gibi daha zengin modalitelerdir. MUSE+'ın ek avantajında **etiketsiz hastaları ekleme** vardır. Bunlar bizim 192-concept, 30-class macro-F1 görevine aktarılabilir skor kanıtı değildir. Makaledeki bootstrap std de bizim model-seed std'mizle aynı belirsizlik türü değildir.[14]

Bu modeli özgün haliyle getirmek **bağımsız hasta-star graph → hasta–modalite batch graph** değişimidir. Yeni graph semantiği, batch composition/inference protokolü ve fit kapsamı gerekir. Eldeki binary 0 değerleri de gözlenmemiş mi yoksa gerçekten yok mu ayırmıyor; rastgele masking yapıp “gerçek missing modality giderildi” denemez. Canonical etiketli hastalarımızı genişletmeden MUSE+'ın koşulları oluşmaz. Repo açık ve preprocessing/train komutları mevcut, fakat mevcut MedGNN harness'ine tak-çalıştır değil.[15]

Burada ilgili **GRAPE**, You ve arkadaşlarının NeurIPS 2020 *Handling Missing Data with Graph Representation Learning* çalışmasıdır: observation/feature bipartite graph, feature-value edge'leri, imputation için edge prediction ve label için node prediction. NeurIPS çalışmasının bildirdiği MAE iyileşmesi 30-sınıflı diagnosis macro-F1 kazancı değildir.[16] Bu nedenle GRAPE/MUSE sağlık veya missing-data anahtar sözcükleri taşıdığı için ilk aday seçilmedi.

**Pretraining de başlangıç paketine eklenmiyor.** GNN pretraining literatürü moleküler/protein görevlerinde sonuç gösterirken bazı naif stratejilerde negative transfer de bildiriyor.[19] Harici pretrained embedding/ontology eklemek aynı-input ve compute sözleşmesini değiştirir. Önce supervised GNN kontrolünün neyi başarabildiği görülmeli.

## 4. Sınırları belli deney tasarımı — henüz uygulanmadı

Bu bölüm **araştırma önerisi**, eğitim yetkisi veya çalışan yeni CLI kaydı değildir. Aşağıdaki hücrelerin hiçbiri bu görevde eğitilmedi. Her aşamaya ancak kullanıcı uygulama/eğitim kapsamını onayladıktan sonra geçilmeli.

### Aşama 0 — bilimsel ve teknik kapı

1. Yukarıdaki input/split/source hash'lerini, sıralı label/fold ve graph fingerprint'lerini dondur. Mevcut NPZ'yi kullan; med005, native hub, KG, temporal veri, relabel veya hasta elemesi yok.
2. Ortak ölçüm `shared/lib/metrics.py`; primary **macro-F1**, secondary balanced accuracy, sınıf başına precision/recall/F1/support; accuracy/top-3 yardımcı. Ağırlıklar yalnız train count'tan fit; validation gerçek dağılımında kalır.
3. Yeni GNN için node permutation, tek graph vs mixed batch eval eşliği, hub-only/singleton finite forward/backward, bağlı edge-mask gradyanı ve checkpoint replay doğrulanır. Bunlar başarı skoru değil wiring kanıtıdır.
4. Model native mekanizmaları korunur: GSAT curriculum, ProtGNN warm-up/projection/aux-loss. Doğrulanmış ProtGNN root-score kusurunun düzeltilmesi ayrı onaylı regression işidir; düzeltilmiş güçlü incumbent kullanılacaksa kodu **deneyden önce** dondur ve eski/düzeltilmiş baseline ayrımını açık tut. Bu yapılmazsa sonuç yalnız mevcut kusurlu implementasyona karşı üstünlük olabilir, yöntem ailesine karşı değil.
5. Başlangıç için bütün yeni predictor varyantlarında aynı optimizer ayarı, normalizasyon, dropout ve seed; önerilen sabit küçük iskelet **hidden128 / üç message-passing katmanı**, mevcut GIN predictor ayarları. Yeni adaylar için önerilen **250.000 toplam trainable-parametre tavanı** validation görülmeden uygulanır; bu sayılar ölçülmüş yeni model büyüklüğü değildir. Aynı width şartını capacity gerekçesiyle sessizce bozmamak gerekir. İlk parametre hesabı ve kısa timing probe'u sonrası PNA'yı sığdırmak için width değişirse bunu açık bir architecture-package karşılaştırması olarak etiketle; saf aggregator etkisi iddiasını kaldır. Boyut seçimi analitik parametre sayımından yapılır, yeni validation HPO denemesi açılmaz; seçilen width bütün seed/loss tekrarlarında sabit kalır.

### Aşama 1 — en fazla beş mimari pilot; tek seed, sabit CE

Her hücre seed1234, star, aynı train/validation, batch128, **30 epoch / 13.980 update**, early stopping yok. **Aynı ağırlıksız CE** kullan; ağırlık kazanımını backbone kazanımına katma. Validation ilk maksimum macro-F1 checkpoint'i seçer; epoch30 ayrıca raporlanır.

| Hücre | Önceki kontrole göre tek değişken / açık paket | Cevapladığı soru |
|---|---|---|
| B0 | Mevcut GSAT GIN predictor, gate yok, KL yok; mean readout | Predictive backbone, GSAT'ın joint bottleneck paketinden iyi mi? |
| B1 | B0, yalnız mean→sum readout | Readout normalizasyonu sınırlayıcı mı? |
| B2 | B1, yalnız katmanlar arası residual bağlantı | Erken temsil/optimizasyon yolu korununca yarar var mı? |
| B3 | B2, yalnız katman 0/1/2/3 JK-concat + aynı tür linear head | Çok-ölçek erişim yararlı mı; ek head kapasitesi ne kadar? |
| B4 | B2'de GIN yerine tek önceden tanımlı PNA operator paketi | Farklı aggregation ailesi ek değer getiriyor mu? |

B0–GSAT karşılaştırması **gate+bilgi-loss paketini** kaldırır; sadece KL'nin nedensel etkisini ayırmaz. “Beta=0, stochastic gate açık” başka bir deneydir; beşli bütçeye gizlice eklenmez. B3 head boyutunu, B4 parametre sayısını değiştirir; bunlar raporlanmadan saf mekanizma kanıtı sayılmaz. Residual/JK/normalizasyon/dropout/loss hepsini bir anda değiştiren tek büyük model denemesi önerilmiyor.

Bu aşama en fazla **69.900 update**; notebook/ham-veri rebuild veya yeni HPO yok. Önceki aynı CE incumbent koşulları ancak code/input/budget binding uyuyorsa pilot referansı olabilir; bağ farklıysa yeni aynı-budget kontrol olmadan tarihsel farka neden atfedilmez. R-GCN/GINE/GraphGPS/MUSE bu beşli pilot bütçesinde yok.

**İlerleme kuralı:** en yüksek validation macro-F1'lı tek predictive challenger seç; çok yakın sonuçta (önceden tanımlı ≤0.005 F1 farkı) daha sade/düşük maliyetli aday tercih et. Bu 0.005 eşiği pratik seçim kuralıdır, istatistiksel anlamlılık sınırı değil. F1 artışı yalnız validation epoch seçimine dayanıyor ve epoch30/rare-class raporu kötüleşiyorsa aday “umut verici” diye genişletilmez. Bu tarama CE ile sınırlıdır; adayın sqrt-loss altında en iyi olmayabileceği açık kısıttır.

### Aşama 2 — önce en güçlü incumbent, sonra üç yöntemin tamamı

**Önce altı hücrelik kapı:** seçilmiş challenger ve **GraphCare sqrt-inverse**, aynı sqrt-inverse CE, seed **1234/1235/1236**, 30 epoch/batch128. Bu paired karşılaştırma **83.880 update** üst sınırı taşır. Üç seed ortalama F1 farkı **+0.01'in altında** ise bu ucuz araştırma turunu durdur; +0.01 umut kapısıdır, “ciddi üstünlük” değildir. Tek seed sıçramasıyla devam etme. Bu erken durdurma diğer loss altında olası bir kazancı dışlamaz; kaynak tasarrufu kararını açıkça sınırlar.

Kapı geçilirse ana validation matrisi:

- **Dört aile:** finalist GNN, GSAT, GraphCare, ProtGNN.
- **İki ortak classification-loss politikası:** `none` ve `sqrt(N/(30*n_c))`. Aux-loss/curriculum yalnız ilgili incumbent'ta korunur. ProtGNN `none`, eski `current inverse` ile aynı değildir; yeni bir kontrollü baseline'dır.
- **Üç aynı model seed:** 1234/1235/1236; canonical split seed'i değiştirilmez.
- **30 epoch, batch128, aynı epoch/update tavanı**, model başına best-val ve epoch30; maksimum **24 hücre / 335.520 update**. Önceki altı kapı hücresi aynı dondurulmuş binding altında bu matrisin altkümesidir; tekrar eğitilerek bağımsız tekrar gibi sayılmaz.
- Her aile için loss, **üç-seed ortalama validation macro-F1** ile seçilir. Hem aynı-loss mimari tablosu hem her ailenin seçilmiş güçlü varyantı raporlanır. Incumbent'lara da aynı loss arama hakkı verilir; finalist yalnız eski zayıf kontrollerle yarıştırılmaz.

Bu matris otomatik başlamaz; ilk beşli pilot başarısızsa toplam araştırma maliyeti orada durur. Duvar saati tahmini verilmedi; ortamlar ve efektif parametreler farklı. Epoch/update eşitliği **süre/FLOP/kapasite eşitliği değildir**. Zaman, peak memory, toplam/aktif parametre sayısı ve model-native schedule tamamlanması raporlanmalı. Bir modelin programı bu tavana sığmazsa post-hoc ona özel ekstra epoch vermek yerine ayrı ve eşit genişletilmiş protokol gerekir.

### Mekanizma doğrulaması ve falsification

- Her seçilmiş checkpoint için aynı **eval-mode train ve validation** CE/F1; training-mode dropout'lu objective ile eval CE arasındaki fark “generalization gap” diye kullanılmaz.
- Bütün 30 sınıfın precision/recall/F1/support'u; no-complaint, hub-only, 1–2 concept, 3+ concept ve seen/novel-signature coverage. Signature sözlüğü train'den; farklı sınıf karışımları nedeniyle subgroup farkını nedensel kazanım sayma.
- Graph mesajlarının işe yarayıp yaramadığını ölçmek için finalized validation checkpoint'inde gerçek kenarlar / sıfır kenar mesajı müdahalesi ayrı tanısal kontrol olabilir; bu eğitimli no-message baseline ile aynı test değildir. Eğer full ve no-message logits aynıysa “ilişkileri öğrendik” iddiası desteklenmez. Böyle bir müdahale yeni hasta bilgisi eklemez, ancak dağılım dışı olabileceği açıklanmalı.
- Kazanç yalnız readout'taysa JK/residual'ı final modele zorla ekleme. B0 zaten en iyiyse sade predictive GIN tutulabilir; daha karmaşık ismi savunmak için model seçilmez.
- PNA paketinin açık avantajı çıkarsa sonraki turda aggregators ve scalers ayrı ablation olabilir; mevcut beşli tarama bu iki etkiyi birbirinden ayırmış sayılmaz.

## 5. “Ciddi fark” ne demek?

**Önerilen, önceden kabul edilmesi gereken eşik:** aynı-input, aynı-budget, güçlü incumbent'ların en iyisine göre **en az +0.03 mutlak macro-F1**. Bu üç yüzde puan demektir, %3 göreli artış değil. Örneğin tek-seed tarihsel 0.481306 referansa eklenince 0.511306 çıkar; **bu bir tahmin/ulaşılabilir skor ilanı değil, eşik aritmetiği örneğidir**. Gerçek hedef yeni matched multi-seed incumbent ortalamasından hesaplanacak.

Yeterli kanıt için birlikte arananlar:

1. Üç paired seed'in ortalama ΔF1'ı ≥+0.03; tek en iyi seed değil. Seed başına delta ve ddof=1 std açık verilir; üç tekrarla belirsizlik hâlâ geniş olabilir.
2. Aynı kişilerden paired, class-stratified bootstrap ile ΔF1 için %95 aralık; örneklenen kişi indeksi iki modelde ve bütün seed'lerde ortak tutulur. Bu, **sabit split ve seçilmiş modeller koşulunda hasta belirsizliği**; seed/split/site belirsizliğinin tümünü kapsamaz. Validation'da model/loss seçimi yapıldığı için bu aralık confirmatory test diye sunulmaz.
3. Önceden önerilen guardrail: balanced accuracy ortalamasında **0.01'den büyük mutlak kayıp olmaması**; rare-class P/R/F1 raporunda ciddi precision/recall çöküşlerinin açık tartışılması. Bu tolerans kullanıcı tarafından kabul edilecek araştırma kuralıdır, klinik güvenlik standardı değildir. Rare-class desteklerinin küçüklüğü nedeniyle sadece birkaç örnek oynayabilir.
4. Mimari, loss ve seed politikası dondurulduktan sonra **n=7.456 test** üzerinde finalist ve matched güçlü incumbent'lar bir kez değerlendirilir; seed ensemble varsayılan değil, üç ayrı seed sonucu raporlanır. Test sonucu yeniden HPO başlatmak için kullanılmaz.

**Test'in tarihsel olarak görülmüş olması unutulmamalı:** eski test skorları mevcut dokümanlarda yayımlanmış durumda. Yeni tuning'de test'i kapalı tutmak yeni seçim sızıntısını önler; eski test'i geriye dönük tamamen görülmemiş holdout'a dönüştürmez. Klinik genelleme iddiası için ayrıca cutoff-safe temporal/dış kohort doğrulaması gerekir.

Kazanım güveni şu anda **düşük**; önerilen deneyin tanısal değerine güven **orta**. Eşit girdide modeller plato yapabilir. +0.03 çıkmazsa doğru sonuç “bu girdide büyük mimari kazanım gösterilmedi”dir, daha yeni isimlerin mutlaka işe yarayacağını varsaymak değil.

## 6. Performans ile açıklanabilirlik: aynı hedef değiller

Residual GIN/PNA **intrinsic prototype veya sparse-rationale açıklayıcı model değildir**. Prediction odaklı tasarım, ProtGNN/GSAT'ın açıklama mekanizmasını bırakabilir; bu hem maliyet hem bilimsel amaç bakımından trade-off'tur. GSAT regularizasyonunun kaldırılmasının mutlaka accuracy artırdığı gösterilmedi. Diğer yandan yüksek fidelity, yanlış tanıya güvenilir klinik gerekçe üretildiği anlamına gelmez: kötü bir classifier da kendi yanlış kararını sadakatle açıklayabilir.

Repo gerçekten **GradExplainer, IntegratedGradExplainer ve GNNExplainer** çalıştırabiliyor. `shared/lib/graphxai_standardized.py:19–21,34–40,69–100` gerçek vendored GraphXAI çağrılarını ve modern PyG `set_masks/clear_masks` uyarlamasını gösteriyor. GNNExplainer node skoru incident edge-mask kütlesinden türetiliyor. Önceki 500-kohort sonuçları bu üç eski model için çalıştırma kanıtıdır; yeni modele otomatik taşınmaz.

**GIN/GINE/JK uyum yolu:** continuous one-hot giriş → PyG `MessagePassing` → graph-level logits wrapper. Grad/IG discrete argmax node ID rekonstrüksiyonundan geçirilmemeli; input encoder differentiable kalmalı. GINE relation IDs sabit metadata olarak taşınmalı; maskelenmiş feature'dan yeniden argmax ile tür çıkarılmamalı. Maskenin ReLU/MLP öncesi/sonrası gerçek mesaj katkısını etkilediği ve `all-ones` mask'te unmasked logits eşliği doğrulanmalı. Her katmanda finite, bağlı edge-mask gradient aranır; finite sıfır gradient ile `None` farklıdır.

**PNA özel kontrolü:** PyG operatörü `MessagePassing` tabanlı olsa da zero-message masking, degree normalizasyonu veya min/max altında gerçek edge deletion ile eşdeğer değildir.[13] Sayısal mask kararlılığı ve müdahalenin neyi değiştirdiği yazılmalı. R-GCN'de relation bazlı edge seçimi/sırası, GraphGPS'te global attention bypass'ı ek risk; wrapper adı uyumluluğu kanıtlamaz.

**Hub-only politika:** edge gradient `not_applicable_edgeless`; kişi tutulur, empty entropy finite sıfır olmalı. Incident-edge node attribution dejenere olabilir; sahte importance veya daha kolay hasta ile replacement yok.

**Final açıklama aşaması:** predictive seçim tamamen bittikten sonra, aynı **500 canonical test kişisi**, aynı common-input graph'ları ve aynı checkpoint-selection politikasıyla **dört model × üç named explainer** değerlendirilir. Önceki 500 kişilik kohort membership'i korunabilir fakat yeni common-input checkpoint'leriyle eski native modellerin açıklamalarını karıştırma. IG32/GNNExplainer50 ve mevcut node zero-fill fidelity protokolü sabit tutulur; sparsity, fidelity±, stability ve prediction doğruluğu ayrı raporlanır. Mevcut fidelity'de kenarlar sabit, node özellikleri sıfırlanır; literal node/edge deletion deneyi değildir. Validation pilotlarını seçmek için bu test kohortunun açıklama skorları kullanılmaz.

## 7. Sonuç ve kanıt sınırı

**Bu repo için en mantıklı ilk yatırım yeni bir “klinik GNN” markası değil, sade GIN kontrolünden türeyen küçük residual/çok-ölçekli predictive GNN'dir; yayımlanmış farklı operator isteyen ikinci seçenek PNA'dır.** Her ikisi aynı küçük hasta graph'ında gerçek message passing yapar. MUSE/GRAPE'ın alan yakınlığı graph/input sözleşmesi farkını, GraphGPS'in genel gücü ise burada ek karmaşıklığın gerekliliğini ortadan kaldırmıyor.

İlk kazanım kaynağı **bilginin daha iyi özetlenmesi ve sınıflandırma amacıyla uyumlu optimizasyon** olabilir. Bunun sınırı **eksik/çakışan concept temsili**. Aynı dondurulmuş girdide güçlü multi-seed kontroller geçilemiyorsa, sonraki ayrı araştırma selected-stay lineage, availability cutoff ve bütün yöntemlere eşit uygun klinik bilgi olmalıdır; model değiştirmek temporal veriyi temizlemez.

### Bu araştırmada yapılan doğrulama

- Üç istenen güncel teşhis/iyileştirme dokümanı, korunmuş sonuç/manifest/audit dosyaları ve aktif input/model/GraphXAI kodları okundu.
- 18 historical metric dosyası ile TSV eşleştirildi; altı model×topology grubunun gerçek üç-seed mean/sample-std'si yeniden hesaplandı. Schema farkı (`test` nested) açıkça ele alındı.
- Common-input manifest'lerinde yalnız tek seed olduğu, veri-kalite tekrarlarının bağımsız seed olmadığı ve eski test / yeni validation / 500 açıklama kohortu ayrımı doğrulandı.
- Source/split/common-input SHA-256 eşleşmeleri yeniden hesaplandı; hasta satırı/prediction veya ham klinik metin rapora alınmadı.
- Kaynaklar güncel web aramasıyla bulunup **`web_extract` ile gerçekten açıldı**; önemli GIN/MUSE/GraphGPS bölümleri extraction'ın kesilmiş ortasından ayrıca okundu. Resmi paper/code/API kaynakları aşağıda; başka benchmark skorları MedGNN skoruna çevrilmedi.
- **Yeni model uygulanmadı, eğitim/fit/inference yapılmadı; kod/veri/cache/checkpoint değiştirilmedi.** Repo içinde bu araştırmanın tek yazısı bu rapor. Önceden mevcut cleanup/dirty değişikliklere dokunulmadı; commit/push yok. Atıf ledger/evidence kayıtları yalnız runtime cache'inde tutuldu.

### Yerel kanıt rehberi

- `comparison/standardized/benchmark_results_summary.tsv`
- `comparison/standardized/results/{protgnn,gsat,graphcare}/{star,cooccur}/seed_*/{run_manifest.json,test_metrics.json veya metrics.json}`
- `comparison/standardized/common_input_20260913/{input_contract_and_audit.json,results_verified.json,verification.json}` ve `trials/*/*/run_manifest.json`
- `comparison/standardized/performance_diagnosis/{signature_ambiguity.json,target_and_small_graphs.json,synthetic_pyg.json,synthetic_graphcare.json}`
- `comparison/standardized/data_quality_balance_20260914/{results_verified.json,prior_control_reproduction.json}`
- `docs/model-performance-diagnosis.md`, `docs/common-input-improvement.md`, `docs/data-quality-balance-improvement.md`, `docs/graphxai-500-results.md`
- `comparison/standardized/common_input_improvement.py`, `gsat_analysis/models/gin.py`, `graphcare_analysis/{adapter.py,model.py,run.py}`, `shared/lib/graphxai_standardized.py`

## Sources

[1] https://arxiv.org/abs/1810.00826
[2] https://github.com/weihua916/powerful-gnns
[3] https://proceedings.mlr.press/v80/xu18c.html
[4] https://proceedings.neurips.cc/paper_files/paper/2020/file/99cad265a1768cc2dd013f0e740300ae-Paper.pdf
[5] https://github.com/lukecavabarrett/pna
[6] https://arxiv.org/abs/1703.06103
[7] https://github.com/tkipf/relational-gcn
[8] https://arxiv.org/abs/2205.12454
[9] https://github.com/rampasek/graphgps
[10] https://arxiv.org/html/1810.00826v3
[11] https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
[12] https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.JumpingKnowledge.html
[13] https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html
[14] https://proceedings.iclr.cc/paper_files/paper/2024/file/f49d76cf84df83a611883c621c96d2d9-Paper-Conference.pdf
[15] https://github.com/zzachw/MUSE
[16] https://arxiv.org/abs/2010.16418
[17] https://arxiv.org/pdf/1703.06103
[18] https://arxiv.org/html/2205.12454v4
[19] https://arxiv.org/abs/1905.12265
