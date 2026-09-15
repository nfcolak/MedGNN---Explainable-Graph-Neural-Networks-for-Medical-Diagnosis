# MedGNN — model performansı neden sınırlı?

## Hüküm ve kanıt düzeyi

**Ortak concept girdisinde temel sorun yalnızca “zayıf GNN” değil: kayıplı klinik temsil, yeni concept kombinasyonlarına genelleme ve sınıf-ağırlığı/hedef-metrik uyuşmazlığı birlikte sınır koyuyor.** ProtGNN’de ayrıca dar prototype mekanizması ve doğrulanmış bir MCTS küçük-graph hatası; GraphCare’de tek-ziyaret uyarlamasının etkisizleştirdiği temporal attention var. Bunların performans üzerindeki nedensel payları ölçülmüş değildir. “Over-smoothing” veya “Bayes ceiling” teşhisi için kanıt yok.

Kapsam: canonical **59.607 train / 7.448 validation / 7.456 test**, 30 sınıf. Bu raporda bütün yeni hata karşılaştırmaları yalnız aynı **7.448 validation** kişisindedir. Train signature istatistikleri yalnız 59.607 train etiketiyle hesaplandı. Yeni eğitim, GNN hasta inference'ı, test inference, cache rebuild, dördüncü yöntem, kaynak kod düzeltmesi veya commit/push yok. Yalnız mevcut LR checkpoint'leri validation üzerinde yeniden çalıştırıldı; model mekanizması probları sentetiktir.

Kanıt işaretleri: **doğrudan** = kod + gerçek artifact / yeniden hesaplama; **mekanizma doğrulandı** = sentetik kontrollü probe, benchmark etkisi bilinmiyor; **hipotez** = karşı-olgusal eğitimle sınanmadı.

| Sıra | Açıklama | Güç / sınır | Yanlışlayacak kanıt |
|---|---|---|---|
| 1 | 192 binary concept, klinik ayrım için sınırlı ve birçok validation kombinasyonu yeni | Doğrudan temsil çakışması + LR girdi karşılaştırması; sınırlamanın toplam hata payı bilinmiyor | Aynı uygun girdide eşit-budget, çok-seed mevcut GNN ciddi artış gösterirse temsilin baskınlığı zayıflar; cutoff-safe ek bilgi faydasızsa “eksik bilgi” yorumu zayıflar |
| 2 | Classification loss sınıf dengesizliği ve macro-F1 ile uyumsuz | Aynı yöntem/seed/budget içinde tek loss faktörü değişmiş; iki yöntemde kazanım, metrik trade-off'ları | Aynı önceden seçilmiş çok-seed tekrarında fark kaybolursa tek-seed bulgusu genellenemez |
| 3 | Mevcut graph tasarımı LR'ın bilmediği hasta bilgisini getirmiyor; GraphCare temporal kolu tek ziyarette dejenere | Topoloji ve forward doğrudan; alpha=1 ve gradient=0 sentetik doğrulandı | Geçerli aynı-input backbone/readout ablation'ı LR'a karşı tekrarlı avantaj gösterirse “graph ek değeri düşük” yorumu zayıflar |
| 4 | ProtGNN prototype darboğazı / sınırlı projection ve küçük-graph skorlama hatası | Skor hatası doğrudan; en iyi checkpoint projection öncesi, dolayısıyla bu hata mevcut best skorun nedeni olamaz | Küçük-graph skorunu izole düzeltmek post-projection metriklerini değiştirmezse performans sebebi değildir; head ablation fark yaratmazsa prototype darboğazı zayıflar |
| 5 | Regularizasyon/curriculum ve doygun validation; genel büyük overfit kanıtı yok | Eğriler gerçek; train eval-mode metrikleri eksik | Aynı-checkpoint dropout-off train/val CE ve macro-F1 geniş fark gösterirse overfit; ikisi de düşük ve kontrollü regularizasyon azaltımı faydalıysa underfit desteklenir |

## 1. Karşılaştırma zemini: input ile architecture etkisini ayırın

Aşağıdaki ortak-input GNN'ler: star, seed1234, batch128, 30 tamamlanan epoch, her biri 13.980 update; best epoch validation macro-F1 ile seçilmiş. LR aynı binary concept bilgisinde C=1/lbfgs, sabit 150-iteration üst sınırı; epoch/update/capacity eşit değil, klasik tanısal baseline. GraphCare native batch32 idi; yeni current, özgün koşulun birebir kopyası değil.

| Ortak girdi modeli | Best epoch (1-based) | Macro-F1 | Accuracy | Balanced acc | Top-3 | Epoch30 F1 |
|---|---:|---:|---:|---:|---:|---:|
| protgnn / current | 15 | 0.440559 | 0.506445 | 0.502145 | 0.691192 | 0.439272 |
| protgnn / sqrt_inverse | 11 | 0.473967 | 0.558673 | 0.491681 | 0.741541 | 0.469846 |
| graphcare / current | 13 | 0.464117 | 0.578947 | 0.443460 | 0.769871 | 0.459858 |
| graphcare / sqrt_inverse | 17 | 0.481306 | 0.573577 | 0.486248 | 0.761815 | 0.477841 |
| gsat / current | 25 | 0.471834 | 0.578545 | 0.452618 | 0.766246 | 0.463407 |
| LR / none | — | 0.472141 | 0.573845 | 0.451099 | 0.767454 | — |

Özgün 18 koşulun eski test sonuçları üç-seed ortalamalarıdır; yeni validation tuning tablosuyla “düşüş/kazanım” hesabı yapılmaz. Aynı topology ≠ aynı bilgi: native PyG/ProtGNN/GSAT hub'ı 132 ek demografik/sayısal kanal taşır; GraphCare yalnız concept + sabit hub. Ortak PyG encoder 193 one-hot kanal, native PyG encoder 331 kanal; LR'ın native vektörü 324 boyut. Bu boyutlar farklı temsilleri sayar, çelişki değildir. Native→common GNN karşılaştırması girdinin yanı sıra encoding ve budget'ı da değiştirir.

Aynı validation'daki daha adil tanısal girdi karşılaştırması, önceden fit edilmiş aynı LR ailesidir:

| LR görünümü / loss | Macro-F1 | Accuracy | Balanced acc |
|---|---:|---:|---:|
| concepts / none | 0.472141 | 0.573845 | 0.451099 |
| concepts / sqrt_inverse | 0.467906 | 0.559076 | 0.489468 |
| native_diagnostic / none | 0.552493 | 0.624731 | 0.543565 |
| native_diagnostic / sqrt_inverse | 0.546886 | 0.608754 | 0.582459 |

Ağırlıksız LR'da native ek kanallar **+0.080353 macro-F1** ile ilişkili. Concept LR 107 iterasyonda yakınsamış; iki native fit 150 sınırında ConvergenceWarning vermiştir. Bu bir “temiz klinik bilgi kazanımı” değildir: native lab/vital/visit-count zaman uygunluğu sorunludur. Yine de mevcut 192 concept'in ayırt etmediği ek sinyal olduğunu gösterir; salt daha derin GNN gerekçesi değildir.

Sınıf F1 örnekleri (LR concept → native; n validation aşağıda):
- Non-ST elevation (NSTEMI) myocardial infarction, n=72: 0.339623 → 0.759124.
- Hypokalemia, n=126: 0.208589 → 0.506122.
- Sepsis, n=40: 0.160000 → 0.416667.
- End stage renal disease, n=43: 0.035088 → 0.278481.

Kanıt: `common_input_20260913/{input_contract_and_audit,results_verified,verification}.json`; önceki `performance_review_20260913/baseline/results.json`; yeni `performance_diagnosis/metrics_and_errors.json`. Girdi uygulaması `common_input_improvement.py:66–109,120–132`, native adapter `graphcare_analysis/adapter.py:175–226`; ek kanal audit'i `docs/performance-improvement-review.md:24–43`.

## 2. Train-only birebir concept signature belirsizliği

Signature = 192 binary concept'in tam üyelik kümesi. Sabit hub ve deterministic star bu kümeden ek birey ayrımı üretmez. Adet, doz, şiddet, serbest metin ve zaman sırası bu temsilde bulunmaz. Signature anahtarları veya hasta satırları aggregate dosyalara yazılmadı.

| Train istatistiği | Değer |
|---|---:|
| Train kişi | 59,607 |
| Benzersiz signature | 39,339 |
| Tek örnekli signature | 37,134 |
| Tekrarlanan signature | 2,205 |
| Birden çok farklı label içeren signature | 1,349 |
| Bu çelişkili signature gruplarındaki kişi | 19,391 |
| Her signature içi çoğunluğa karşı kalan train kişi | 6,865 |

Train kişilerin **%32.53**'si aynı gözlenebilir girdinin farklı etiketlerle görüldüğü gruplarda. Tam aynı train signature'ı paylaşan farklı kişi çiftlerinin etiket uyuşmazlığı **%31.79** (büyük gruplar daha çok çift üretir; kişi-ağırlıklı oran değildir).

Train içinde her signature'a çoğunluk etiketini atayan lookup, **52,742/59,607 = %88.48** empirical resubstitution accuracy verir. **Bu Bayes tavanı değildir; validation/test için üst sınır veya ulaşılabilir hedef değildir.** Train'in %62.30'si tek örnekli signature olduğu için otomatik ezberlenir. Yalnız tekrarlanan signature satırlarında karşılık gelen train oranı %69.45. Sonlu-örnek çakışmaları gizli klinik değişkenleri veya etiket gürültüsünü birbirinden ayırmaz.

| Validation coverage | Kişi |
|---|---:|
| Train'de görülen exact signature | 2,816 |
| Train'de hiç görülmeyen exact signature | 4,632 (%62.19) |
| Train'de çelişkili label'lı signature'a düşen | 2,290 |
| Signature görülmüş, gerçek val label o train grubunda yok | 346 |

Yalnız train'den dondurulmuş signature-majority, yeni signature'da global train majority fallback ile validation accuracy **0.309748**, görülen signature altgrubunda **0.645952**. Bu yeni bir fit/tuning veya dördüncü yöntem önerisi değil; lookup ezberinin genelleme yetersizliği için deterministik aggregate probe. Tie-break en küçük canonical class index; validation etiketleri majority seçmekte kullanılmadı.

Kanıt: `performance_diagnosis/analyze.py` ve `signature_ambiguity.json`. Exact novelty, “yeni hastalık” veya dağılım kayması kanıtı değildir: bilinen concept'lerin yeni kombinasyonları da yenidir.

## 3. Aynı hastalardaki hatalar: karmaşık model LR'dan ne kadar farklı?

| GNN | Eşleştirilen LR | İkisi de doğru | İkisi de yanlış | Yalnız GNN doğru | Yalnız LR doğru | Δ macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| common/protgnn/current | lr/concepts/none | 3415 | 2817 | 357 | 859 | -0.031582 |
| common/protgnn/sqrt_inverse | lr/concepts/none | 3830 | 2843 | 331 | 444 | +0.001826 |
| common/graphcare/current | lr/concepts/none | 4029 | 2891 | 283 | 245 | -0.008024 |
| common/graphcare/sqrt_inverse | lr/concepts/none | 3934 | 2836 | 338 | 340 | +0.009165 |
| common/gsat/current | lr/concepts/none | 4009 | 2874 | 300 | 265 | -0.000307 |
| native/protgnn | lr/native_diagnostic/none | 3599 | 2386 | 409 | 1054 | -0.061797 |
| native/gsat | lr/native_diagnostic/none | 4221 | 2361 | 434 | 432 | +0.006359 |
| native/graphcare | lr/concepts/none | 4012 | 2877 | 297 | 262 | -0.007855 |

Native GSAT ile native LR birbirine çok yakın accuracy taşır; native ProtGNN, LR'ın doğru yaptığı çok daha fazla kişiyi kaçırır. Ortak GraphCare sqrt'ın LR'a F1 kazancı, neredeyse aynı doğru kişi sayısında hata dağılımını değiştirmesidir; genel accuracy kazanımı değil. Paired sayımlar bağımsız-hasta sırası/label doğrulamasıyla hesaplandı. Güven aralığı veya significance testi yapılmadı; validation üstünde epoch/loss seçimi iyimserdir, bunlar tekrarlı-seed kazanç kanıtı değildir.

Örtüşen coverage grupları (accuracy; sınıf karışımları farklı olduğundan nedensel etki gibi okunmaz):

| Altgrup | n | Prot sqrt | GraphCare sqrt | GSAT current | LR concepts |
|---|---:|---:|---:|---:|---:|
| seen_signature | 2816 | 0.6555 | 0.6445 | 0.6641 | 0.6623 |
| novel_signature | 4632 | 0.4998 | 0.5304 | 0.5266 | 0.5201 |
| no_cc | 890 | 0.2438 | 0.2640 | 0.2607 | 0.2438 |
| no_med | 2190 | 0.6658 | 0.6584 | 0.6767 | 0.6785 |
| hub_only | 59 | 0.1186 | 0.1356 | 0.0847 | 0.0847 |
| concepts_1_to_2 | 2799 | 0.6574 | 0.6499 | 0.6667 | 0.6624 |
| concepts_3_plus | 4590 | 0.5041 | 0.5327 | 0.5312 | 0.5261 |

Complaint bulunmaması güçlü bir hata işaretidir; “concept sayısı azsa kötü” genellemesi yanlıştır: 1–2 concept grubu daha yüksek accuracy taşır. Kimileri kolay tanınan şikâyetlerle gelirken çok-concept grup daha karmaşık olabilir; bu son cümle test edilmemiş klinik açıklamadır. No-CC sınıf kompozisyonu `metrics_and_errors.json/coverage/*/label_counts` içinde, bu rapor onu nedensel standardizasyon yapılmış gibi sunmaz.

## 4. Class imbalance: en güçlü mevcut kontrollü iyileşme kanıtı

Train en büyük/en küçük sınıf: **7.224 / 322 = 22.435**. Current ProtGNN `N/(30*n_c)` ağırlıklı CE kullanır; max/min örnek ağırlık oranı 22.435. Sqrt ile bu oran **4.737** olur. Current GraphCare/GSAT ağırlıksız CE. CE precision/recall dengesini doğrudan macro-F1 için optimize etmez.

| Sqrt − current, aynı 30-epoch protokol | Δ macro-F1 | Δ accuracy | Δ balanced acc |
|---|---:|---:|---:|
| protgnn | +0.033408 | +0.052229 | -0.010465 |
| graphcare | +0.017189 | -0.005371 | +0.042787 |

ProtGNN'in inverse ağırlığı bazı azınlık sınıflarını aşırı tahmin eder; GraphCare/GSAT CE bazılarını neredeyse hiç seçmez. Aynı “azınlıklar kötü” sonucu farklı mekanizmalardan geliyor:

| Common model / class | Val destek | Tahmin sayısı | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| common/protgnn/current / End stage renal disease | 43 | 312 | 0.0417 | 0.3023 | 0.0732 |
| common/protgnn/sqrt_inverse / End stage renal disease | 43 | 124 | 0.0726 | 0.2093 | 0.1078 |
| common/protgnn/current / Sepsis | 40 | 134 | 0.0821 | 0.2750 | 0.1264 |
| common/graphcare/current / End stage renal disease | 43 | 5 | 0.2000 | 0.0233 | 0.0417 |
| common/graphcare/sqrt_inverse / End stage renal disease | 43 | 39 | 0.1282 | 0.1163 | 0.1220 |
| common/graphcare/current / Sepsis | 40 | 3 | 0.3333 | 0.0250 | 0.0465 |
| common/graphcare/sqrt_inverse / Sepsis | 40 | 10 | 0.2000 | 0.0500 | 0.0800 |
| common/gsat/current / End stage renal disease | 43 | 6 | 0.0000 | 0.0000 | 0.0000 |
| common/gsat/current / Sepsis | 40 | 2 | 0.5000 | 0.0250 | 0.0476 |

Bütün 30 sınıf × 12 görünümün report'u aggregate JSON'dadır. GSAT sqrt yalnız önceki **3-epoch native** pilotta denenmişti: macro-F1 düşerken balanced accuracy arttı. Bunun sonucunu 30-epoch common GSAT loss karşılaştırması diye aktarmayın; böyle bir koşul yok. Loss değişimi için en kuvvetli eldeki kanıt ProtGNN/GraphCare eşleştirmesidir, seed genellenebilirliği hâlâ belirsiz.

Kanıt: `common_input_improvement.py:90–109`; `run_common_input.py:107–109,130–138`; özgün `protgnn_analysis/train.py:600–655`; `performance_diagnosis/metrics_and_errors.json`.

## 5. ProtGNN: başlık, prototype projection ve doğrulanmış küçük-graph kusuru

### 5.1 Gerçek aktif model, yorumlardan farklı

`protgnn_analysis/models/GCN.py:114–149`: üç normalize-adjacency GCN katmanı → ReLU + **p=0.5 GNN dropout** → mean pooling → sınıf başına üç prototype (toplam 90) ile squared-L2/log similarity → bias'sız linear last layer. `emb_normlize=False`. Prototype yolunda tanımlı MLP ve `dropout=0.51` classification kolu çalışmıyor. `config.py:76` üzerindeki “gnn_dropout unused” yorumu güncel forward için yanlıştır: sentetik hook tek forward'da GNN dropout'un **3** çağrısını, classifier dropout'un **0** çağrısını gördü (`synthetic_pyg.json`). Dolayısıyla 0.51'i ayarlamanın prototype classifier'a regularizasyon etkisi varmış gibi HPO gerekçesi kurulamaz.

Loss: ağırlıklı CE + 0.1 cluster + 0.1 margin-separation + 0.0005 cross-class L1 (`common_input_improvement.py:102–109`). Native trainer diversity hesaplıyor ama katsayısı **0.0** (`train.py:633–644`); sınıf-içi prototype ayrışmasını zorlayan aktif diversity yok. `joint_optimizer_lrs` / `last_layer_optimizer_lr` config alanları mevcut tek Adam'da kullanılmıyor (`train.py:524`, `run_common_input.py:109`); bu deneylerde optimizer grubuna özel LR uygulandığı iddia edilmemeli. Düşük performansa neden oldukları henüz ablation ile ölçülmedi.

### 5.2 Projection en iyi checkpoint'e ulaşmıyor

Warm-up epoch1–10'da last layer frozen; epoch11'de açılıyor. Projection zero-based20 / insan sayımıyla epoch21'in **başında**, o epoch'un optimizer adımlarından önce. Her sınıf için aynı bir kez karıştırılmış listedeki ilk **10 class-matched graph**, sınıfın üç prototype'ı için tekrar kullanılıyor (`run_common_input.py:33–54`; native `train.py:554–578`). “Her prototype farklı candidate kümesi görür” native yorumunun garanti ettiği şey gerçekleşmiyor; farklı olan mevcut prototype'a göre MCTS puanları olabilir.

| Loss | Best epoch / F1 | Projection öncesi epoch20 F1 | Projection + bir epoch eğitim sonrası epoch21 F1 | En iyi post-projection |
|---|---|---:|---:|---|
| current | 15 / 0.440559 | 0.428560 | 0.432855 | epoch30 / 0.439272 |
| sqrt_inverse | 11 / 0.473967 | 0.461529 | 0.466683 | epoch30 / 0.469846 |

İki koşulda 87/90 prototype değişmiş; en iyi checkpoint'ler projection **öncesi**. Bu nedenle “projection yüzünden best performansı düşük” iddiası mantıken yanlış. Epoch20→21 iki koşulda da artar, fakat bu projection + bir epoch eğitim toplamıdır; saf projection kazancı da değildir. Projection öncesi/sonrası anlık checkpoint çifti yok. Sonraki joint eğitim prototype'ları tekrar hareket ettirdiğinden last checkpoint'i exact eğitim-subgraph kopyası diye sunmak da doğru değildir.

Prototype vektörleri best→last sınıf içinde yaklaşmış (pairwise L2 median):
- current: 0.1836 → 0.0504; last sınıflar-arası median 0.9683.
- sqrt_inverse: 0.2343 → 0.0590; last sınıflar-arası median 0.9199.

Bunlar redundancy hipotezini destekleyen geometri gözlemleri; tam çöküş değil (L2<1e-4 sınıf-içi çift yok). Projection ve kalan training birlikte değişti, hangi aşamanın yakınlaştırdığı ayrıştırılamaz. Prototype activation/assignment çeşitliliği gerçek train/val graph'larında ölçülmedi.

### 5.3 Küçük graph MCTS skor hatası: gerçek ama etkisi sınırlı yorumlanmalı

`my_mcts.py:17–29` root P'yi 0 başlatıyor; `38–41` ≤min_atoms=3 düğümlü root'u puanlamadan döndürüyor. Root `99–109` içinde ayrıca skorlanmıyor; `121–129` embedding'i hesaplıyor ama returned score hâlâ sıfır. Caller yalnız similarity>0 olduğunda projection yapıyor. Sentetik 1/2/3-node graph'ların dönen similarity'si **0**, aynı coalition'ın gerçek `gnn_prot_score` değeri pozitif; 4-node kontrolü doğru pozitif döndü. Bu deterministik bir doğruluk kusurudur, hiperparametre tercihi değil.

Train **22,669/59,607 (%38.03)** graph bu ≤3-node rejiminde; hepsinin eğitimi bozuk değil, yalnız bu büyüklükteki graph projection candidate olarak pozitif skor alamaz. İki projection artifact'inde değişmeyen üç prototype aynı Laceration sınıfına ait. Bu sınıf train'inde 771/971 graph küçük; gerçek seçilmiş 10 candidate kimliği saklanmadığından “hepsi küçük olduğu için” bağlantısı olası, kesin değil. Best checkpoint zaten projection öncesi olduğundan bu kusur common best sıralamasını açıklamaz.

Ek temsil nüansı: MCTS “subgraph”ı node/edge silerek değil **zero-filling** ile kuruyor; bütün kenarlar ve global mean denominator kalıyor (`my_mcts.py:122–128,146–156`). Dolayısıyla prototype açıklaması literal induced-subgraph embedding'i değil. Bunu saf subgraph benzerliği olarak yorumlamak yanıltıcı olabilir; zero-filling seçimi tek başına prediction bug'ı değildir.

## 6. GraphCare: tek ziyaret ve mean-pooling uyarlamasının sınırı

`graphcare_analysis/run.py:57–77` gerçek modelde max_visit=1, joint EHR+graph, iki BAT katmanı, random trainable embeddings; LLM/KG expansion bu star koşulunda yok. Genel config `graph_structure=full` olsa da **kaydedilmiş koşul star**; default ile sonuç karıştırılmamalı.

`model.py:51–73`: alpha softmax **visit axis dim=1** üzerinden. Tek visit'te bütün alpha'lar tam 1 olur; beta `[batch,1,1]` olduğundan katman başına aynı graph'ın bütün source node/edge'lerine aynı scalar yayılır. Sentetik gerçek checkpoint probe'u alpha=1, alpha parametre gradient'i=0 ve graph-içi attention range=0 doğruladı. Bu upstream'in çok-ziyaret attention'ının tek-ziyaret girdiye doğal dejenerasyonu; dim=1'i yanlış diye değiştirmek model semantiğini değiştirir.

Alpha'nın **74,884/158,120 (%47.36)** tanımlı parametresi bu input rejiminde prediction-loss gradient'i almıyor. Weight decay sayısal değerlerini yine değiştirebilir; 'hiçbir optimizer update yok' iddiası değil. Toplam parametre sayısı efektif kapasiteyi abartır; azalan attention expressivity performans kusuru olarak ölçülmüş değildir.

Direct EHR kolu `model.py:97–108`: concept embedding'lerin **ortalaması** → affine `lin` → graph koluyla concat → tek linear MLP. Eval'da EHR katkısı normalized bag-of-concepts üzerinde lineer katkıdır; graph kolu nonlinear kalır. Sayıları/dosage'ı zaten binary giriş kaybetmiştir; mean ayrıca raw concept sayısını doğrudan bir feature olarak vermiyor. **“Mean tüm cardinality bilgisini yok eder” denemez:** sabit hub ve mesaj toplama graph kolunda node count etkisi taşıyabilir. Bu mimari analiz LR'a yakın sonucun makul açıklamasıdır; pooling ablation yapılmadan “mean pooling yanlış” kararı verilmez.

Katmanlar arasında **hardcoded dropout=0.5** (`model.py:83`), graph/EHR/post-concat aşamalarında configured 0.3 (`94,101,106`) var. Config'teki tek 0.3 tüm regularizasyonu anlatmaz. Aşırı regularizasyon/underfit hipotezi açık; forward veya loss bozukluğu gösterilmedi.

Hub-only raw EHR mean clamp ile tam sıfır ve logits finite; iki sentetik graph birlikte/ayrı eval logit farkı <1e-4. Güncel kritik “empty graph NaN” veya evaluation batch-mixing sorunu bu problarda yeniden üretilemedi. GraphCare'in azınlık recall sorununun loss ağırlığıyla kısmen düzelmesi, tüm başarısızlığı pooling'e bağlamaya karşı doğrudan karşı-kanıttır.

Kanıt: `performance_diagnosis/synthetic_graphcare.json`; `external/GraphCare/graphcare_/model.py:95–171` parametre yapısı, `29–83` BAT mesajları; first-party `graphcare_analysis/model.py:25–121` gerçek forward.

## 7. GSAT ve learning curves: bottleneck var, baskın zarar kanıtı yok

GSAT aynı 3-layer GIN'i iki kere kullanır: extractor embedding + gated predictor. Node binary-concrete gates, edge gate=`a_u*a_v`; coefficient1, temperature1, r=0.9/0.8/0.7 (`gsat_analysis/models/gsat.py:83–135`, `config.py:39–62`). Mean readout resmi add-pool tercihinden belgelenmiş sapmadır (`models/gin.py:9–11`).

İki önemli mekanizma ayrımı:
- `_info_loss` training'de sigmoid olasılığına değil **sampled binary-concrete attention'a** uygulanıyor (`gsat.py:112,126`); docstring Bern(p) ifadesi hesaplanan sayıyı tam anlatmıyor. Sentetik p=r=0.7'de probability KL ≈0 iken aynı logit'ten 20.000 sample ortalama KL yaklaşık **0.1921**. Bu örneklenmiş bottleneck maliyeti, “extractor bozuk / sabit attention” kanıtı değildir; resmi implementasyonla farklılık gösterilmeden bug diye etiketlenmedi.
- Zero edge gate, node deletion değildir: GIN self residual `(1+eps)*x` korunur (`gin.py:35–45`) ve node'lar mean pooling'de kalır. Sentetik identity-MLP conv probe'u sıfır edge gate'te self feature'ın aynen kaldığını doğruladı. Böylece “r=0.7 graph bilgisinin %30'u tamamen silindi” yorumu yanlış. Star hub gate'i bütün spokes'u çarpar; bunun prediction maliyeti mevcut kayıtlarla izole ölçülemez.

Common GSAT r geçişlerinde (1-based): epoch10→11 train objective 1.6063→1.6925, F1 0.470686→0.456486; epoch20→21 objective 1.6219→1.6901, F1 **0.459060→0.467352**. İkinci geçişte F1 artıyor ve best epoch25/final-r rejiminde. Dolayısıyla curriculum'un her düşüşü performansı bozduğu veya başlıca sınırlayıcı olduğu iddiası desteklenmiyor.

Native star/1234 log'unda (`comparison/standardized/training.log:464–495`) pred CE 2.297→1.183; info 0.128→0.187, validation best zero-based20'de 0.5589, son zero-based30'da 0.5521. r düşerken info maliyeti büyür, fakat prediction CE çok daha büyüktür; “KL loss dominate ediyor” kanıtı yok. Bazı native GSAT koşulları final-r başlamadan durur (aynı log `562,592,661`); config'teki “post-r-decay early-stop” yorumu gerçek guard değildir (`train.py:203–211`). Ortak 30-epoch koşul early stopping kullanmayarak bütün curriculum'u tamamlamıştır.

Extractor graph-aware InstanceNorm kullanıyor; sentetik aynı graph tek başına/başka graph yanında eval farkı <1e-4. Backbone train-time BatchNorm ve dropout farklıdır; eval bağımsızlığı testi train regularizasyonunun önemsiz olduğunu göstermez.

### Eğriler neyi gösteriyor, neyi göstermiyor?

| Common model | Epoch10 train objective | Epoch30 train objective | Epoch10 validation CE | Epoch30 validation CE | Best→last F1 kaybı |
|---|---:|---:|---:|---:|---:|
| protgnn/current | 2.6713 | 1.9907 | 2.1158 | 2.1277 | 0.001287 |
| protgnn/sqrt_inverse | 2.5900 | 1.8979 | 1.8097 | 1.8300 | 0.004121 |
| graphcare/current | 1.5968 | 1.5449 | 1.5682 | 1.5435 | 0.004259 |
| graphcare/sqrt_inverse | 1.7655 | 1.7122 | 1.6328 | 1.6080 | 0.003465 |
| gsat/current | 1.6063 | 1.6496 | 1.5387 | 1.5435 | 0.008427 |

Validation CE burada bütün 150 kaydedilmiş prediction'dan yeniden hesaplanan **ağırlıksız** CE'dir. Train objective dropout/noise açık minibatch training-mode ortalamasıdır; Prot'ta weighted CE+aux, GSAT'ta CE+KL, GraphCare sqrt'ta weighted CE. Train objective ile validation CE'yi doğrudan çıkarıp “generalization gap” diye adlandırmak geçersizdir. Weighted minibatch CE'nin sample-count ile ortalaması da tek global ağırlıklı CE'ye birebir eşit değildir.

- GraphCare current epoch10→30 validation CE azalır; F1 plato yakınında. Büyük overfit çöküşü yok; basitçe daha uzun training'in ciddi sıçrama sağlayacağı da gösterilmedi.
- ProtGNN training objective belirgin azalırken validation CE/F1 düzleşir. Regularizasyon/aux-head uyumsuzluğu, az miktar overfit ve input sınırı ayrıştırılamaz. Native star1234 best train-mode acc **0.534451**, val **0.538131**; last train **0.558022**, val **0.537863** (`results/protgnn/star/seed_1234/training_metrics.csv`). Devasa train–val farkı yok; dropout açık train ölçümü gerçek eval-mode kapasiteyi aşağı çekebilir.
- GSAT değişen r ile objective düzeyleri doğrudan trend karşılaştırması değildir. Son dönem validation plato/noise; prediction/attention dağılımının training cohort'ta checkpoint-bazlı aggregate ölçümü yok.
- Bütün yöntemlerde common history train accuracy/macro-F1 ve ayrı aux bileşenlerini saklamıyor (`run_common_input.py:130–151`). Bu yüzden kesin “underfit” veya “overfit” sınıflandırması yapılamaz. Yeni full train inference bu salt-artifact/synthetic kapsamda yapılmadı.

Kanıt: `performance_diagnosis/learning_curve_diagnosis.json`, native altı ProtGNN eğrisi de burada özetlenmiştir. Epoch seçimleri korunmuş best checkpoint'lere aittir, test üzerinden yeniden seçim yok.

## 8. Hedef ve klinik veri hazırlama: yanlış problemi açıklamayın

1. **Hedef yalnız “rastgele ilk tanı” değil, seçilmiş kategori zincirinin çıktısı.** ICD9→10 crosswalk ilk preferred eşleme; başlık comma sonrası trim; 3-character category; kategori en sık canonical title ile adlandırılır (`merge_ed.py:120–193,208–225`). Özel görünen bir sınıf adı daha geniş 3-character grubu temsil edebilir; clinician-reviewed phenotype tanımıyla aynı olduğu kanıtlanmadı.
2. Explicit merges T1/T2 diabetes, pneumonia/COPD, injuries ve chronic cardiovascular risk'i gruplar (`227–299`). Bunlar heterojenliği ve bazı kolay karışıklıkları azaltır, fakat label granülaritesi eşit değildir. Aynı complaint birden çok hedefle uyumlu olabilir; mevcut confusion bunu destekler, otomatik “etiket yanlış” demek değildir. Skin/soft-tissue infection→limb injury/pain common GSAT'ta 123, GraphCare current'ta 114, sqrt'ta 86 validation hata; bunların klinik olarak yanlış etiketlendiği ayrıca incelenmedi.
3. **Kritik düzeltme:** `_topn_titles` seq_num sırasına göre disease_1/2/3 üretir (`363–378`), ancak **`549–555` disease_2 dolu kişileri filtreleyip disease_2/3 sütunlarını siler**. Bu snapshot'ta explicit multi-label hastaları alıp tek tanıyı rastgele seçmişiz gibi teşhis koymak yanlış. Single-disease filtresi top30/merge SONRASINDA olduğundan “gerçek hastanın hiçbir başka hastalığı yok” da kanıtlanmaz. İkinci/üçüncü tanı prevalansı snapshot'tan ölçülemedi; sütun yokluğu read-only probe ile doğrulandı.
4. Cohort selection missing acuity/vitals, UNKNOWN/OTHER transport, hedef yokluğu, tek retained disease ve top300-med match filtrelerini içerir (`512–555,624–629`). Common-input'ta vital sayıları kaldırılmış olsa da vital-completeness eligibility seçimi kalır. Bu, genel ED popülasyonuna geçerliliği kısıtlar; doğruluk düşüşünün tek yönlü nedeni değil, seçilimi kolaylaştırabilir de.
5. Complaint text canonicalization/prevalence≥0.002/ilk5 ve med top300 seçimi bütün upstream dağılımından gelir (`615–670`). Sonraki train≥1% med filtresi 300'den 104 ilaç bırakır; önceki audit canonical med üyeliklerinin %78.88'ini koruduğunu, 2.038 kişiyi retained-med'siz bıraktığını ölçmüştü (`performance_review_20260913/input_audit.json`). Daha nadir ilaçları geri koymak otomatik iyileşme garantisi değil; variance artabilir. Complaint laterality stripping ve text→binary kaybının hasta seviyesindeki etkisi burada ölçülmedi.
6. Stay/timestamp lineage `497,522`'de silinir; `687–688` yalnız subject'a göre sort edip ilk satırı seçer, **earliest visit garantisi yok**. Ziyaretleri chronological olarak ilkmiş gibi adlandırmayın. Özgün subject-aware split kimlikler arası train/test leakage'ini önler; zamansal availability sorununu çözmez.
7. Native `n_ed_visits` bütün ziyaret sayısı (`29–32`); labs bütün [intime,outtime] penceresi (`extract_ed_labs.py:5–13`), vital-trends stay-wide (`merge_ed.py:74–103`). Common set bunları kullanmıyor ama medrecon stay-wide (`390–400`); önceki raw zaman audit'i medrecon'un çoğunun intake sonrası yazıldığını buldu (`docs/common-input-improvement.md:20–30`). Raw audit tüm medication satırlarını kapsar, canonical selected-stay zamanı kanıtı değildir. **temporal_clean=false**; medication-history-available source-snapshot dışına klinik kullanım iddiası taşınmamalı.

Bu filtrelerin/timing sorunlarının varlığı doğrudan; performansı kaç puan azalttığı/şişirdiği bilinmiyor. Ham veri yeniden hazırlanmadan “sızıntı temizlendi” iddiası yok.

## 9. En güçlü sonraki deney — uygulama yapılmadı

**Önce mevcut aile içinde eşit-input, tek-faktör predictive-vs-bottleneck ablation'ı.** Dördüncü yöntem değil. Aynı frozen 192-concept star + constant hub, aynı canonical fold/seed listesi/batch/update/selection; mevcut GSAT GIN backbone'da attention gate'i ve bilgi cezasını birlikte standart unmasked predictor kontrolü olarak kaldıran önceden tanımlı kontrol ile tam GSAT'ı karşılaştırın. İkinci, ayrı koşul yalnız beta=0 olup stochastic gate kalacaksa o farklı hipotezi test eder; ikisini aynı ablation diye karıştırmayın. Tam aileyi “yeni yöntem” olarak paketlemeyin.

Gerekçe: ortak GSAT LR ile eşit düzeyde; bu kontrol mesaj-geçiren backbone'un predictive kapasitesini stochastic/explanation maliyetinden ayırır. Her seçilmiş checkpoint için **aynı eval mode train/validation CE, macro-F1, per-class recall, seen/novel-signature** ve aux bileşenlerini aggregate kaydedin. Çok-seed dağılımı olmadan farkı kalıcı saymayın. Test yalnız son seçim dondurulduktan sonra açılmalı. Bu rapor eğitim yetkisi vermez.

ProtGNN için daha ucuz ve bilimsel olarak zorunlu sonraki kontrol, ≤3-node root scoring regression ve sadece bu düzeltmenin **aynı pre-projection checkpoint / aynı train candidate draw / aynı optimizer-RNG** üzerinden immediate-pre/post projection ve sonraki sabit-budget validation etkisidir. Şu anda hata doğrulandı; performans kazancı iddia edilmedi. Best checkpoint projection öncesi olduğu için bu, birinci sıradaki genel performans açıklamasının yerine geçmez.

Klinik gerçeklik sorusu için ayrı uzun vadeli öncelik: seçilmiş-stay lineage + açık karar cutoff'u + train-only upstream filtrelerle aynı kohortu muhafaza eden yeniden hazırlama ve uygun tabular bilgiyi üç yönteme eşitleme. Mevcut native LR kazanımı bunun büyüklüğünü önceden garanti etmez.

## 10. Yeniden üretim, doğrulama ve sınırlar

Yeni dosyalar yalnız `docs/model-performance-diagnosis.md` ve `comparison/standardized/performance_diagnosis/` altında. Kaynak model/config/cache/results değiştirilmedi. Hesaplama script'leri bu izole kökte tekrar çalıştırılabilir; yalnız diagnostic aggregate dosyalarını yazar.

```bash
# Repo kökünden; fit/training/test forward yok.
PYTHONDONTWRITEBYTECODE=1 python3 comparison/standardized/performance_diagnosis/analyze.py
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3 comparison/standardized/performance_diagnosis/synthetic_probes.py pyg
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .venv-graphcare/bin/python3 comparison/standardized/performance_diagnosis/synthetic_probes.py graphcare
PYTHONDONTWRITEBYTECODE=1 python3 comparison/standardized/performance_diagnosis/target_aggregate.py
python3 comparison/standardized/performance_diagnosis/render_report.py
```

- `analysis_verification.json`: 150 common epoch F1 bağımsız sklearn ile ve seçilmiş metrikler confusion-count NumPy ile doğrulandı; 5 common + 3 native + 4 LR karşılaştırması, canonical label/fold hizalaması. Hasta feature/prediction satırları yeni çıktılara yazılmadı.
- `signature_ambiguity.json`: sadece train signature-label grupları + validation coverage/lookup aggregate; hiçbir test prediction veya test signature analizi yok.
- `metrics_and_errors.json`: 30 sınıflı reports, confusion özetleri, paired error ve coverage label dağılımları.
- `learning_curve_diagnosis.json`: 150 epoch'un objective/val CE/F1'ı, altı native ProtGNN curve özeti, projection geçişleri.
- `synthetic_pyg.json`, `synthetic_graphcare.json`: yeniden üretilmiş root-score kusuru, aktif dropout, gate ve alpha dejenerasyonu; gerçek hasta performansı diye sunulmuyor.
- `target_and_small_graphs.json`: canonical sıra doğrulaması, train küçük-graph coverage; secondary target sütunları olmadığı için multimorbidity prevalence ölçümü yok. İlk secondary-column probe bu yoklukla durdu; fallback tahmin yerine kısıt kaydedildi.
- `preservation_check.json` ve `report_verification.json`: koruma hash'leri / report tablolarının aggregate ile sayısal tutarlılık ve kimlik-token taraması için son doğrulama kayıtları.
- Ana ortam Pyright import uyarısı runtime engeli değildi; gerçek Python/Torch probe'ları çalıştı. Ana PyG ortamında mevcut LibreSSL warning görüldü. GraphCare kendi eski Torch ortamında gerçek forward/backward ile sınandı.
- Gizlilik sınırı: bir keşif komutunun schema çıktısı yanlışlıkla canonical patient-map anahtarlarını araç izine bastı. Değerler rapora/aggregate dosyalara aktarılmadı ve tekrarlanmadı; bu nedenle **“oturumun hiçbir çıktısında kimlik yok” denemez**. Araç izi ayrıca erişim/gizlilik incelemesi gerektirir; bu rapor dağıtıma uygun hasta-satırı içermeyen aggregate çıktı sunar, önceki araç izini temizlediğini iddia etmez.

**Son sınır:** mekanizma doğruluk kusuru ile kötü benchmark skorunun nedeni farklı iddialardır. Yeni hasta-level GNN activation/train-eval replay, controlled no-bottleneck/no-prototype/readout eğitimleri, cutoff-safe rebuild ve çok-seed belirsizlik ölçülmeden baskın nedeni tek bileşene indirgeyemeyiz.
