# İleri yöntemler: ikinci tarama ve yapısal ölçüm

Tarih: 2026-09-20. İlk taramanın (`message-passing-literature.md`) kapsamadığı
yöntem ailelerini taradım, sonra **hangisinin bizim grafa uyduğunu ölçtüm**.
Kod yazılmadı, deney yapılmadı.

## Önce: grafımız aslında ne? (4.000 graf, ölçüm)

Bu ölçüm yöntem seçimini belirliyor, çünkü ileri yöntemler yapıya göre ayrışıyor.

| ölçüm | değer | anlamı |
|---|---:|---|
| hub'a **doğrudan** bağlı düğüm oranı | **%61** | graf büyük ölçüde yıldız |
| yaprak (derece ≤ 1) düğüm oranı | %44 | yarısı hiç komşuya sahip değil |
| üçgen içeren graf | %75,9 (526.947 üçgen) | ama klik kaynaklı (aşağıda) |
| **en uzun trajectory zinciri** | **medyan 1, max 1** | **zincir yok** |
| trajectory taşıyan graf | %36,6 | çoğunda hiç yok |
| **aynı anda ölçülen lab (panel)** | **medyan 13, max 29** | güçlü yüksek-mertebe grup |
| aynı ziyarette birlikte tanı | medyan 4,3, max 25,9 | klik yapısı |

Üç sonuç, hepsi elemeli:

1. **Zincir yok.** `trajectory_of` zincirlerinin maksimum uzunluğu **1**. Yani
   ardışık ölçüm zinciri diye bir şey pratikte oluşmuyor — her geçmiş ziyarette
   genelde tek ölçüm var. → **SSM / Mamba / sequence yöntemleri boşa gider.**
2. **Yıldız baskın.** Düğümlerin %61'i doğrudan hub'a bağlı, %44'ü yaprak.
   Mesaj geçirmenin "komşudan komşuya bilgi taşıma" avantajı büyük ölçüde yok —
   zaten herkes hub'dan bir adım uzakta. → **Set-tabanlı yöntemler ciddi aday.**
3. **Panel yapısı güçlü ve şu an KAYIP.** Medyan 13 lab aynı anda ölçülüyor
   (metabolik panel, tam kan sayımı). Grafta bunlar birbirine bağlı değil, sadece
   visit'e bağlı. "Bu 13 test bir panel olarak beraber istendi" bilgisi hiçbir
   kenarda yok. → **Hipergraf gerçek bir fırsat.**

Üçgenlerin kaynağı `comorbid_with` klikleri ve `co_complaint` çiftleri — yani
gerçek çok-hop yapı değil, zaten klik olarak kodladığımız gruplar.

## Yöntem aileleri: uygun / uygun değil

### ✅ Hipergraf (AllSet, `2106.13264`)

Panel = hiperkenar. Medyan 13 elemanlı grup, şu an pairwise kenara indirgenmiş
bile değil. AllSet hiperkenar→düğüm ve düğüm→hiperkenar geçişini öğrenilebilir
iki multiset fonksiyonu olarak kuruyor; DeepSets ve Set Transformer'ı özel durum
olarak kapsıyor.

**Neden uygun:** "şu 13 test birlikte istendi" bir grup olgusu. Pairwise kenara
çevirmek 13·12/2 = 78 kenar üretir ve grup kimliğini kaybeder.

### ✅ Set-tabanlı okuma (Set Transformer / DeepSets, `2206.11925`)

Yıldız yapıda, ziyaret düğümünün işi komşularının **kümesini** özetlemek.
Bu tam olarak bir set fonksiyonu. Mesaj geçirme bunu dolaylı yapıyor; Set
Transformer doğrudan ve attention ile yapar.

`2206.11925` ayrıca kritik bir uyarı veriyor: derin permütasyon-değişmez ağlarda
gradyan patlaması/sönmesi olur, LayerNorm bilgi siler. **Biz zaten bu tuzağa
düştük** (sum-pooling logit patlaması, `pool_norm` ile çözüldü). Makale "set norm"
ve equivariant skip connection öneriyor — bizim düzeltmemizin doğru versiyonu.

### ✅ WaveGNN (`2412.10621`, 2024) — klinik düzensiz zaman serisi

Doğrudan bizim veri tipimiz: düzensiz örneklenmiş, farklı frekanslı, eksik
gözlemli klinik ölçümler. İnterpolasyon yapmadan **doğrudan düzensiz seri
üzerinde** çalışıyor, intra-series ve inter-series bağımlılıkları ayrı modelliyor.

**Neden uygun:** bizim `baseline_of` aralıklarımız 1,6 saat – 5,8 yıl arası.
Decay-aware yaklaşım tam bu dağılım için.

### ✅ Informative missingness (`2606.17106`, 2026)

*"Laboratuvar testlerinin yokluğu, ölçümün kendisi kadar bilgilendirici olabilir;
eksiklik klinisyen kararını yansıtır."*

Bizde **hangi testin istendiği** güçlü sinyal (v1 ablasyonunda `concept_present_only`
tek başına 0,1247 macro-F1 veriyordu, tabanın üstünde). Şu an bunu ayrı
modellemiyoruz, sadece düğüm varlığı olarak kodluyoruz.

### ❌ SSM / Graph-Mamba (`2509.13735`, `2408.08583`)

Zincir uzunluğu max 1. Dizi modeli için dizi yok.

### ❌ Topolojik derin öğrenme (`2505.15405`, `2409.12033`)

Simplicial/cellular complex, n-body etkileşim için. Bizim yüksek-mertebe yapımız
tek tip (panel/klik) ve hipergraf bunu zaten karşılıyor. TDL'in ek makinesi
karşılığını vermez.

### ❌ Subgraph GNN / ESAN (`2110.02910`, `2206.11168`)

1-WL sınırını aşmak için. Bizim sorunumuz ayırt edicilik değil — graflar zaten
farklı (audit: 35 coarse çakışmada farklı payload). Maliyet çok yüksek
(her graf için N alt-graf).

### ❌ Graph foundation model (`2409.14500` kendi bulgusu)

GraphLand makalesi mevcut GFM'leri kendi veri setlerinde test edip
*"fail to produce competitive results"* diyor. Olgunlaşmamış.

### ⚠️ LLM + graf (`2407.13989`, `2407.12860`)

Text-attributed graph için. Bizim token'larımız (`lab:50912`, `dx:icd10:A419`)
metin değil kod. LLM'in ICD/LOINC bilgisi bir **başlangıç gömmesi** olarak
kullanılabilir — özellikle seyrek kodlar için (74 kenarlı `medical:measures`).
Ama bu bir yan yol, ana çözüm değil.

## Güncellenmiş öncelik sırası

İlk rapordaki sıra kısmen değişti, çünkü yapısal ölçüm yıldız+panel yapısını ortaya çıkardı.

| # | yöntem | gerekçe | maliyet |
|---|---|---|---|
| 1 | **Panel hiperkenarı** (AllSet tarzı) | medyan 13'lük grup bilgisi şu an tamamen kayıp | orta |
| 2 | **Meta-ilişki mesaj fonksiyonu** (HGT) | `baseline_of` ≠ `instance_of`; ilk raporun 1. maddesi | orta |
| 3 | **Set Transformer okuma** | yıldız yapı = set problemi; sum/mean pooling zayıf | düşük |
| 4 | **Yapısal kodlama** (hub uzaklığı, derece, panel üyeliği) | ucuz, `nomp` bile faydalanır | çok düşük |
| 5 | Decay-aware zamansal ağırlık (WaveGNN/TeMP-TraG) | 5,8 yıllık aralıklar eşit ağırlıkta | düşük |
| 6 | Hibrit GBDT+GNN (BGNN) | literatürde beklenen cevap; **ayrı satır** olarak raporlanmalı | yüksek |

**En önemli değişiklik:** panel hiperkenarı 1. sıraya çıktı. Sebebi ölçüm —
medyan 13 eşzamanlı lab, ve bu bilgi grafta hiç yok. Mimariyi değiştirmeden önce
**girdiye eksik olan yapıyı eklemek** daha yüksek beklenen getiri.

## Dürüst uyarı

Yıldız ölçümü (%61 hub'a doğrudan bağlı, %44 yaprak) rahatsız edici bir olasılığa
işaret ediyor: bu veri **doğası gereği** büyük ölçüde bir küme problemi olabilir,
graf problemi değil. O durumda Set Transformer, GNN'i geçer ve XGBoost'un
kazanması da açıklanır.

Bu test edilebilir bir hipotez: 3. maddeyi (Set Transformer okuma) uygulamak
aynı anda hem bir iyileştirme hem bir teşhis. Eğer set okuma mesaj geçirmeyi
geçerse, cevap "daha iyi GNN" değil, "GNN gerekmiyor" olur — ve bu da
raporlanabilir bir sonuçtur.

## Kaynaklar (ikinci tur)

| id | başlık | not |
|---|---|---|
| `2106.13264` | AllSet: Multiset Function Framework for Hypergraph NN | panel hiperkenarı |
| `2206.11925` | Set Norm and Equivariant Skip Connections | derin set ağlarında gradyan |
| `2412.10621` | WaveGNN: decay-aware irregular clinical time-series | bizim veri tipimiz |
| `2606.17106` | Informative Missingness in clinical time series | test istememe = sinyal |
| `2107.14293` | Self-Supervised Transformer for Sparse Clinical TS | ön-eğitim yolu |
| `2505.15405` | HOPSE: higher-order positional/structural encoder | TDL, elendi |
| `2409.12033` | TDL with State-Space Models (Mamba) | elendi |
| `2110.02910` | ESAN: Equivariant Subgraph Aggregation | elendi |
| `2509.13735` | State Space Models over Directed Graphs | elendi (zincir yok) |
| `2407.12860` | STAGE: LLM embeddings for TAG | yan yol |
