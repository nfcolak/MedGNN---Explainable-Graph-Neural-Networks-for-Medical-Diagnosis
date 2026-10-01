# clinical_graph_v2 — karar-anı klinik graf

v1 event-graph teşhisinin (`docs/new-input-diagnosis.md`) ardından sıfırdan kuruldu.

## Mantık düzeltmesi — eski sonuçlarla karıştırmayın

Yeni üretici sözleşmesi `clinical_graph_logic_v2`'dir. Vital değerler, birim-ayrık
sayısal kodlama ve demografik alanlar artık model girdisinin açık parçalarıdır;
ön işleme ve metrik sürümleri koşu bağlamına kaydedilir. Eski checkpoint'ler yeni
ön işleme ile sessizce yüklenemez. Eski graflar yalnız metadata güncellenerek yeni
sözleşmeye yükseltilemez: geçmiş tanılar yeni üreticiyle ayrı bir klasöre kurulmalıdır.

**Mevcut `clinical_graph_v3_full`, `clinical_graph_v3_max6` ve eski koşu sonuçları
tarihsel artefaktlardır.** Aşağıdaki eski sayılar düzeltilmiş akışın performansı
değildir; burada yeni eğitim sonucu sunulmuyor. Eski komutlardaki dolu çıktı
klasörlerinin üzerine yazmayın; yeniden üretim için yeni yollar seçin.

Hasta-eşit değerlendirme, ziyaret tanılarını ve tahminlerini koruyup her ziyarete
`1 / hastanın değerlendirilen ziyaret sayısı` ağırlığı verir. Majority tanı ve
ortalama tahmin artık bu metriğin tanımı değildir. Macro-F1 tanımlı tüm sınıfları
kullanır. Yeni tahmin dosyaları ziyaret `sample_ids` alanını taşır; aynı hastanın
birden çok ziyareti bulunan eski dosyalarda yalnız hasta sırasıyla alt-grup hizası
kanıtlanamaz ve analiz reddedilir.

`--no-edge-payload` yalnız sayısal kenar alanlarını siler, ilişki türünü korur.
Rewiring, tip/ilişki başına düğüm giriş-çıkış derecelerini korur; **sayısal-yüksüz
gerçek-kenar kontrolüne** karşı kullanılmalıdır. `--rewire-relation` bu nedenle
`--no-edge-payload` gerektirir. Takas yapılamayan graflar değişmeden kalabilir.
Tek bir skor farkı, grafın gereksizliğini veya nedensel faydasını kanıtlamaz.

> CEI-GNN v3 study results (screen/validation/GraphXAI, aggregate only, with provenance): [`docs/cei-v3-delivery-2026-10-01.md`](../../../docs/cei-v3-delivery-2026-10-01.md).

## v3 yöntem adaptörleri — uygulama sözleşmesi

Bu dal, aynı `clinical_inputs_v3` tensor girdisi üzerinde dört yöntem seçeneği
tanımlar. **Bu, gerçek veri üzerinde doğrulanmış bir benchmark sonucu değildir.**
Graf/etiket üretimi, gerçek eğitim ve test-fold değerlendirmesi ayrı onay gerektirir;
yeni koşu bağları bu nedenle `test_evaluated=false` yazar.

| `--method` | korunan ana mekanizma | klinik grafa uyarlama | yerel varsayılan bütçe |
|---|---|---|---|
| `clinical_gnn` | mevcut EdgeConditioned/HGT/GCHM-PNA yolu | davranışı korunur; `--conv` yalnız bu yönteme aittir | hidden 96, 3 katman, dropout 0,1, AdamW, 12 epoch |
| `protgnn` | sınıf prototipleri, cluster/separation kayıpları, warm-up ve yalnız train-loader projection | düğüm türü, ilişki/meta-ilişki kimliği ve sayısal edge payload kullanan encoder; upstream checkpoint uyumlu değildir | hidden 128, 3 katman, dropout 0,51, Adam, 300 epoch |
| `gsat` | ortak GIN predictor, node-level stochastic attention, tek edge-mask müdahalesi ve Bernoulli KL bilgi darboğazı | her mesaj ilişki/meta-ilişki ve sayısal edge payload taşır; eval sigmoid attention ile deterministiktir | hidden 128, 3 katman, dropout 0,3, Adam, 100 epoch |
| `graphcare` | BAT-style mesaj, visit-conditioned alpha/beta attention ve joint readout | vocabulary-square attention yerine hidden-width sparse visit projection; empty visit sıfır state/0 beta mass; global düğümler ayrı yol; doğrudan sayısal/context kanalı korunur | hidden 128, 2 katman, dropout 0,3, Adam, 100 epoch |

GraphCare global düğümleri BAT mesajlarından, normal graph readout'tan ve doğrudan
klinik kanaldan çıkarır; bu düğümler yalnız ayrı global-context projeksiyonunda
tüketilir. Doğrudan kanal global olmayan ve knowledge olmayan klinik düğümlerle
sınırlıdır.

Her adaptörün `binding.json` kaydı toplam/etkin parametre sayısını, native
varsayılanları, kullanıcı override'larını, optimizer/bütçeyi, objective ve schedule
ayarlarını, adaptasyon sürümünü ve recursive kaynak kod hash'lerini taşır. Aynı epoch
sayısı veya benzer parametre sayısı eşit compute ya da yöntem eşdeğerliği kanıtlamaz.

### Visit membership sözleşmesi

Her graf satırıyla bire bir hizalı `visit_membership.jsonl` sidecar'ı zorunludur.
Sözleşme sürümü `clinical_visit_membership_v1`'dir ve satır yalnız şunları taşır:

- mevcut anonim `sample_id`;
- kaynaktan doğrulanmış, eskiden yeniye sıralı `visit_ordinals`;
- `[visit_ordinal, graph_node_index]` biçiminde sparse `membership_pairs`;
- düğüm başına açık `global_node_mask`.

Visit üyeliği düğüm sırasından veya zaman bucket'ından tahmin edilmez. Bir düğüm
birden çok ziyarete bağlı olabilir. Üyeliği olmayan düğüm otomatik olarak global
sayılmaz; yalnız açık global mask buna izin verir. Kaynak soyundaki boş ziyaretler
ordinal satırını korur. PyG batch'inde visit ve node indeksleri ayrı offsetlenir ve
visit'in sahibi `num_visits` üzerinden bulunur.

Sidecar hash'i/satır sayısı, graf hash'i, target binding, class order,
train/validation sample-ID hash'leri ve `clinical_inputs_v3` preprocessing hash'i
uyuşmadan consumer çalışmaz. Eski sidecar'sız graflar metadata düzenlenerek
yükseltilemez; ileride yeniden üretim onaylanırsa yeni ve boş bir artifact yolu gerekir.

### CLI uyumluluğu ve ortak değerlendirme disiplini

- `--method` verilmezse mevcut `clinical_gnn` davranışı ve `edge_conditioned`
  varsayılanı korunur.
- `--conv` yalnız `--method clinical_gnn` ile geçerlidir. ProtGNN, GSAT veya
  GraphCare ile açık `--conv` verilmesi artifact okunmadan ve output oluşturulmadan
  reddedilir.
- Ortak override verilmezse her adaptör yukarıdaki native profilini kullanır;
  verilen değer ve native karşılığı birlikte kaydedilir.
- Method-native override'lar ad alanlıdır: `--protgnn-*` warm-up/projection/MCTS ve
  prototype objective ayarlarını, `--gsat-*` temperature/information-bottleneck
  curriculum ayarlarını, `--graphcare-*` recency/message-dropout ayarlarını taşır.
  Başka yöntemle verilen native seçenek artifact okunmadan reddedilir.
- Tüm yöntemler aynı patient-disjoint fold'ları, train-only preprocessing'i,
  train class weight'lerini, validation macro-F1 checkpoint seçimini, visit-level
  tahminleri ve patient-equal metriği kullanır. Test fold loader'ı oluşturulmaz.
- Adaptörler yalnız forward mekanizmasını, auxiliary objective'i, epoch hook'unu ve
  optimizer gruplarını sahiplenir. ClinicalGNN AdamW/norm-5 clipping'i korur;
  ProtGNN Adam/value-2 clipping, GSAT ve GraphCare ise Adam/no-clipping kullanır.

Bu milestone'un izinli kanıtı yalnız in-memory sentetik wiring testleridir. Bunların
geçmesi gerçek hasta artefaktı uyumluluğunu, temporal availability'yi, benchmark
performansını veya yöntem üstünlüğünü göstermez. Aşağıdaki eski üretim/eğitim
bölümleri tarihsel ClinicalGNN çalışma notlarıdır; yeni adaptör milestone'u için
çalıştırma talimatı değildir.

## Neden yeniden kuruldu

Eski v1'in kendi koşullarındaki (yeni mantık sözleşmesinden önceki) ölçümü iki sorun göstermişti:

1. **Graf yapısı hiçbir şey katmıyordu.** Tam girdi 0,1995 macro-F1, sadece lab değer
   istatistikleri 0,1901 → grafın tüm katkısı **+0,009**. `structure_only` 0,0843 ile
   çoğunluk tabanının (0,1099) altındaydı. Sebep: `contains_event`, `observes_concept`
   birer üyelik kaydı, `next_event_time` ise zaten kendi zaman damgasını taşıyan
   düğümleri sıralıyordu. Hepsi özellik vektöründen yeniden kurulabilirdi.
2. **En güçlü kanal atılmıştı.** Chief complaint tek başına 0,4329 macro-F1 veriyordu
   (eski girdide); v1 zaman damgasız triyajı dışladığı için bu blok yoktu.

## Ne değişti

### 1. Karar anı kanıtı geri geldi — aynı kesim kuralıyla

| kaynak | v1 | v2 | gerekçe |
|---|---|---|---|
| chief complaint | yok | **var, zaman varsayımıyla** | bağımsız kayıt zamanı yok; `intime` atanması gerçek kullanılabilirliği kanıtlamaz |
| triyaj vitalleri | yok | **var, zaman varsayımıyla** | aynı sınır; aralık dışı değerler sayılıp düşürülür, kırpılmaz |
| varış demografisi | yok | **var** | yaş/cinsiyet/ırk/geliş şekli açık özellikler; `anchor_age` kesin ziyaret yaşı değildir |
| **geçmiş tanılar** | yok | **opsiyonel** (`--with-diagnosis`) | yalnız **tamamlanmış önceki** ziyaretlerden; indeks ziyaretin tanısı etikettir, asla girmez |
| `disposition` | — | **reddedilir** | ziyaretin sonucu, karar anında bilinmez |
| indeks ziyaret tanısı | — | **reddedilir** | etiketin ta kendisi |

Kesim (cutoff) v1'den **bit-bit devralınır**; bu üretici kendi kesimini hesaplayamaz.

#### Geçmiş tanı katmanı (`--with-diagnosis`)

`diagnosis.csv`'de **zaman damgası yok** — yalnız `stay_id` ve `seq_num`. Bu yüzden
uygunluk *yapısal* olarak kurulur: tanı, indeks ziyaret başlamadan **önce bitmiş** bir
ziyarete aitse alınır. Ancak kapanış zamanı tanının gerçek kayıt/kullanılabilirlik
zamanını kanıtlamaz. Yeni sözleşme bu varsayımı açıkça kaydeder ve
`temporal_clean=false` taşır; bu bayrak tek başına gerçek gelecek sızıntısı bulgusu değildir.

ICD-9 ve ICD-10 birlikte var (%49/%51) ve hastaların **%38,6'sında** indeks ile geçmiş
ziyaret farklı sürümle kodlanmış. Ham kod karşılaştırmak tekrarları sessizce kaçırır,
bu yüzden repodaki GEM tablosuyla ICD-10'a normalize edilir:

| | tekrar yakalama |
|---|---:|
| ham kod | %43,9 |
| **GEM eşlemesi ile** | **%48,7** |

GEM, görülen ICD-9 kodlarının %99,2'sini kapsıyor. `approximate=1` eşlemeler düğümde
bayraklanır, eşlenemeyen kodlar kendi ICD-9 ad alanında kalır.

### 2. Kenarlar iki sınıfa ayrıldı

`STRUCTURAL` kenarlar grafı gezilebilir yapar ve **hiçbir sonuçta gerekçe gösterilemez**.
`INFORMATIVE` bir tasarım etiketidir, ek bilgi veya öngörü kazancı kanıtı değildir.
Yeniden kurulabilirlik için hem yönlü uçların çokluğu hem sayısal yük sınanır:

| ilişki | yük | bilgi sınırı |
|---|---|---|
| `baseline_of` | delta, geçen süre, hız | "kreatinin 11 günde 1,8 arttı" — karşılaştırma ortağı her hastada farklı |
| `trajectory_of` | aynı | ardışık ölçüm zinciri |
| `co_complaint` | — | şikâyet üyeliklerinden kurulabilir; ek ham olgu olduğu varsayılmaz |
| `recurrence_of` | kaç ziyarette, en son ne zaman | "KOAH, 3 yatış, sonuncusu 40 gün önce" bir geçmiş bayrağı değildir |
| `comorbid_with` | — | aynı geçmiş ziyarette birlikte kodlanan tanılar birbirini kısıtlar |
| `medical:*` | provenance | hastalar arası paylaşılan dış bilgi |

### 3. Boyut 18,4 GB → ~3,5 GB

v1 her geçmiş olayı döküyordu. v2 yalnız **index vizitte de geçen** analitlerin
geçmişini alır — eşleşmeyen geçmiş değer zaten bir özellik kolonudur, delta tanımlamaz.

## Ölçülen sonuç (3.000 graf, gerçek koşu)

```
düğüm/graf 54,6   kenar/graf 80,7   informative/graf 26,1
```

**Geri çekilen eski yeniden-kurulabilirlik yorumu**, 3.000 graf:

Aşağıdaki tarihsel tablo `informative` etiketini ek bilgi kanıtı sayan eski
denetimden gelmiştir. Sayılar yalnız geçmiş kayıt olarak korunur; tam yeniden
kurma/payload denetiminin sonucu olarak **kullanılamaz**. Yeni denetim ham düğüm
görünümünü, dış bilgi gereksinimini ve bilinmeyen yük alanlarını ayrı raporlar;
bu görünüm XGBoost'un kayıplı özet vektörüyle aynı şey değildir.

| | v1 | v2 (tanısız) | **v2 + `--with-diagnosis`** |
|---|---|---|---|
| tablo görünümünden tam kurulabilen | %100 | 1.033/3.000 (%34) | **712/3.000 (%24)** |
| ek olgu taşıyan graf | 0 | 1.967, ort. 39,9 olgu | **2.288, ort. 69,8 olgu** |
| aynı coarse görünüm → farklı payload | 0 | 263 | **297** |
| informative kenar | 0 | 78.430 | **168.358** |

Bu tabloya dayanarak tanı katmanının ek bilgi miktarı veya graf modelinin üstünlüğü
çıkarılamaz. Örneğin `recurrence_of` son görülme zamanı, uç kimlikleri kurulabilse
bile düğüm görünümünden çıkmayabilir.

**Sızıntı doğrulaması** (tanılı koşu):

```
index_diagnosis_leaks      : 0
diagnosis_nodes_reverified : 20350
```

20.350 tanı düğümünün **hepsi** kaynaktan yeniden türetilip indeks ziyaretin kendi
kodlarıyla karşılaştırıldı. Koruma negatif kontrolle de sınandı — indeks ziyaret
bilerek geçmiş listesine enjekte edildiğinde üretici reddediyor:

```
SALDIRI 1 (index stay prior listesinde): REDDEDILDI
SALDIRI 2 (sadece index stay)          : REDDEDILDI
KONTROL  (meşru çağrı)                 : GEÇTI, sızıntı=0
```

Delta dağılımı da düz değil (ör. `lab:51265` trombosit: n=423, pstdev 65,1).

## Sızıntı denetimi

265 farklı düğüm token'ı tarandı:
- tanı/geçmiş namespace token'ı: **0**
- etiket-adı çakışması: **2** — `cc:head injury`, `cc:hypotension`

Bu ikisi triyajda hastanın **kendi beyan ettiği şikâyet**, tanı kopyası değil.
v1'in eski girdisindeki 33 `hx_*` çakışmasından farklı: orada özellikler etiketle
aynı kod kümesinden türetilmişti. Yine de bu iki sınıfın metrikleri ayrı raporlanmalı.

## Kullanım

### Güncel çalışma kohortu: en fazla 6 ziyaret

Çoklu-ziyaret analizinde bir kişinin ham `edstays.csv` içindeki farklı `stay_id`
sayısı **6'dan büyükse kişi bütünüyle çıkarılır**; hiçbir kişinin yalnızca ilk 6 ziyareti
bırakılmaz. Mevcut filtrelenmiş artefaktlar:

- `comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_max6/`
- `comparison/standardized/event_inputs/clinical_graph_v3_max6/`
- `comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_local_v2_max6/`

Filtre ölçütü `n_ed_visits` ile aynıdır: distinct raw `stay_id` sayısı `<= 6`.
Bu türevde eligible örnek sayısı 60.273 kişi / 185.888 ziyaretten 51.994 kişi /
100.418 ziyarete indi (eligible kişi farkı 8.279). Onarımın ham canonical
ziyaret sayımında 6'dan fazla stay'i olduğu için 8.309 canonical kişi filtrelenir;
bu sayı, eligible kişi farkından farklıdır. Kaynak all-visits veri dosyaları
korunmuştur; manifest/index metadata'sındaki hardlink sorunu için
`repair_metadata` bağımsız yedek ve atomik değiştirme kullanır. Metadata onarımı
klinik grafı yeniden üretmez ve eski hedeflerin provenance kaydını değiştirmez.

Sıfırdan üretimde aynı kuralı Step 1 komutuna ekleyin:

```bash
--max-visits-per-subject 6
```

Sıfırdan çalıştırma **4 adımdır**. Bu repo yalnız kodu taşır; üretilen artefaktlar
(`event_inputs/`, ~7 GB) MIMIC türevi olduğu için gitignore'dadır ve aşağıdaki
adımlarla yeniden üretilir.

Aşağıdaki 4 adımlı komutlar kaynak **all-visits** artefaktını yeniden üretir.
Güncel `<=6` kohortu için yukarıdaki `_max6` yollarını kullanın ve Adım 1'e
`--max-visits-per-subject 6` ekleyin.

**Ön koşul:** `data/Original CSVs/` altında MIMIC-IV-ED dosyaları
(`labevents.csv`, `edstays.csv`, `triage.csv`, `patients.csv`, `diagnosis.csv`,
`icd9_to_icd10_mapping.csv`) ve `comparison/canonical_split.json`.

### Adım 1 — Olay index'i + kohort (yalnız sıfırdan üretimde)

Bu adım 18 GB `labevents.csv` tarar. Yukarıda belirtilen mevcut full/max6 index'leri
zaten onarıldı; yalnız `index_ready` durumundadır ve **bu komutu mevcut dizinlerin
üzerine çalıştırmayın**. Sıfırdan yeni bir index üretecekseniz yeni bir output yolu
seçin. Aşağıda, çalışmadaki max6 kohortu için iki seçenekten biri kullanılmalıdır:
mevcut onarılmış index'i `GRAPH_ROOT` olarak geçirmek (yeniden tarama yapmaz) veya
sıfırdan üretirken max6 politikasını aynı üreticiye uygulamak.

```bash
# Sıfırdan üretim gerekiyorsa; mevcut dizinlerin üzerine yazmayın:
python3 -m comparison.standardized.event_graph_v1.first_lab \
  --raw-root "data/Original CSVs" \
  --canonical comparison/canonical_split.json \
  --knowledge comparison/standardized/event_graph_v1/knowledge_seed.csv \
  --output comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_logic_max6 \
  --max-visits-per-subject 6 \
  --lab-scope all-numeric-known-unit --execute
```

Kesilirse `--resume-prepared` ile devam eder (tarama tekrarlanmaz).
Disk tasarrufu için bitince `graphs.jsonl` (17 GB) silinebilir; v3 onu kullanmaz.

### Adım 2 — Yeni cohort'a etiket sidecar'ı (~10 dk)

Graflar etiketsizdir; hedefler `sample_id` ile ayrı bağlanır. Onarılmış index için
mutlaka açık `--graph-root` ve daha önce kullanılmamış output klasörü verin; eski
sidecar'lar manifest hash'i değiştirilerek sessizce yenilenmez. Aşağıdaki komut
mevcut, onarılmış max6 index'ini kullanır:

```bash
python3 -m comparison.standardized.event_graph_gchm_xgb_v1.local_labels_v2 \
  --graph-root comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_max6 \
  --output comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_logic_v2_max6
```

### Adım 3 — Yeni mantık sözleşmeli klinik graf (~10 dk)

Adım 1'in `events.sqlite` + `cohort.csv`'sini devralır, 18 GB taramayı tekrarlamaz.

```bash
python3 -m comparison.standardized.clinical_graph_v2.build \
  --inherit-from comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_max6 \
  --raw-root "data/Original CSVs" \
  --canonical comparison/canonical_split.json \
  --knowledge comparison/standardized/event_graph_v1/knowledge_seed.csv \
  --output comparison/standardized/event_inputs/clinical_graph_logic_v2_max6 \
  --complaint-min-count 50 --with-diagnosis --execute
```

`--with-diagnosis` olmadan tanı katmanı üretilmez. Karşılaştırma gerekiyorsa diğer
kolu da farklı, boş bir output yoluna üretin. Eski `clinical_graph_v3_full` ve
`clinical_graph_v3_max6` çıktıları tarihsel kalır; bu yeni sözleşmeye metadata ile
yükseltilemez.
`--limit N` sınırlı smoke içindir ve artefaktı `bounded_smoke_not_benchmark` işaretler.

#### HGT arm'ı hakkında (`--conv hgt`)

`EdgeConditionedLayer` ilişki one-hot'ını paylaşılan mesaj MLP'sine verir.
Önceki `baseline_relation_is_additive_bias` yorumu **geçersizdi**: yalnız ilk
doğrusal katman ölçülmüştü. Gerçek `Linear → ReLU → Linear` mesajında ilişki,
ReLU kapılarını değiştirerek içeriğe bağlı etki yaratabilir. Yeni mekanizma
kontrolü tam mesajı sınar; bu doğruluk kontrolü performans farkının nedenini kanıtlamaz.

`--conv hgt` bunun yerine kenarı `⟨kaynak tipi, ilişki, hedef tipi⟩` üçlüsüne
ayrıştırır. Bu grafta **15 meta-ilişki** var (ölçüldü, 3.000 graf; hiçbir ilişki
birden fazla tip imzası taşımıyor, yani ayrıştırma belirsizlik üretmiyor).

BG-HGNN'in (2024) iki uyarısı koda gömülü:

| uyarı | burada ne yapıldı | nasıl doğrulanıyor |
|---|---|---|
| parametre patlaması | tipli ağırlıklar **head boyutunda** (`d/H`) çalışır, tam genişlikte değil; yalnız K ve V tipli, Q/çıkış/güncelleme paylaşımlı | Eski mekanizma kontrolündeki `hgt_parameters_not_exploding`: 111.360 vs naive ilişki-başına-MLP 927.744 (paylaşımlının 1,92×'i); yeni girdinin tam koşu kapasite eşitliğini kanıtlamaz |
| ilişki çöküşü | tipli projeksiyonların birbirinden ayrışması ölçülür, varsayılmaz | `relation_separation()` her koşuda `result.json`'a init ve son değerle yazılır; `aggregate.py` çökmeyi rapor eder |

**Kapasite uyarısı (tarihsel sayılar):** eski girdide `--conv hgt --hidden 96`
426.270, eski diğer kollar 266.142, `--hidden 128` baseline 422.014 parametreydi.
Yeni girdi sözleşmesi özellik boyutunu değiştirdiğinden bu sayılar yeni kolların
kapasite eşitliğini göstermez. Her yeni koşuda binding'deki toplam ve etkin
parametre sayılarını okuyup `aggregate.py` kohort/arm raporlarını kullanın;
aynı parametre sayısı da eşit eğitim maliyeti kanıtı değildir.

Eski HGT ekleme çalışmasının bit-eşitlik ölçümü, bu mantık düzeltmesinden **önceki**
koda aittir. Yeni girdi ve ablasyon semantiği eski koşularla bit-eşitlik iddia etmez.
NOMP toplam kayıtlı parametre sayısı, etkin parametre sayısına eşit değildir;
mesajlaşma atlanırken kullanılmayan katmanlar etkin kapasiteye sayılmaz.

### Adım 4 — Eğitim ve kontroller

```bash
# mekanizma kontrolleri (egitimsiz, saniyeler) -- once bunu kosun
python3 -m comparison.standardized.clinical_graph_v2.mechanism_check

# ana kosu (~10 dk/seed, CPU)
python3 -m comparison.standardized.clinical_graph_v2.train \
  --artifact comparison/standardized/event_inputs/clinical_graph_logic_v2_max6 \
  --targets comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_logic_v2_max6/targets.csv \
  --output comparison/standardized/clinical_runs_logic_v2_max6/main_seed1234 \
  --epochs 12 --seed 1234 --edges all --execute

# ayni sinifin kontrolleri (NOMP etkin kapasite-esit degildir)
#   --no-message-passing   mesaj gecirme yok (dugum-only kontrol)
#   --no-edge-payload      delta/aralik yuku sifirlanir
#   --edges informative    yalniz informative iliskiler

# HGT arm'i (WWW'20) -- tipli dikkat
python3 -m comparison.standardized.clinical_graph_v2.train \
  --artifact comparison/standardized/event_inputs/clinical_graph_logic_v2_max6 \
  --targets comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_logic_v2_max6/targets.csv \
  --output comparison/standardized/clinical_runs_logic_v2_max6/hgt_seed1234 \
  --conv hgt --hidden 96 --heads 4 --epochs 12 --seed 1234 --execute

# HGT ile KAPASITE-ESIT baseline (426.270'e karsi 422.014 param, %1 fark)
python3 -m comparison.standardized.clinical_graph_v2.train \
  --artifact comparison/standardized/event_inputs/clinical_graph_logic_v2_max6 \
  --targets comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_logic_v2_max6/targets.csv \
  --output comparison/standardized/clinical_runs_logic_v2_max6/wide_seed1234 \
  --hidden 128 --epochs 12 --seed 1234 --execute

# tablo kontrolu (XGBoost, ayni artefakt)
python3 -m comparison.standardized.clinical_graph_v2.tabular_control \
  --artifact comparison/standardized/event_inputs/clinical_graph_logic_v2_max6 \
  --targets comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_logic_v2_max6/targets.csv \
  --out comparison/standardized/clinical_runs_logic_v2_max6/xgb_control

# sonuclari topla + seed yayilimini olc
python3 comparison/standardized/clinical_graph_v2/aggregate.py
```

### İsteğe bağlı — graf denetimi

Grafın tablo görünümünden yeniden kurulup kurulamadığını ölçer:

```bash
python3 -m comparison.standardized.clinical_graph_v2.audit \
  --artifact comparison/standardized/event_inputs/clinical_graph_logic_v2_max6
```

Çıktı klasörü varsa üretim **başlamaz** (no-overwrite); yeniden denemek için
yeni bir yol verin.

Status: implemented but PROPOSED, not accepted (ADR-007 bidirectional edges/GCHM-PNA v2, ADR-008 dev-selected protocol). Results from these flags are exploratory until the ADRs are accepted.

### GCHM-PNA v2 ve eşit bütçeli protokol

Ölçülen sorun: üretici ziyaret→kanıt kenarlarını tek yönde yazıyor; ileri yönde
şikâyet/vital/ölçüm bilgisi indeks ziyaret hub'ına **hiçbir derinlikte** ulaşmıyor.
Ayrıca düğümlerin %71'i en fazla bir mesaj alıyor, bu yüzden v1'in 12 PNA bloğu
çoğunlukla boş genişlik (557k param, en büyük arm, erken aşırı öğrenme).

- `--edge-direction bidirectional`: geri dönüşü üretilmeyen 9 ilişkiye tipli ters kenar
  (`rev:*`) ekler; ileri blok ve düğüm tensörleri bayt-eşit kalır. Varsayılan `forward`
  tarihsel kodlamadır. Tüm arm'lar ve XGBoost aynı bayrağı kullanabilir.
- `--conv gchm_v2` (`gchm_v2.py`): hub durumu + ilişki tipiyle çarpımsal kapı,
  kompakt PNA (4d→d, öğrenilen ölçekleyiciler), hub-okuma. Her parçanın ablasyon
  anahtarı var: `--modulation additive`, `--aggregation sum`, `--no-hub-gate`,
  `--readout pool`. Varsayılan genişlik 92 → ProtGNN'in 399.884 parametresinin altında.
- `--selection-fold dev --dev-limit N`: epoch/tur seçimi, örneklemin hastalarından ayrık,
  kullanılmayan TRAIN hastalarından çekilen dev kümesinde yapılır; validation koşu başına
  **bir kez** okunur (`--final-eval none` ile hiç okunmaz). Test fold'u açılmaz.

Eşit bütçeli protokol (varsayılan kuru koşu; yazmak için `--execute` gerekir):

```bash
python3 -m comparison.standardized.gchm_v2_protocol.protocol                     # plan + ETA
python3 -m comparison.standardized.gchm_v2_protocol.protocol --stage pilot --execute
python3 -m comparison.standardized.gchm_v2_protocol.protocol --execute           # tune→report
python3 -m comparison.standardized.gchm_v2_protocol.protocol --status
```

Her arm'a aynı 6 deneme bütçesi, dev'de seçim, 3 seed final, v2 ablasyonları ve
değişmemiş v1 referansı. Önceden kilitli kazanma kuralı: en yüksek ortalama **ve**
ikinciye karşı hasta-bootstrap %95 aralığı sıfırın üstünde. Aksi durumda sonuç olduğu
gibi raporlanır. Tasarım: `docs/superpowers/specs/2026-09-24-gchm-pna-v2-design.md`.

Status: experimental; no ADR exists.

### GCHM-PNA v3 (`--conv gchm_v3`, `gchm_v3.py`)

v2'nin encoder'ı ve hub-kapılı PNA katmanları olduğu gibi kullanılır (import edilir,
kopyalanmaz). Değişen yalnız okuma tarafı:

| Parça | Ne yapar | Kapatma |
|---|---|---|
| wide yol | her düğüm token'ı sınıflara doğrudan doğrusal oy verir (`W_tok[t] + v·W_val[t]`, sıfırdan başlar) | `--no-wide` |
| label-wise okuma | her tanının kendi sorgusu düğümler üzerinde dikkat yapar (CAML tarzı), havuz başına eklenir | `--v3-readout pool` |
| jumping knowledge | encoder + her katmanın durumu birleştirilir | `--no-jk` |
| kenar dropout | yalnız eğitimde rastgele kenar düşürür (varsayılan 0.1) | `--edge-dropout 0` |

Varsayılan genişlik 88 → 391.448 parametre, ProtGNN'in 399.884'ünün altında
(`mechanism_check` → `v3_parameters_below_protgnn`). Altı v3 mekanizma kontrolü eğitimsiz çalışır.

Tasarım kanıtı (validation/test **açılmadan**): 10k protokol train örneği ile eğitim,
skor 5.000 kullanılmamış TRAIN ziyaretinde; hastaları hem train örneğinden hem protokolün
dev kümesinden ayrık. Bidirectional kenarlar, 30 epoch, en iyi epoch macro-F1:

| Arm | Param | Seed | Tasarım macro-F1 |
|---|---|---|---|
| GCHM-PNA v3 (h88) | 391k | 3 | 0.6440 / 0.6465 / 0.6459 |
| GCHM-PNA v2 | 380k | 1 | 0.6325 |
| ProtGNN | 404k | 1 | 0.6393 |
| GraphCare | 125k | 1 | 0.6301 |
| GSAT | 204k | 1 | 0.6143 |
| XGBoost (token bag, tur seçimi tasarımda) | — | 1 | 0.6462 |

Bu tablo protokol sonucu değildir: tek bölme, rakipler tek seed. Üstünlük iddiası için v3,
eşit bütçeli protokolde (dev seçimi, 3 seed, bootstrap) koşulmalıdır.

## Garantiler

- 18 GB `labevents.csv` taraması **tekrarlanmaz**; v1 index'i sha256 ile doğrulanıp okunur.
- Her düğüm kesimle sınırlıdır; `time_hours > 0` veya `available_hours > 0` hata verir.
- **İndeks ziyaretin tanısı asla grafa giremez.** `prior_history()` indeks `stay_id`'yi
  reddeder (çağıran yanlış liste verse bile), doğrulama her tanı düğümünü kaynaktan
  yeniden türetir ve indeks-özel kodlarla kesişim arar. Sıfır değilse üretim başarısız.
- Yazımdan sonra tam readback: ölçüm/şikâyet çoklu-kümeleri kaynaktan yeniden türetilir,
  delta'lar uç noktalardan yeniden hesaplanır, bildirilmemiş ilişki reddedilir.
- Artefakt hiç informative kenar içermiyorsa üretim **başarısız sayılır**.
- Şikâyet sözlüğü yalnız **train fold**'da fit edilir; sözlük dışı formlar sayılıp
  düşürülür, `other` düğümüne toplanmaz.

## Sınırlar

- Şikâyet sözlüğü ham yüzey formlarının frekans allowlist'idir; `abd pain` ve
  `abdominal pain` ayrı token'dır. Eşanlamlı birleştirme klinik bir yargıdır, yapılmadı.
- Triyaj vitallerinin bağımsız zaman damgası yoktur; `intime`'a atfedilir.
- **Tanıların hiç zaman damgası yoktur.** Uygunluk yapısaldır (önceki ziyaret indeks
  ziyaretten önce bitmiş), kaydedilmiş bir tanı zamanı değil. Aynı ziyaret içinde
  hangi tanının önce konduğu bilinemez; `seq_num` kodlama sırasıdır, zaman değil.
- ICD-9 kodları GEM ile ICD-10'a eşlenir; `approximate` eşlemeler düğümde bayraklanır,
  eşlenemeyenler ICD-9 ad alanında kalır. GEM klinik olarak gözden geçirilmemiştir.
- `storetime` veritabanı kullanılabilirlik vekilidir, klinisyen görünürlüğü kanıtı değil.
- Bilgi ilişkileri kaynak-doğrulanmıştır, klinik olarak gözden geçirilmemiştir.
- `anchor_age` demografik çapadır, bu ziyaretteki tam yaş değildir.
- Artefakt **etiketsizdir**; hedefler ayrı, hash'li sidecar ile `sample_id` üzerinden bağlanır.
- Bu artefakt yeni bir girdi sürümüdür; eski benchmark ile parite **iddia etmez**.
- Yeniden kurulabilirlik denetimi grafın bilgi taşıdığını gösterir; bunun **modele
  kazanç olarak yansıyacağını göstermez**. O ayrı bir deneydir.
