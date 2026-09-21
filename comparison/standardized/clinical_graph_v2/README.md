# clinical_graph_v2 — karar-anı klinik graf

v1 event-graph teşhisinin (`docs/new-input-diagnosis.md`) ardından sıfırdan kuruldu.

## Neden yeniden kuruldu

v1'in ölçülen sorunu ikiydi:

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
| chief complaint | yok | **var** | triyaj `intime`'da kayıtlı; 185.888/185.888 ziyarette `intime ≤ cutoff` doğrulandı |
| triyaj vitalleri | yok | **var** | aynı gerekçe; aralık dışı değerler sayılıp düşürülür, kırpılmaz |
| varış demografisi | yok | **var** | yaş/cinsiyet/ırk/geliş şekli varışta bilinir |
| **geçmiş tanılar** | yok | **opsiyonel** (`--with-diagnosis`) | yalnız **tamamlanmış önceki** ziyaretlerden; indeks ziyaretin tanısı etikettir, asla girmez |
| `disposition` | — | **reddedilir** | ziyaretin sonucu, karar anında bilinmez |
| indeks ziyaret tanısı | — | **reddedilir** | etiketin ta kendisi |

Kesim (cutoff) v1'den **bit-bit devralınır**; bu üretici kendi kesimini hesaplayamaz.

#### Geçmiş tanı katmanı (`--with-diagnosis`)

`diagnosis.csv`'de **zaman damgası yok** — yalnız `stay_id` ve `seq_num`. Bu yüzden
uygunluk *yapısal* olarak kurulur: tanı, indeks ziyaret başlamadan **önce bitmiş** bir
ziyarete aitse alınır. Tanı zaten ziyaret kapandıktan sonra kodlanır, dolayısıyla
tamamlanmış ziyaret doğru birimdir.

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
`INFORMATIVE` kenarların yükü iki uç noktaya birden bakmadan hesaplanamaz:

| ilişki | yük | neden tabloya sığmaz |
|---|---|---|
| `baseline_of` | delta, geçen süre, hız | "kreatinin 11 günde 1,8 arttı" — karşılaştırma ortağı her hastada farklı |
| `trajectory_of` | aynı | ardışık ölçüm zinciri |
| `co_complaint` | — | "göğüs ağrısı + nefes darlığı" ikisinin toplamı değil |
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

**Yeniden kurulabilirlik denetimi** (v1'i mahkûm eden test), 3.000 graf:

| | v1 | v2 (tanısız) | **v2 + `--with-diagnosis`** |
|---|---|---|---|
| tablo görünümünden tam kurulabilen | %100 | 1.033/3.000 (%34) | **712/3.000 (%24)** |
| ek olgu taşıyan graf | 0 | 1.967, ort. 39,9 olgu | **2.288, ort. 69,8 olgu** |
| aynı coarse görünüm → farklı payload | 0 | 263 | **297** |
| informative kenar | 0 | 78.430 | **168.358** |

Tanı katmanı grafı zenginleştiriyor: tablodan kurulabilen graf oranı %34'ten %24'e
düşüyor, graf başına ek olgu 39,9'dan 69,8'e çıkıyor. Yani eklenen şey bir presence
kolonu değil.

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

```bash
python3 -m comparison.standardized.clinical_graph_v2.build \
  --inherit-from comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2 \
  --raw-root "data/Original CSVs" \
  --canonical comparison/canonical_split.json \
  --knowledge comparison/standardized/event_graph_v1/knowledge_seed.csv \
  --output comparison/standardized/event_inputs/clinical_graph_v2_full \
  --complaint-min-count 50 --with-diagnosis --execute

python3 -m comparison.standardized.clinical_graph_v2.audit \
  --artifact comparison/standardized/event_inputs/clinical_graph_v2_full
```

`--with-diagnosis` olmadan tanı katmanı hiç üretilmez; ikisini ayrı çıktı klasörlerine
üretip karşılaştırmak, tanının katkısını ölçmenin doğru yoludur.

`--limit N` sınırlı smoke içindir ve artefaktı `bounded_smoke_not_benchmark` olarak
işaretler. Çıktı klasörü varsa üretim **başlamaz** (no-overwrite).

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
