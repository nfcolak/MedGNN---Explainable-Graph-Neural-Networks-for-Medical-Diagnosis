# Message passing yöntemleri: literatür taraması ve bizim graf için karar

Tarih: 2026-09-20. Kaynak: arXiv API + web araması. **Kod yazılmadı, deney yapılmadı.**

Sorun: v3 grafında mesaj geçirme kazanç üretmiyor (`main` − `nomp` = −0,0071 = 2,1× seed SS,
ters yönde) ve XGBoost hepsini geçiyor (+0,0139 = 4,7× SS). Mevcut tasarımın bilinen
kusuru: **15 ilişki tipinin hepsi tek paylaşılan MLP'den geçiyor** — `baseline_of`
(delta taşıyan) ile `instance_of` (sadece üyelik) aynı dönüşümü görüyor.

## Bizim grafın ölçülen özellikleri (4.000 graf)

| özellik | değer | neden önemli |
|---|---:|---|
| düğüm tipi | 8 | HGT/RGCN seçimini belirler |
| ilişki tipi | 15 | parametre patlaması eşiği |
| **meta-ilişki (kaynak,ilişki,hedef)** | **15** | HGT'nin parametrelediği birim |
| düğüm/graf medyan | 37 | küçük graf → global attention ucuz |
| düğüm/graf max | 822 | tam attention hâlâ mümkün |
| **graf çapı (hub'dan hop)** | **medyan 3, max 4** | **over-squashing riski düşük** |
| in-degree medyan / max | 1 / 91 | hub ağır, yapraklar seyrek |
| visit-hub derecesi | ort. 7,6 / max 91 | tek darboğaz noktası |

Meta-ilişki dağılımı çok dengesiz: en sık `instance_of` 142.931, en seyrek
`medical:measures` 74. **1.900× fark.**

## Literatürden çıkan dört bulgu

### 1. HGT: meta-ilişki bazlı parametre paylaşımı (WWW 2020, `2003.01332`)

Kenarı `⟨kaynak_tip, ilişki, hedef_tip⟩` üçlüsüne ayrıştırıp ağırlıkları bu üç
bileşene dağıtıyor. RGCN'in "her ilişkiye ayrı matris" yaklaşımına göre **daha az
parametreyle** daha iyi sonuç veriyor. Makalenin kendi ablasyonu (`HGT_noHeter`,
tüm meta-ilişkiler aynı ağırlığı paylaşıyor) **bizim şu anki modelimizle aynı** —
ve HGT ondan %9-21 daha iyi.

PyG'de hazır: `torch_geometric.nn.HGTConv`.

Makaleden doğrudan alıntı: *"For relations that don't have sufficient occurrences,
it's hard to learn accurate relation-specific weights."* — bizim `medical:measures`
(74 kenar) tam bu durumda.

### 2. BG-HGNN: ilişki sayısı arttıkça RGCN/HGT çöküyor (`2403.08207`, 2024)

İki kusuru adlandırıyor:
- **parametre patlaması**: ilişki başına ayrı ağırlık seti
- **ilişki çöküşü** (relation collapse): model ilişkileri ayırt etme yetisini kaybediyor

Çözümü: ilişki heterojenliğini paylaşılan düşük boyutlu bir uzaya damıtmak.
Parametre verimliliğinde 28,96×, throughput'ta 110,30× kazanç bildiriyor.

**Bizim için ölçülen maliyet:**

| yaklaşım | mesaj MLP ağırlığı |
|---|---:|
| paylaşılan tek MLP (şimdiki) | ~27.648 |
| ilişki başına ayrı (RGCN) | ~414.720 (**15×**) |
| HGT tarzı ayrıştırılmış | ~285.696 (10,3×) |

15 ilişki RGCN için sınırda; BG-HGNN'in uyardığı bölge.

### 3. TRANS: EHR'ye özel zamansal heterojen graf transformer (IJCAI 2024, `2405.03943`)

Bizim veri yapımızla **birebir aynı**: ziyaret düğümleri + tıbbi olay düğümleri,
olaylardan ziyarete bilgi akışı, ziyaretler arası zamansal kenarlar.

Üç bileşen ekliyor:
- zamansal kenar özellikleri (bizde `interval_hours`, `last_seen_hours` zaten var)
- **global positional encoding**
- **local structural encoding**

Son ikisi bizde **yok**. Modelin hangi düğümün hub, hangisinin yaprak olduğunu
bilmesinin tek yolu mesaj geçirme — oysa bu bilgi doğrudan verilebilir.

### 4. Over-squashing bizim sorunumuz DEĞİL (`2302.02941`, ICML 2023)

Teorik sonuç: over-squashing yüksek commute-time'lı düğümler arasında oluşur;
derinlik yardımcı olmaz, topoloji belirleyicidir.

**Bizim grafın çapı medyan 3, max 4.** 3 katmanlı model zaten tüm grafı görüyor.
Yani graph rewiring, sanal düğüm, uzun menzilli teknikler **bizim problemimizi
çözmez**. Bu bir eleme sonucu ve boşa harcanacak haftaları önlüyor.

### 5. GBDT'nin kazanması normal ve literatürde belgeli

- **GraphLand** (`2409.14500`, 2024): 14 endüstriyel veri seti, *"GBDTs provided with
  additional graph-based input features can sometimes be very strong baselines"*.
- **BGNN** (`2101.08543`, ICLR 2021): GBDT heterojen özellikleri, GNN topolojiyi
  işliyor; uçtan uca birlikte eğitiliyor. Yeni ağaçlar GNN'in gradyan
  güncellemelerine uyuyor.
- **EBBS** (`2110.13413`): boosting + graf yayılımı, yakınsama garantili.

Yani bizim XGBoost sonucumuz bir başarısızlık değil, **beklenen durum**. Literatürdeki
cevap "GNN'i düzelt" değil, **"ikisini birleştir"**.

## Karar: ne uygulanmalı

Etki/maliyet sırasına göre. Hepsi mevcut `ClinicalGNN` üzerinde constructor
bayrağı olarak kurulabilir, böylece kapasite eşitliği korunur.

### Öncelik 1 — Meta-ilişki bazlı mesaj fonksiyonu (HGT tarzı)

Şu anki tek MLP yerine, mesaj ağırlığını `⟨kaynak_tip, ilişki, hedef_tip⟩` üzerinden
ayrıştır. 15 meta-ilişki için tam RGCN (15× parametre) yerine HGT'nin paylaşım
şeması: kaynak-tip projeksiyonu + ilişki matrisi + hedef-tip projeksiyonu.

Gerekçe: bu tam olarak mevcut tasarımın en bariz iç çelişkisini kapatıyor
(`baseline_of` ≠ `instance_of`). HGT'nin `noHeter` ablasyonu bizim modelimiz ve
%9-21 fark bildiriliyor.

Risk: seyrek meta-ilişkiler (74 kenarlı `medical:measures`) aşırı öğrenebilir.
Azaltma: BG-HGNN'in uyarısı gereği paylaşımlı parametreleme, ilişki başına tam matris değil.

### Öncelik 2 — Yapısal + pozisyonel kodlama (TRANS / GraphGPS)

Düğüme ekle: hub'a uzaklık, derece, düğüm tipi içindeki sıra, ziyaret içi zaman sırası.
Bu bilgiler şu an yalnız mesaj geçirmeyle dolaylı öğrenilebiliyor.

Ucuz: sadece düğüm özelliği eklemek, mimari değişikliği yok. Mevcut `nomp` arm'ı
bile bundan faydalanır — ki bu iyi bir kontrol.

### Öncelik 3 — Hibrit: XGBoost + GNN (BGNN tarzı)

**Dikkat:** skill kuralı gereği, GNN'e XGBoost'un *tahminlerini* vermek hibrit üretir,
rakip değil. BGNN bunu uçtan uca ortak eğitimle yapıyor — ayrı bir yöntem olarak
raporlanmalı, "GNN XGBoost'u geçti" diye değil.

Doğru kurulum: GBDT ham tablo özelliklerini işler, GNN topolojiyi; ortak optimize edilir.
Sonuç tablosunda kendi satırı olur.

### Öncelik 4 — Zamansal ağırlıklı agregasyon (TeMP-TraG, `2503.16901`)

Yakın zamanlı kenarlara daha çok ağırlık. Bizde `baseline_of` aralıkları 1,6 saat ile
50.503 saat (5,8 yıl) arasında değişiyor — 5 yıl önceki bazal ile dünkü aynı ağırlıkta.
Makale 4 GNN'de ortalama %6,19 iyileşme bildiriyor.

### Uygulanmayacak

- **Graph rewiring / sanal düğüm / uzun menzil**: graf çapı 3-4, over-squashing yok.
- **Tam graph transformer (global attention)**: hub zaten 3 hop'ta her yere ulaşıyor;
  ek maliyet karşılığı yok. GraphGPS'in yerel+global ayrımından sadece yerel kısım gerekli.
- **Meta-path tabanlı yöntemler (HAN vb.)**: elle meta-path tasarımı gerektiriyor,
  HGT bunu otomatik öğreniyor.

## Doğrulama planı (deney açıldığında)

Her adım mevcut protokole uymalı:
1. Mekanizma testleri eğitimsiz geçmeli (mesaj ayrıştırılamazlığı, kapasite eşitliği).
2. Arm'lar **aynı sınıfta constructor bayrağı** olmalı, parametre sayısı raporlanmalı.
3. En az 3 seed; delta, ölçülen seed SS'inin (şu an 0,0033) katı olarak yazılmalı.
4. `nomp` kontrolü her turda tekrar koşmalı — mesaj geçirmenin kazanç ürettiği
   ancak bu arm'ı geçtiğinde söylenebilir.
5. XGBoost kontrolü aynı artefakt üzerinde kalmalı.
6. Test fold kapalı.

## Kaynaklar

| id | başlık | katkı |
|---|---|---|
| `2003.01332` | Heterogeneous Graph Transformer (WWW'20) | meta-ilişki parametrelemesi |
| `2403.08207` | BG-HGNN (2024) | parametre patlaması + ilişki çöküşü |
| `2405.03943` | TRANS (IJCAI'24) | EHR zamansal heterojen graf |
| `2404.14815` | THAM (2024) | zaman-farkında EHR transformer |
| `2205.12454` | GraphGPS | yerel/global kodlama ayrımı |
| `2302.02941` | Over-squashing (ICML'23) | derinlik yardımcı olmaz, topoloji belirleyici |
| `2409.14500` | GraphLand (2024) | GBDT güçlü baseline |
| `2101.08543` | BGNN (ICLR'21) | GBDT+GNN ortak eğitim |
| `2503.16901` | TeMP-TraG (2025) | zamansal ağırlıklı mesaj |
| `2110.13413` | EBBS | yakınsak boosted smoothing |
