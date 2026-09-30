# Yeni event-graph girdisi neden kötü sonuç veriyor?

Tarih: 2026-09-20. Kapsam: **sadece validation**, test fold açılmadı.
Tüm sayılar bu belgede pinlenmiş iki artefakttan, tek bir ortak öğrenici (XGBoost)
ile üretildi. Mevcut hiçbir koşu klasörü değiştirilmedi.

| | eski (native) | yeni (event-graph) |
|---|---|---|
| artefakt | `native_inputs/protgsat_snapshot_v1` | `event_inputs/..._features_v1` |
| fingerprint | `2a2566e6…6dcf9f7c0` (contract) | `b5988ae7…30fdf36b4` (features.npz) |
| örnek | 74.511 hasta (tek ziyaret) | 97.618 ziyaret örneği |
| kolon | 324 | 3.483 |
| train/val | 59.607 / 7.448 | 77.930 / 9.582 |
| train-majority tabanı | 0,1212 | 0,1099 |

Ortak bütçe: XGBoost `hist`, depth 4, lr 0,05, 400 tur, `sqrt_inverse` sınıf ağırlığı
(tam train sayımlarından, yalnız eğitim satırlarına), seed 1234. Bu bütçe, dondurulmuş
`matched_gchm_xgb_v1` protokolünden (1200 tur, lr 0,03) **farklıdır** ve o tablolarla
karıştırılmamalıdır; buradaki amacı yalnızca iki girdiyi aynı koşulda kıyaslamak.

## 1. Fark modelde değil, girdide

| hücre | kolon | train | acc | bal.acc | macro-F1 | top3 |
|---|---:|---:|---:|---:|---:|---:|
| old/full | 324 | 59.607 | 0,6359 | 0,5981 | 0,5755 | 0,8387 |
| new/full | 3.483 | 77.930 | **0,2708** | 0,2114 | **0,1995** | 0,4874 |
| old/full_train6000 | 324 | 6.000 | 0,6000 | 0,5119 | 0,5208 | 0,7998 |
| new/full_train6000 | 3.483 | 6.000 | 0,2321 | 0,1603 | 0,1569 | 0,4332 |

Aynı öğrenici, aynı metrik, aynı ağırlık politikası. Eski girdi 0,58 macro-F1,
yeni girdi 0,20. 6.000 örneklik eşleştirilmiş eğitimde de aynı uçurum duruyor
(0,52 / 0,16), yani sorun örnek sayısı ya da GNN mimarisi değil.

EventGCHM koşusunun 0,084 macro-F1'i (`event_training/representative_6000_fullval_v1`)
bunun altında; ama tavanı belirleyen girdi: aynı girdiyle GBDT ancak 0,157'ye çıkıyor.

## 2. Eski girdinin sinyali neredeydi — ve yeni girdide o blok yok

Eski artefaktın blok ablasyonu (hepsi 59.607 train, 400 tur):

| blok | kolon | acc | macro-F1 |
|---|---:|---:|---:|
| old/full | 324 | 0,6359 | 0,5755 |
| old/concepts_only (ilaç + şikâyet düğümleri) | 192 | 0,5643 | 0,4755 |
| **old/complaints_only (yalnız chief complaint)** | **88** | **0,5221** | **0,4329** |
| old/no_complaints (şikâyet çıkarılmış tam girdi) | 236 | 0,3915 | 0,3601 |
| old/demo_labs_vitals | 83 | 0,3460 | 0,3177 |
| old/labs_only | 52 | 0,2692 | 0,2263 |
| old/meds_only | 104 | 0,2104 | 0,1255 |
| old/hx_only | 30 | 0,1688 | 0,0896 |
| old/no_hx | 294 | 0,6335 | 0,5731 |

Tek başına 88 kolonluk **chief complaint** (triyaj yakınması) 0,4329 macro-F1 veriyor;
tam girdinin 0,5755'inin dörtte üçü. Şikâyeti çıkardığınızda tam girdi 0,5755 → 0,3601'e
düşüyor (−0,215) — hiçbir blok bu kadar taşımıyor.

Yeni artefaktta **chief complaint hiç yok**. `event_graph_v1/ingest.py` zaman damgası
olmayan triyajı bilinçli olarak dışarıda bırakıyor (manifest: "Vitals, medrecon, triage,
diagnoses and pyxis excluded"). Yani yeni girdi, eski başarıyı taşıyan tek en güçlü
kanalı tasarım gereği atmış durumda.

Yeni girdinin tüm blokları toplandığında eski girdinin **yalnız laboratuvar** bloğu
seviyesinde: old/labs_only 0,2263 vs new/full 0,1995. Yeni girdi pratik olarak
"sadece laboratuvar, biraz daha ayrıntılı" bir girdidir.

| hücre | kolon | acc | macro-F1 |
|---|---:|---:|---:|
| new/full | 3.483 | 0,2708 | 0,1995 |
| new/event_stats_only (lab değer istatistikleri) | 3.000 | 0,2601 | 0,1901 |
| new/concept_present_only (hangi lab testi var) | 370 | 0,1942 | 0,1247 |
| new/structure_only (kenar/kapsam sayıları) | 108 | 0,1607 | 0,0843 |
| train-majority tabanı | — | 0,1099 | — |

`structure_only` 0,0843 ile taban üstünde ama çok zayıf: graf yapısı tek başına
neredeyse sadece "kaç test istendi" bilgisini taşıyor. Tam girdinin 0,1995'inin
0,1901'i tek başına lab değer istatistiklerinden geliyor; graf yapısı, kenar
ilişkileri ve bilgi (knowledge) düğümleri üstüne yalnız +0,009 ekliyor. Yani
**yeni graf yapısı hemen hemen hiçbir şey katmıyor** — girdi fiilen bir lab tablosu.

## 3. İkinci neden: hedef sızıntısının kapanması

Eski girdide 33 kolon-etiket ismi çakışması var (`hx_acute_kidney_failure` →
"Acute kidney failure", `hx_heart_failure` → "Heart failure", …). `hx_only` tek başına
0,0896 macro-F1 veriyor ve tam girdiden çıkarılması yalnız −0,0024 kaybettiriyor;
yani sızıntı **tek başına** eski skoru açıklamıyor — açıklayan şikâyet bloğu. Yine de
eski artefakt `temporal_clean=false` ve şikâyet/laboratuvar özetleri karar anına göre
doğrulanmamış. Yeni artefakt kesinlikle `storetime ≤ cutoff` diyor.

Sonuç: eski 0,58 bir **post-hoc yeniden kurgu** skoru, yeni 0,20 ise daha dar ama
zaman güvenli bir **tahmin** skoru. İkisi aynı görev değil; doğrudan kıyas yanıltıcı.

## 4. Üçüncü neden: görev zorlaştı

- Etiketleme oranı düştü: 185.888 ziyaretin yalnız 97.618'i (%52,5) etiketlendi;
  64.764 ziyaret hiçbir donmuş sınıfa girmedi, 23.506'sı çoklu sınıf olduğu için düştü.
- Hasta başına birden çok ziyaret var (32.827 hasta çok-ziyaretli). Aynı hastanın
  farklı ziyaretleri farklı etiket taşıyabiliyor; tek-ziyaretli eski kurulumda bu yoktu.
- Sınıf dengesi zorlaştı: en küçük train sınıfı 322 → 127, validation'da 40 → 20.
- Graf boyutu patladı: 34,9M düğüm / 117,2M kenar, 18,4 GB. Bu yüzden GNN koşusu
  6.000 örnekle sınırlandırıldı ve seçim 2.510 düğüm tavanına takıldı; kaynak
  dağılımdan sapma `distribution_audit.md`'de kayıtlı (düğüm medyanı 76 → 44).

## 5. EventGCHM koşusunun kendi sorunu: eğitim bitmemiş

`representative_6000_fullval_v1/history.json`:

train loss 3,337 → 3,081 (10 epoch, hiç düzleşmemiş), validation macro-F1 her epoch
artıyor (0,021 → 0,084) ve son epoch'ta hâlâ yükseliş eğiliminde. 30 sınıf için
`ln(30) = 3,401`; model rastgeleden yalnız 0,32 nat uzakta. Bu model **under-trained**;
0,084 bir yakınsama sonucu değil, 10 epoch'luk bir ara durum. 30 sınıfın 11'inde F1 = 0.

## Özet

1. **Ana sebep:** yeni girdi, eski skorun çoğunu taşıyan chief-complaint kanalını
   içermiyor (tasarım gereği: zaman damgasız triyaj dışlandı). Ablasyonla ölçüldü:
   şikâyet tek başına 0,433, çıkarılınca tam girdi −0,215.
2. Yeni girdi fiilen "sadece laboratuvar" bir girdidir; eski `labs_only` skoruyla aynı
   mertebede. Buna 3.000 kolon eklemek bilgi eklemiyor.
3. Eski skor ayrıca sızıntılı ve `temporal_clean=false`; iki sayı aynı görevin sayısı değil.
4. Görev de zorlaştı: %47,5 örnek etiketsiz, çok-ziyaret, daha küçük nadir sınıflar.
5. GNN koşusu 10 epoch'ta durdurulmuş ve yakınsamamış; 0,084 tavanı değil, ara durumu.

## Ne yapılmalı (öncelik sırasıyla)

1. **Chief complaint'i zaman güvenli biçimde geri getir.** Triyaj yakınması ziyaret
   girişinde kaydediliyor ve ilk lab sonucundan önce mevcut; `triage.csv` için
   `edstays.intime`'ı kabul edilebilir kullanılabilirlik zamanı olarak açıkça belgeleyip
   ayrı bir girdi sürümü (`..._v3_with_complaint`) üret. Bu tek değişikliğin beklenen
   etkisi, ölçülen ablasyona göre diğer tüm seçeneklerden büyük.
2. Aynı ayrımı demografi (yaş/cinsiyet) ve ev ilaçları (medrecon) için de yap; bunlar
   da karar anında biliniyor.
3. GNN'i en az yakınsayana kadar eğit (early-stopping ile 30+ epoch) — mevcut 0,084
   kıyas için kullanılamaz.
4. Kıyas tablolarında eski/yeni girdiyi ayrı satırlarda tut; `temporal_clean` bayrağını
   her satırın yanına yaz.

## Kanıt

- Kod: `comparison/standardized/new_input_diagnosis_v1/probe.py`
- Ham sonuçlar: `results.jsonl`, `results_new_cheap.jsonl`, `old.log`, `new.log`, `new_cheap.log`
- Her satırda artefakt fingerprint, kolon sayısı, örnek sayısı, seed, tur sayısı ve
  validation olasılıklarının sha256'sı var.

Sınırlar: tek seed, seed yayılımı ölçülmedi; bu yüzden ≲0,01'lik farklar yorumlanmadı
(rapordaki farklar 0,2-0,4 mertebesinde, bu eşiğin çok üstünde). Test fold açılmadı.
Bütçe dondurulmuş matched protokolden farklı. 20 hücrenin tamamı tamamlandı;
`new/concept_present_only` iki ayrı süreçte koşup bit-bit aynı olasılıkları üretti
(`24e1fadf…`), yani prob deterministik.
