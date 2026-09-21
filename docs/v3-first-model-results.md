# v3 klinik graf: ilk model sonuçları

Tarih: 2026-09-20. **Validation only — test fold hiç açılmadı.**
Artefakt: `clinical_graph_v3_full` (185.888 graf, tanı katmanlı, sızıntı doğrulandı=0).
Etiketli: 77.930 train / 9.582 validation. Çoğunluk tabanı accuracy: 0,1099.

## Sonuçlar

| model | param | acc | bal.acc | macro-F1 | top3 |
|---|---:|---:|---:|---:|---:|
| main (tam graf) seed1234 | 266.142 | 0,5123 | 0,4749 | 0,4531 | 0,7114 |
| main seed1235 | 266.142 | 0,5134 | 0,4596 | 0,4553 | 0,7163 |
| main seed1236 | 266.142 | 0,5114 | 0,4758 | 0,4590 | 0,7194 |
| **nomp** (mesaj geçirme YOK) seed1234 | 266.142 | 0,5147 | 0,4829 | **0,4653** | 0,7184 |
| nomp seed1235 | 266.142 | 0,5186 | 0,4683 | 0,4606 | 0,7244 |
| nopayload (delta silinmiş) seed1234 | 266.142 | 0,5148 | 0,4646 | 0,4556 | 0,7228 |
| **XGBoost (tablo kontrol)** | 5.319 özellik | **0,5380** | 0,4826 | **0,4697** | **0,7464** |

Tüm GNN arm'ları **birebir aynı parametre sayısında** (266.142) — fark kapasiteden değil.

## Ölçülen seed yayılımı

| arm | n | macro-F1 ort. | SS |
|---|---:|---:|---:|
| main | 3 | 0,4558 | 0,0030 |
| nomp | 2 | 0,4629 | 0,0033 |

**Eşik: 0,0033.** Bundan küçük farklar gürültüdür.

## Bulgular

### 1. Mesaj geçirme kazanç sağlamıyor — hatta zarar veriyor

`nomp` (mesaj geçirme kapalı, sadece düğüm özellikleri + pooling) ile `main` farkı
**−0,0071 = 2,1× seed SS**. Gürültü eşiğinin üstünde, yani gerçek ama **ters yönde**.

`nopayload` (delta/aralık yükü sıfırlanmış) − `nomp` = **−0,0074 = 2,2× SS**.
Yani kenar yükünü eklemek de yardımcı olmuyor.

Bu, grafın *bilgi taşıdığını* gösteren denetimle çelişmiyor: audit grafın tabloya
indirgenemez olgular içerdiğini kanıtladı, ama **bu modelin o olgulardan
faydalanamadığını** gösteriyor. İki ayrı iddia.

### 2. Tablo kontrolü hepsini geçiyor

XGBoost, aynı graflardan düzleştirilmiş 5.319 özellikle **0,4697 macro-F1** ve
**0,5380 accuracy** alıyor. GNN ortalamasından **+0,0139 = 4,7× seed SS** önde.
Accuracy farkı daha büyük: +0,025.

Bu, MedGNN'in eski bulgusunun tekrarı — girdi değişti, sonuç değişmedi.

### 3. Sonuçlar mutlak olarak iyi

Çoğunluk tabanı 0,1099 iken 0,54 accuracy. 30 sınıfın **hiçbirinde F1 = 0 yok**
(v1'de 11 sınıf sıfırdı). En iyi sınıflar klinik olarak anlamlı:

| sınıf | F1 | n |
|---|---:|---:|
| Major depressive disorder | 0,800 | 442 |
| GI bleed | 0,783 | 503 |
| Epilepsy | 0,754 | 180 |
| Alcohol abuse with intoxication | 0,709 | 366 |
| Back or spine pain | 0,674 | 755 |

Destek-F1 korelasyonu yalnız 0,383 — yani başarı sadece sınıf büyüklüğünden gelmiyor.
Epilepsy 180 örnekle 0,754 alıyor.

## Yol boyunca bulunan ve düzeltilen hata

İlk koşuda tüm mesaj geçirmeli arm'lar çoğunluk sınıfında dondu (loss 3,3041 sabit,
macro-F1 0,0066). Teşhis: sum-pooling graf boyutuyla ölçekleniyordu, graflar 1–2.500
düğüm arası değiştiği için logit ölçeği patlıyordu (|z|max ≈ 80, std 9,06) → softmax
doygun → gradyan ölü. `pool_norm` (LayerNorm) eklendi; logit std 9,06 → 0,236 ve
graf boyutundan bağımsız hale geldi. Mekanizma kontrolleri düzeltme sonrası 6/6 geçti.

## Mekanizma doğrulaması (eğitimsiz)

```
PASS  message_not_additively_separable       0.2996   (roundoff'un ~10^6 kati)
PASS  receiver_state_changes_message         0.3286
PASS  edge_payload_ablation_equal_params     0.0663   (+ parametre sayisi esit)
PASS  degenerate_graph_finite                0.7952
PASS  batching_does_not_mix_graphs           2.4e-07
PASS  gradients_reach_every_block            []
```

Yani mesaj fonksiyonu gerçekten hem göndereni hem alıcıyı hem kenar yükünü karıştırıyor.
Mimari çalışıyor; sorun mimarinin bozuk olması değil.

## Sınırlar

- **Test fold açılmadı.** Bunlar model-seçim skorları, held-out sonuç değil.
- Epoch seçimi validation macro-F1 ile yapıldı, aynı fold raporlanıyor.
- 12 epoch sabit bütçe; `main` seed1234 epoch 7'de, seed1236 epoch 10'da seçildi —
  yakınsama tam değil, daha uzun eğitim sonucu değiştirebilir.
- `nopayload` ve XGBoost tek seed; yayılımları ölçülmedi.
- Hiperparametre araması **yapılmadı** (iki taraf için de).
- XGBoost 5.319 özellik ile GNN 266k parametre — eşit hesap bütçesi değil.
- Artefakt `temporal_clean=true` ama `storetime` bir kullanılabilirlik vekili.

## Dürüst özet

Graf yapısı v1'e göre çok daha iyi (F1=0 sınıf yok, 0,11 → 0,54 accuracy). Ama
**grafın kendisi bu modelde kazanç üretmiyor**: mesaj geçirmeyi kapatmak sonucu
iyileştiriyor, düz bir tablo modeli ise hepsini geçiyor. Graf bilgi içeriyor
(audit kanıtladı), model o bilgiyi kullanamıyor.

## Sonraki adımlar

1. Daha uzun eğitim + LR scheduler — `main` yakınsamamış olabilir.
2. Kenar tipine özel mesaj fonksiyonu (şu an tüm ilişkiler tek MLP paylaşıyor;
   `baseline_of` ile `instance_of` aynı dönüşümden geçiyor).
3. XGBoost'un kazandığı sınıfları tek tek incele — nerede kaybediyoruz?
4. Seed sayısını 5'e çıkar, `nopayload` ve XGBoost için de yayılım ölç.
