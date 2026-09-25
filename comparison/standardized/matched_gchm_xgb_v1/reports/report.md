# Eşleştirilmiş GCHM–XGBoost: doğrulama raporu

**Yalnız doğrulama: n=7448; test_evaluated=false; temporal_clean=false.**
GCHM, GNN ailesidir; XGBoost harici tabular referanstır. Aşağıdaki eşleşmeler aile içi sıralama değildir.
Her checkpoint ilk maksimum (6 ondalık) macro-F1 ile seçildi; ensemble üç seed'in eşit olasılık ortalamasıdır.

## Metrikler

| Model / ağırlık / seed | Macro F1 | Dengeli doğruluk | Doğruluk | Micro F1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| gchm/none/1234 | 0.5655 | 0.5534 | 0.6312 | 0.6312 | 0.8277 | 0.8961 |
| gchm/none/1235 | 0.5672 | 0.5621 | 0.6337 | 0.6337 | 0.8272 | 0.8978 |
| gchm/none/1236 | 0.5641 | 0.5488 | 0.6281 | 0.6281 | 0.8225 | 0.8921 |
| gchm/none/ensemble | 0.5748 | 0.5647 | 0.6412 | 0.6412 | 0.8375 | 0.9051 |
| gchm/sqrt_inverse/1234 | 0.5615 | 0.5968 | 0.6220 | 0.6220 | 0.8210 | 0.8882 |
| gchm/sqrt_inverse/1235 | 0.5637 | 0.5920 | 0.6238 | 0.6238 | 0.8244 | 0.8922 |
| gchm/sqrt_inverse/1236 | 0.5526 | 0.5816 | 0.6149 | 0.6149 | 0.8208 | 0.8879 |
| gchm/sqrt_inverse/ensemble | 0.5706 | 0.5991 | 0.6293 | 0.6293 | 0.8314 | 0.9009 |
| xgboost/none/1234 | 0.5856 | 0.5750 | 0.6459 | 0.6459 | 0.8441 | 0.9055 |
| xgboost/none/1235 | 0.5862 | 0.5756 | 0.6472 | 0.6472 | 0.8464 | 0.9084 |
| xgboost/none/1236 | 0.5870 | 0.5763 | 0.6461 | 0.6461 | 0.8436 | 0.9079 |
| xgboost/none/ensemble | 0.5856 | 0.5745 | 0.6463 | 0.6463 | 0.8455 | 0.9071 |
| xgboost/sqrt_inverse/1234 | 0.5808 | 0.5998 | 0.6403 | 0.6403 | 0.8417 | 0.9107 |
| xgboost/sqrt_inverse/1235 | 0.5802 | 0.6000 | 0.6394 | 0.6394 | 0.8389 | 0.9117 |
| xgboost/sqrt_inverse/1236 | 0.5792 | 0.5984 | 0.6399 | 0.6399 | 0.8416 | 0.9108 |
| xgboost/sqrt_inverse/ensemble | 0.5786 | 0.5975 | 0.6390 | 0.6390 | 0.8402 | 0.9115 |
| gchm/none seed ort. ± örnek SS | 0.5656 ± 0.0015 | 0.5548 ± 0.0068 | 0.6310 ± 0.0028 | 0.6310 ± 0.0028 | 0.8258 ± 0.0029 | 0.8953 ± 0.0030 |
| gchm/sqrt_inverse seed ort. ± örnek SS | 0.5593 ± 0.0058 | 0.5901 ± 0.0077 | 0.6203 ± 0.0047 | 0.6203 ± 0.0047 | 0.8221 ± 0.0020 | 0.8894 ± 0.0024 |
| xgboost/none seed ort. ± örnek SS | 0.5863 ± 0.0007 | 0.5756 ± 0.0006 | 0.6464 ± 0.0007 | 0.6464 ± 0.0007 | 0.8447 ± 0.0015 | 0.9073 ± 0.0016 |
| xgboost/sqrt_inverse seed ort. ± örnek SS | 0.5801 ± 0.0008 | 0.5994 ± 0.0009 | 0.6399 ± 0.0005 | 0.6399 ± 0.0005 | 0.8407 ± 0.0016 | 0.9111 ± 0.0005 |

## Ensemble farkları ve koşullu %95 güven aralıkları

A−B yönü satır adındadır. Tek-seed eşleşmeleri, sqrt−none farkları, sınıf recall aralıkları ve tüm ayrıntılar JSON'dadır.
| Karşılaştırma | Δ Macro F1 [GA] | Δ Dengeli doğruluk [GA] | Δ Doğruluk [GA] | Recall kazanılan sınıf |
|---|---:|---:|---:|---:|
| gchm_minus_xgboost/none/ensemble | -0.0108 [-0.0232, +0.0006] | -0.0097 [-0.0212, +0.0014] | -0.0051 [-0.0117, +0.0017] | 12/30 |
| gchm_minus_xgboost/sqrt_inverse/ensemble | -0.0080 [-0.0181, +0.0022] | +0.0016 [-0.0095, +0.0123] | -0.0097 [-0.0164, -0.0030] | 13/30 |
| sqrt_minus_none/gchm/ensemble | -0.0043 [-0.0139, +0.0061] | +0.0344 [+0.0242, +0.0447] | -0.0119 [-0.0187, -0.0059] | 19/30 |
| sqrt_minus_none/xgboost/ensemble | -0.0071 [-0.0143, +0.0012] | +0.0231 [+0.0153, +0.0316] | -0.0074 [-0.0119, -0.0026] | 19/30 |

## Nadir sınıflar ve precision–recall değiş tokuşu

Nadir sınıflar yalnız eğitim desteğinin alt çeyreğiyle (sabit sınıf sıralı bağ çözümü) tanımlandı. Eski doğrulama ≤100 / >400 grupları yalnız keşifsel karşılaştırmadır. FP tüm doğrulama kümesinden sayılır.
| Ensemble | Eğitim-nadir precision | Eğitim-nadir recall | Eğitim-nadir F1 | Eğitim-nadir FP | Nadir-dışı recall | Eski ≤100 recall | Eski >400 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| gchm/none/ensemble | 0.5297 | 0.3948 | 0.4328 | 155 | 0.6165 | 0.4250 | 0.6677 |
| gchm/sqrt_inverse/ensemble | 0.3988 | 0.5098 | 0.4434 | 341 | 0.6263 | 0.5367 | 0.6226 |
| xgboost/none/ensemble | 0.4770 | 0.4092 | 0.4363 | 178 | 0.6248 | 0.4563 | 0.6713 |
| xgboost/sqrt_inverse/ensemble | 0.4109 | 0.4705 | 0.4365 | 289 | 0.6362 | 0.5131 | 0.6412 |

## Kavram sayısına göre ensemble

| Ensemble | Grup | n | Macro F1 | Dengeli doğruluk | Doğruluk |
|---|---|---:|---:|---:|---:|
| gchm/none/ensemble | 0-1 concepts | 1652 | 0.5267 | 0.5423 | 0.6955 |
| gchm/none/ensemble | 2-4 concepts | 2504 | 0.5893 | 0.5789 | 0.6901 |
| gchm/none/ensemble | 5-9 concepts | 2186 | 0.5301 | 0.5185 | 0.5883 |
| gchm/none/ensemble | 10+ concepts | 1106 | 0.5091 | 0.5145 | 0.5543 |
| gchm/sqrt_inverse/ensemble | 0-1 concepts | 1652 | 0.5589 | 0.6102 | 0.6828 |
| gchm/sqrt_inverse/ensemble | 2-4 concepts | 2504 | 0.6100 | 0.6334 | 0.6817 |
| gchm/sqrt_inverse/ensemble | 5-9 concepts | 2186 | 0.5194 | 0.5547 | 0.5672 |
| gchm/sqrt_inverse/ensemble | 10+ concepts | 1106 | 0.4955 | 0.5355 | 0.5533 |
| xgboost/none/ensemble | 0-1 concepts | 1652 | 0.5793 | 0.5899 | 0.7064 |
| xgboost/none/ensemble | 2-4 concepts | 2504 | 0.6081 | 0.5960 | 0.6961 |
| xgboost/none/ensemble | 5-9 concepts | 2186 | 0.5236 | 0.5084 | 0.5823 |
| xgboost/none/ensemble | 10+ concepts | 1106 | 0.5063 | 0.5109 | 0.5705 |
| xgboost/sqrt_inverse/ensemble | 0-1 concepts | 1652 | 0.5799 | 0.6296 | 0.6992 |
| xgboost/sqrt_inverse/ensemble | 2-4 concepts | 2504 | 0.6114 | 0.6258 | 0.6845 |
| xgboost/sqrt_inverse/ensemble | 5-9 concepts | 2186 | 0.5218 | 0.5375 | 0.5787 |
| xgboost/sqrt_inverse/ensemble | 10+ concepts | 1106 | 0.5193 | 0.5495 | 0.5651 |

## Yöntem ve sınırlar

- 2000 tekrar; seed=20260919. Gerçek sınıfa göre tabakalı eşleştirilmiş hasta bootstrap'ı: her sınıfta ortak (A tahmini, B tahmini) hücreleri multinomial örneklenir. Aynı hastayı iki model için birlikte yeniden örneklemeyle dağılımsal olarak eşdeğerdir; sınıf destekleri sabittir. Aralıklar yuvarlanmamış metriklerden hesaplanır.
- Seed örnek SS (ddof=1) eğitim rastgeleliğinin betimlemesidir; ensemble veya bootstrap aralığı değildir.
- temporal_clean=false: geçmiş tanı alanları mevcut/gelecek tanıları içerebilir; tüm ziyaret laboratuvarları ve yatış-geneli vital özetlerinin karar anında erişilebilirliği doğrulanmamıştır. Bunlar klinik erken tahmin değil, sızıntı riski taşıyan doğrulama/rekonstrüksiyon sonuçlarıdır.
- Ağırlıklandırma nadir sınıf recall kazanırken precision ve çoğunluk doğruluğunu düşürebilir; FP/FN ve precision birlikte okunmalı, klinik maliyet analizi yapılmalıdır.
- Aynı doğrulama kümesi checkpoint seçimi ve raporlama için kullanıldı. Sabit seçilmiş checkpoint'lere koşullu bootstrap seçim yanlılığını veya eğitim belirsizliğini kapsamaz.
- Bilgi, bölmeler, ağırlık politikası, metrik ve seed eşleşir; hesap bütçesi eşit değildir: GCHM 30 epoch, XGBoost 1200 tur. Mimari, optimizer ve aranan checkpoint sayısı farklıdır.
- Güven aralıkları sınıf bazında noktasaldır; çoklu karşılaştırma düzeltmesi yoktur. Keşifsel sonuçlar doğrulayıcı üstünlük veya klinik kullanım kanıtı değildir.
- test_evaluated=false; test tahmini/metriği üretilmedi. Harici, zaman-güvenli değerlendirme gerekir.

JSON: tüm tek-seed/ensemble metrikleri; her sınıf precision, recall, F1, TP, FP, FN ve destek; gruplar; eşleşmiş farklar ve koşullu aralıklar; girdi/kaynak/checkpoint/tahmin SHA-256 kayıtları.
