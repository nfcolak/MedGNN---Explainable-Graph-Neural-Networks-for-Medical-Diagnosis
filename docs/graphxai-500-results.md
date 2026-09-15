# GraphXAI: gerçek 500 kişilik sonuçlar

**Tamamlandı:** ortak kanonik test kohortunda ProtGNN, GSAT ve GraphCare için ayrı ayrı 500 kayıt; 1.500 model-kişi kaydı ve 4.500 gerçek GraphXAI açıklayıcı sonucu. `star`, `seed=1234`; mevcut eğitilmiş checkpoint’ler, yeniden eğitim yok.

| Model | GraphXAI açıklayıcı | n | Fidelity+ ↑ | Fidelity− ↓ | Sparsity (90% kütle) |
|---|---|---:|---:|---:|---:|
| protgnn | GradExplainer | 500 | 0.289649 | 0.458721 | 0.025552 |
| protgnn | IntegratedGradExplainer | 500 | 0.412981 | 0.333737 | 0.397364 |
| protgnn | GNNExplainer | 500 | 0.249205 | 0.507873 | 0.104888 |
| gsat | GradExplainer | 500 | 0.295626 | 0.270092 | 0.018816 |
| gsat | IntegratedGradExplainer | 500 | 0.277159 | 0.292355 | 0.486765 |
| gsat | GNNExplainer | 500 | 0.185136 | 0.387112 | 0.091110 |
| graphcare | GradExplainer | 500 | 0.370278 | 0.057964 | 0.363489 |
| graphcare | IntegratedGradExplainer | 500 | 0.393683 | 0.033283 | 0.410326 |
| graphcare | GNNExplainer | 500 | 0.117135 | 0.331761 | 0.106472 |

Tablo ortalamalarıdır; standart sapmalar `results.csv` içindedir. Fidelity±, tahmin edilen sınıfın olasılık düşüşüdür: `+` en önemli düğümleri sıfırlar, `−` yalnızca onları tutar. Ortak kural `k=max(1,floor(0.2*N))`; kenarlar sabittir. Sparsity, sabit top-k oranı değil, önem kütlesinin %90’ını korurken dışlanabilen düğüm oranıdır. Negatif fidelity değerleri mümkündür.

## Doğrulama

- Tüm 500 kişi kanonik test bölümünden deterministik seçildi; üç yöntemde materyalize düğüm/kenar fingerprint’leri, etiket sırası ve kohort aynı.
- Her modelde 500/500 wrapper testi; maksimum mutlak logit farkı **0.0**.
- 4.500/4.500 açıklayıcı sonucu mevcut, tüm metrikler sonlu; tüm gözlenen backward amaçları sonlu.
- Checkpoint, dataset, split ve mevcut cache SHA-256 değerleri çalışma öncesi/sonrası aynı. Eski 50 kişilik kanonik kohort değiştirilmedi.
- IntegratedGradExplainer: 32 adım. GNNExplainer: 50 gerçek optimizasyon epoch’u. Vendored kaynak ve uyumluluk adaptörü hash/snapshot’ları kaydedildi.
- Önce 3 kişi × 3 model smoke, sonra iki kenarsız gerçek grafı da içeren 5 kişi × 3 model smoke başarılı.
- **106 test geçti**, 14 bağımlılık deprecation uyarısı; `verification_tests.log`.

## Kapsam ve sınır durumları

- protgnn: kohort sınıflandırma doğruluğu 54.20%; başarılı ana çalışma süresi 53.08 saniye.
- gsat: kohort sınıflandırma doğruluğu 63.40%; başarılı ana çalışma süresi 83.47 saniye.
- graphcare: kohort sınıflandırma doğruluğu 59.00%; başarılı ana çalışma süresi 49.96 saniye.
- Üç modelin ana çalışması toplam 186.52 saniye; smoke/test/önkontrol ve tekrar girişimi hariç.
- Her yöntemde iki tek-düğümlü/kenarsız graf var: kenar gradyanı `not_applicable_edgeless`, doğrulanmış gibi gösterilmedi. GNNExplainer’ın boş kenar entropisi sıfır; gerçek özellik-maskesi optimizasyonu çalıştı. Kenar-kütlesinden düğüm indirgeme bu iki grafta sıfır skor verir; düğüm seçimi zorunlu tek düğümdür.
- protgnn: sıfır tahmin kenar gradyanı 14 kayıt (iki kenarsız kayıt dahil). Bunlar saklandı/işaretlendi; kişi değiştirme veya rastgele açıklama fallback’i yok.
- gsat: sıfır tahmin kenar gradyanı 2 kayıt (iki kenarsız kayıt dahil). Bunlar saklandı/işaretlendi; kişi değiştirme veya rastgele açıklama fallback’i yok.
- graphcare: sıfır tahmin kenar gradyanı 2 kayıt (iki kenarsız kayıt dahil). Bunlar saklandı/işaretlendi; kişi değiştirme veya rastgele açıklama fallback’i yok.
- GraphCare embedding maskesi, diğer iki model sürekli özellik girdisi kullanır; ortak düğüm metriği aynı olsa da özellik parametreleştirmesi farklıdır. Tek seed/topoloji/altkohort sonuçlarından genel üstünlük çıkarılmamalıdır.

## Karşılaşılan hatalar

- Önceki çalışma cihaz uyumsuzluğu: mevcut CPU/saved-config rekonstrüksiyonu smoke ve 500 kayıtla doğrulandı.
- Önceki GNNExplainer kontrolü, bağlı fakat sıfır gradyanı yanlışlıkla desteklenmeyen model sayıyordu; regresyon testi önce başarısız oldu, ardından ayrım düzeltildi.
- Kenarsız graflarda upstream `mean([])` NaN amaç üretimi gerçek testle yakalandı; yalnızca boş kenar entropisi sıfıra düzeltildi. Dolu grafların algoritması değişmedi.
- Runtime arka plan bildirimi exit-code olmadan erken “exited” raporladı; asıl iş dosya kanıtlarına göre devam edip tamamlandı. Bu yanıltıcı bildirim üzerine gereksiz GSAT tekrar girişimi 500 açıklama hesapladı fakat mevcut hedefi ezmemek için `FileExistsError` ile ihracı reddedildi. Ana sonuçlar değiştirilmedi; tekrar günlüğü saklandı.

## Artifaktlar

- Çalışma kökü: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/standardized/graphxai_500_20260914T144701`
- Karşılaştırmalı CSV: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/standardized/graphxai_500_20260914T144701/results.csv`
- Makine doğrulaması: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/standardized/graphxai_500_20260914T144701/verification.json`
- Üç-yöntem kayıt doğrulaması: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/standardized/graphxai_500_20260914T144701/full_outputs/star/seed_1234/validation.json`
- Kişi kayıtları: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/standardized/graphxai_500_20260914T144701/full_outputs/star/seed_1234`
- Kohort: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/standardized/graphxai_500_20260914T144701/cohort500.json`
- Kaynak anlık görüntüleri: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/standardized/graphxai_500_20260914T144701/source_snapshot`

Kişi tanımlayıcıları yalnızca yerel kanıt artifaktlarında; bu raporda hasta kimliği veya ham klinik veri yok. Commit/push yapılmadı.
