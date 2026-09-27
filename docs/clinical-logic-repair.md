# Klinik graf mantık onarımı

## Kapsam

Kullanıcı, incelemedeki hataların düzeltilmesini ve açıkça **testing phase** kapsamında
dar testlerin yazılıp çalıştırılmasını onayladı. Tam model eğitimi, tam klinik graf
üretimi ve büyük ham laboratuvar taraması bu çalışmanın kapsamı değildir. Önceden
kirli çalışma ağacı korunur; commit/push yapılmaz.

Kaynak ve metadata geri dönüş kopyası:
`.hermes/checkpoints/clinical-logic-repair-20260922-191824/`.
Bu klasörde değişikliklerden önceki kaynaklar, bağımsız metadata kopyaları ve
`preexisting.patch` bulunur. Eski deney sonuçları yeni kodla üretilmiş gibi güncellenmez.

## Devam ediyor — kapsamı sınırlandırılmış

Kullanıcı çalışmaya devam edilmesini istedi. Başlangıçtaki kirli çalışma ağacı
korunuyor; commit/push yapılmadı. Yeni bağımsız kod incelemeleri sürüyor.

**Son gözlenen doğrulama:** sekiz `tests/test_clinical_*.py` dosyasında **133 passed**;
`python3 -m comparison.standardized.clinical_graph_v2.mechanism_check` → **21/21 passed**;
scoped `compileall` ve `git diff --check` başarılı. Entegrasyon testleri sentetik
4 train + 2 validation örneğiyle üç GNN ailesinin bir epoch'unu ve eşleşen XGBoost
giriş noktasını çalıştırdı; gerçek hasta verisiyle tam model eğitimi yapılmadı.
Full ve max6 index'lerinin metadata readback'i **9/9 kontrol** verdi: `index_ready`,
hash/sayı doğruluğu, tek-link metadata dosyaları, exact-byte yedekler ve temiz kilit/
pending durumu.
Veri/metadata bağımsız incelemesi duraklatma sonrasında `passed: true` döndü
(`deleg_e9300aa3`; engelleyici bulgu yok). Full ve max6 kaynak index metadata'sı
yedekli onarıldı; ikinci uygulama iki artifact için de `already_repaired` döndürdü.
Metadata hardlink'leri ayrıldı; graf/target bytes değiştirilmedi. Girdi/model ve
graph-information bağımsız incelemeleri sürüyor. Gerçek klinik graph rebuild ve
model eğitimi ayrı yetki gerektirir; başlatılmadı.

İlk iki bağımsız kod incelemesi üç engelleyici parity bulgusu verdi: tabular
`source_code` binding'i, stratify `preprocessing_sha256` eşleşmesi ve rewiring
karşılaştırmalarında iki kolun da payload'sız olması. Her bulgu için regresyon
önce kırmızı görüldü, düzeltmeler uygulandı ve testler yeşil oldu. Güncel kodun
bağımsız yeniden incelemesinde üç düzeltme de `passed: true` aldı. Ek regresyonlar
missing/empty preprocessing hash'lerini ve aynı kohortta uyumlu bir karşılaştırma
ile uyumsuz rewiring kolunun birlikte bulunmasını da kapsıyor.

### Devam todo'su

- [x] Veri/metadata güvenlik incelemesi: `passed: true`, güvenlik/mantık bulgusu yok.
  Salt okunur rapor: `deleg_e9300aa3`; gerçek onarım veya yeniden üretim yapılmadı.
- [x] Full/max6 metadata onarımı ve readback: hardlink'ler ayrıldı, canonical
  cohort/index hash'leri doğrulandı, yedekler exact-byte eşleşti, ikinci apply
  `already_repaired` oldu. Graf ve hedef sidecar'ı değiştirilmedi.
- [x] Girdi/model/evaluation ve graph-information/rewiring/aggregation bağımsız
  ilk incelemelerini al; üç parity bulgusunu dar regresyonla düzelt.
- [x] Güncel düzeltmeleri bağımsız yeniden incelet; engelleyici bulgu kalmadı.
- [x] README komutlarını yeni, boş logic-v2 artifact/run yollarına ve açık
  `--graph-root` kullanımına güncelle; eski skor ve kapasite sayılarını tarihsel işaretle.
- [x] Son değişikliklerden sonra ortak dar testleri (**133 passed**), mekanizma
  kontrollerini (**21/21**), scoped `compileall`, `git diff --check` ve metadata
  readback'ini (full/max6 **9/9**) tekrarla; sonuçları bu rapora kaydet.
- [ ] **Ayrı onayla:** yeni klasöre düzeltilmiş graf + hedef sidecar üretimi ve
  karşılaştırma kollarının yeniden eğitimi. Eski grafları metadata ile yükseltme.

### Önceki ara checkpoint (tarihsel)

- Hasta-eşit metrik: farklı ziyaret tanıları korunur; ağırlık değerlendirilen
  ziyaret sayısının tersidir. Macro-F1 tanımlı sınıf evrenini kullanır.
- Alt-grup aralıkları: sıfır + yarı-açık pozitif aralıklar tam ve ayrık bölünür;
  her tablo her ziyaretin bir kez sayılmasını zorunlu kılar.
- Tahmin hizası: yeni çıktılarda ziyaret kimliği saklanır; yalnız hasta kimliği
  taşıyan tekrarlı ziyaret dizileri belirsiz sayılıp reddedilir. Önbellek graf
  içeriği ve hedef eşlemesi hash'lerine bağlanır. Kontrol kolunun kimlikleri de doğrulanır.
- Yeni eğitim/tam tablo kontrolü eski klinik graf manifestlerini reddeder:
  `logic_contract_version=clinical_graph_logic_v2` ve açık zamansal belirsizlik gerekir.
- Girdi, model/kontrol, veri üretimi/metadata ve bilgi denetimi düzeltmeleri
  ayrı dosya sahiplikleriyle yürütülüyor; ortak doğrulama bekleniyor.

Ara doğrulama (yalnız parent kapsamı):

```bash
python3 -m pytest tests/test_clinical_evaluation.py tests/test_clinical_contracts.py tests/test_clinical_stratify.py -q
```

Gözlenen sonuç: **14 passed**. Bu sayı entegrasyonun tamamlandığı anlamına gelmez.
İki mevcut `clinical_graph_v3_full` / `clinical_graph_v3_max6` manifestinin yeni
koşu guard'ı tarafından reddedildiği ayrıca gözlendi. Ağ kullanılmayan testlerde
ortamın urllib3/LibreSSL başlangıç uyarısı var; bu araştırma kodunun başarısızlığı değildir.

## Tamamlama kontrolleri

- [x] Ortak vital/demografi/birim sözleşmesi, train-only fit ve kaydet/yükle eşitliği.
- [x] Sayısal-yük ablasyonu, tam mesaj mekanizması ve NOMP etkin parametre sayısı.
- [x] Derece-koruyan rewiring, deterministik sonuçlar ve değişmeyen grafik oranı.
- [x] Geçmiş/hedef ICD eşleme birliği ve tekil önceki ziyaret sayımı.
- [x] Zaman varsayımlarının üretici ve manifestte ayrı işaretlenmesi.
- [x] Metadata onarımının testleri; gerçek metadata üzerinde yedekli uygulama ve readback.
- [x] Bilgi denetiminde yön, çokluk, payload ve bilinmeyen bilgi ayrımı.
- [x] Seed belirsizliği ve uyumsuz koşu gruplarının doğru raporlanması.
- [ ] Son dar ortak test paketi ve bağımsız girdi/analiz kod incelemeleri.

## Yeniden üretim sınırı

Kod düzeltmesi, eski `graphs.jsonl` içindeki ICD geçmişini geriye dönük düzeltmez.
Metadata onarımı da graf üretimi değildir. Yeni deneylerden önce yeni bir çıktı
klasörüne graf üretimi ve aynı kohorta bağlı hedef sidecar'ı gerekir; bu çalışmada
kendiliğinden başlatılmaz. Kaynak zaman damgası olmayan alanlar için gerçek
kayıt zamanı uydurulmaz; `temporal_clean=false` bu belirsizliği ifade eder.
