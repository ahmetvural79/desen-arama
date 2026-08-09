# Değişiklik Günlüğü

## v1.1.0 — 9 Ağustos 2026

BMP arşivlerinin hiç aranamaması ve benzerlik sonuçlarının güvenilmez olması
giderildi. Tüm değişiklikler ölçümle doğrulandı; ayrıntılı teşhis
[`gelistirme-plani-v2.md`](gelistirme-plani-v2.md), ölçüm sonuçları
[`bench-sonuclari.md`](bench-sonuclari.md) dosyasındadır.

### Dosya formatları

- **BMP, TIFF, WebP ve TGA artık indeksleniyor.** v1.0'da varsayılan uzantı
  listesi `.jpg/.jpeg/.png` ile sınırlıydı: arayüz BMP'yi *sorgu* olarak kabul
  ediyor ama tarayıcı BMP dosyalarını hiç görmüyordu. Uzantı listesi üç ayrı
  yerde kopyalanmıştı; tek kaynağa indirildi.
- Ayarlar'da uzantılar serbest metin yerine **onay kutusu listesi**.
- Mevcut yapılandırmalar otomatik göçle yükseltiliyor; kapsam genişleyince
  yeniden tarama öneriliyor (artımlı — indekslenmiş dosyalar tekrar işlenmez).
- Uzantı eşleşmesi büyük/küçük harfe ve eksik noktaya karşı toleranslı.

### Benzerlik skorları

- **Skorlar kalibre edildi.** Eskiden alakasız görseller %50 civarı skor
  alıyordu (algısal hash'te alakasızlığın beklenen mesafesi bit sayısının
  yarısıdır). Artık alakasız ≈ 0, birebir = 1. AI modunda taban
  **kütüphaneden** öğreniliyor: hepsi halı olan bir arşivde DINOv2 alakasız iki
  deseni bile ~0.88 kosinüsle eşlediği için ölçek koleksiyonun kendi medyanına
  göre yeniden hesaplanıyor.
- Hash arama eşiği 12 → 22. Eski eşik yalnızca birebir kopyaları geçiriyor,
  aynı desenin farklı renkli veya kırpılmış varyantlarını eliyordu. Eşikler
  artık hash boyutuna göre otomatik ölçekleniyor.
- Sonuçlarda **kalite bandı** (Kopya / Çok benzer / Benzer / Zayıf).
- **Renk harmanlaması çarpımsal oldu.** Renk histogramı kesişimi kalibre
  değildir: benzer paletteki alakasız görseller 0.9+ alırken aynı desenin
  farklı renkli varyantı 0.0 alır. Eski toplamsal formül bu yüzden sıralamayı
  tersine çeviriyordu. Artık desen kapı, renk en fazla α oranında düzeltme.
  Varsayılan α **0**'a çekildi — colorway varyantları halı arşivinde birincil
  senaryodur; kaydırıcıdan yükseltilebilir.

### AI (DINOv2) modu

- **Bozuk model adresi düzeltildi.** `dinov2-base` deposu mevcut değildi
  (HTTP 401); "DINOv2-B (kaliteli)" seçeneği her zaman başarısız oluyordu. Her
  iki model `onnx-community` kaynaklarına taşındı.
- **Sessiz fallback kaldırıldı.** Model bulunamayınca çok daha zayıf bir
  çıkarıcıya sessizce düşülüyordu (ölçümde bu yedek, farklı bir deseni aynı
  desenin döndürülmüş hâlinden daha benzer sayıyordu). Artık durum bildiriliyor,
  indirme ilerleme çubuğu ve iptalle öneriliyor, reddedilirse hızlı moda
  dönülüyor.
- İndirilen model **SHA256** ile doğrulanıyor; aynalar sırayla deneniyor;
  kurumsal ağ engelinde çevrimdışı kurulum yolu gösteriliyor.
- **Ön işleme letterbox oldu.** Eski merkez kırpma halının bordürünü tamamen
  atıyordu; bordür halıda ana ayırt edici öğedir. Kırpılmış sorguda aynı desenin
  kosinüsü 0.867 → 0.918 çıktı.

### Kısmi / kırpılmış sorgular

- **Karo (tile) indeksleme eklendi.** Her görsel tam kare + N×N karo olarak
  gömülüyor, sorguda karolar arası maksimum alınıyor. Kırpılmış sorgu
  recall@10 0.00 → 0.40. İndekslemeyi ~10× yavaşlattığı için varsayılan
  kapalı; Ayarlar'dan açılır.
- **Çok ölçekli sorgu** (sorgunun merkez kırpması da aranır) — indeksi
  büyütmez, varsayılan açık.
- **Hibrit mod yeniden tasarlandı.** Adaylar artık hash ∪ embedding
  birleşiminden geliyor (eskiden yalnızca hash'ten, bu da hibridin recall'ını
  hash'inkiyle sınırlıyordu) ve iki skorun **maksimumu** alınıyor. Ölçümde hash
  colorway varyantlarında (0.65 vs 0.25), karo destekli embedding ise kırpmada
  (0.40 vs 0.00) güçlü çıktı; maksimum ikisini de koruyor:
  hit@10 0.85 → **0.98**, colorway recall 0.25 → **0.67**. Hibrit arayüzde
  "önerilen" olarak işaretlendi.
- Aday vektörleri tek toplu sorguyla çekiliyor (eskiden aday başına bir sorgu).

### Dayanıklılık

- **Embedding imzası.** Vektörlerin hangi model, ön işleme ve karo ayarıyla
  üretildiği kaydediliyor; koşullar değişince embedding'ler otomatik yeniden
  hesaplanıyor. Eskiden ön işleme değişince eski indeks sessizce yanlış sonuç
  üretirdi. Metadata korunur, pahalı tarama tekrarlanmaz.
- Veritabanı şeması 1 → 2 (karo desteği); göç otomatik.

### Ölçüm

- `tests/bench.py` artık **senaryo kırılımlı** recall@10 raporluyor
  (kopya / kırpılmış / orijinal / farklı renk / döndürülmüş). Bu kırılım
  olmadan "kırpılmış sorgular hiç çalışmıyor" gibi arızalar toplam skorun
  içinde görünmez kalıyordu.
- Test sayısı 60 → 129.

## v1.0.0

İlk sürüm.
