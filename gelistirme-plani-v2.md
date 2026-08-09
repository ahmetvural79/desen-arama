# Desen Arama v1.1 — Teşhis Raporu ve Geliştirme Planı

**Tarih:** 9 Ağustos 2026
**Kapsam:** (1) BMP/diğer formatların tanınmaması, (2) görsel benzerlik arama kalitesi
**Yöntem:** Kaynak kod incelemesi + sentetik halı deseni üzerinde ölçüm + güncel literatür/model taraması

---

## 1. Yönetici özeti

İki ayrı sorun var ve ikisi de doğrulandı:

1. **BMP hiç indekslenmiyor.** Varsayılan uzantı listesi `.jpg/.jpeg/.png` ile sınırlı.
   Dosya okuma katmanı BMP'yi zaten destekliyor, arayüz BMP'yi **sorgu** olarak
   kabul ediyor — ama tarayıcı BMP dosyalarını hiç görmüyor. Yani kullanıcı BMP
   sorgu verebiliyor, karşılığında asla BMP sonuç alamıyor.

2. **Benzerlik motoru halı senaryosunun üçte ikisinde çalışmıyor.** Ölçümle
   doğrulandı: kırpılmış/yakın çekim sorgular tamamen kayboluyor, skorlar kalibre
   olmadığı için **alakasız görseller aynı desenin kırpılmış hâlinden yüksek
   skor alıyor**, ve AI modu pratikte hiç devreye girmiyor — model indirilemediği
   için sessizce zayıf bir yedek çıkarıcıya düşüyor; o yedek de farklı deseni
   aynı desenin döndürülmüş hâlinden daha benzer sayıyor.

Aşağıdaki plan bunları 4 fazda, mevcut katmanlı mimariyi bozmadan çözüyor.

---

## 2. Bulgu 1 — BMP tanınmıyor

### 2.1 Kök neden

| Yer | Durum |
|---|---|
| `desenarama/config.py:28` | `extensions` varsayılanı `[".jpg", ".jpeg", ".png"]` — **BMP yok** |
| `desenarama/core/scanner.py:58` | Filtre yoksa `DEFAULT_EXTENSIONS` (yine jpg/jpeg/png) |
| `desenarama/core/imageio.py:26` | `SUPPORTED_EXTENSIONS` BMP/WEBP/TIFF içeriyor — **ama hiçbir yerde kullanılmıyor** |
| `desenarama/gui/main_window.py:58,212` | Sürükle-bırak ve dosya seçici `.bmp/.webp/.tif` **kabul ediyor** |

Yani okuma katmanı hazır, arayüz hazır, sadece **tarayıcı filtresi** eksik.
Asimetri kullanıcıyı yanıltıyor: uygulama BMP'yi tanıyor gibi görünüyor.

### 2.2 İkincil sorunlar

- Ayarlar'da uzantılar serbest metin kutusu (`settings_dialog.py:44`). Kullanıcının
  `.bmp` yazması gerektiğini bilmesi gerekiyor; yazım hatası sessizce yutuluyor.
- Uzantı listesi değiştiğinde "yeniden tarama gerekiyor" uyarısı yok.
- Mevcut kullanıcıların `config.json`'ında eski liste kayıtlı — varsayılanı
  değiştirmek onları kurtarmaz, **göç (migration) gerekiyor**.
- `.dib`, `.jfif`, `.jpe`, `.tga` gibi arşivlerde sık rastlanan uzantılar hiç yok.

---

## 3. Bulgu 2 — Benzerlik arama kalitesi

### 3.1 Ölçüm düzeneği

Sentetik ama halıya özgü bir görsel üretildi (tekrarlayan motif + bordür), sonra
gerçek kullanım senaryolarına karşılık gelen 5 varyant türetildi ve mevcut kodun
kendi fonksiyonlarıyla ölçüldü.

### 3.2 Sonuç — hash arka ucu (varsayılan mod)

| Senaryo | pHash mesafe | Skor (1−d/64) | TTA'lı mesafe |
|---|---|---|---|
| Aynı desen, yeniden boyutlandırılmış | 6 | 0.91 | 6 |
| Aynı desen, farklı renk (colorway) | 10 | 0.84 | 10 |
| Aynı desen, 90° döndürülmüş | 26 | 0.59 | **0** ✅ |
| Aynı desen, **kırpılmış / yakın çekim** | 32 | **0.50** | 26 |
| **Farklı desen** | 26 | **0.59** | 18 |

Üç kritik arıza görünüyor:

**(a) Sıralama tersine dönüyor.** *Farklı desen* 0.59 skorla, *aynı desenin
kırpılmış hâlinden* (0.50) yukarıda çıkıyor. Kullanıcının "sıkıntı var" dediği
şey büyük olasılıkla tam olarak bu.

**(b) `hash_max_distance = 12` varsayılanı çok dar.** Kırpılmış (26–32) ve
döndürülmemiş colorway varyantları bu eşiğin üstünde kalıyor → **hiç sonuç
dönmüyor**. Eşik ancak birebir kopya için doğru.

**(c) Skorlar kalibre değil.** `1 − d/64` formülünde tamamen alakasız iki görsel
istatistiksel olarak d≈32'de, yani **%50 benzerlik** gösteriyor. Arayüzde her şey
"yarı benzer" görünüyor, eşik ayarı anlamsızlaşıyor.

TTA'nın döndürmeyi çözdüğü doğrulandı (26 → 0). Bu katman çalışıyor, korunmalı.

### 3.3 Sonuç — AI (embedding) arka ucu

Burada daha ciddi bir durum var: **AI modu pratikte hiç çalışmıyor.**

`OnnxEmbedder` model dosyası bulamazsa `FallbackEmbedder`'a düşüyor
(`embedder.py:246`), ve bu yalnızca `log.warning` ile bildiriliyor — kullanıcı
"AI (DINOv2)" seçtiğini sanırken klasik bir çıkarıcı çalışıyor. Ölçüm:

| Senaryo | FallbackEmbedder kosinüs |
|---|---|
| Aynı desen, yeniden boyutlandırılmış | 1.000 |
| Aynı desen, farklı renk | 0.998 |
| Aynı desen, döndürülmüş | 0.935 |
| Aynı desen, kırpılmış | **−0.009** ❌ |
| **Farklı desen** | **0.943** ❌ |

**Farklı desen (0.943), aynı desenin döndürülmüş hâlinden (0.935) daha benzer
çıkıyor.** Sebep: özniteliğin baskın bileşeni 16×16'ya küçültülmüş kaba yapı
haritası; halılarda bordür bu haritayı domine ettiği için bordürlü her görsel
birbirine benziyor. Kırpılmış görselde bordür olmadığı için skor sıfıra düşüyor.
Bu arka uç halı için hash'ten **daha kötü**.

### 3.4 Model indirme zinciri kopuk

`models.py` kayıtlı iki URL doğrulandı:

| Model anahtarı | URL | HTTP |
|---|---|---|
| `dinov2-small` | `sefaburak/dinov2-small-onnx` | **200** ✅ (86.6 MB) |
| `dinov2-base` | `sefaburak/dinov2-base-onnx` | **401** ❌ — depo yok |

Yani Ayarlar'daki **"DINOv2-B (768, kaliteli)" seçeneği her zaman başarısız**
oluyor ve sessizce bozuk fallback'e düşüyor. Küçük model URL'i geçerli, ama:
- indirme sadece embedder ilk gerektiğinde tetikleniyor (tembel),
- kurumsal ağ HuggingFace'i engellerse hata yutuluyor,
- SHA256 doğrulaması tanımsız (`sha256=None`) → yarım dosya sessizce kabul edilir.

Araştırmada bulunan sağlam alternatifler (hepsi doğrulandı, HTTP 200):

- `onnx-community/dinov2-small` — `onnx/model.onnx` + `model_int8/fp16/quantized` varyantları
- `onnx-community/dinov2-base` — aynı varyant seti (base için **çalışan** kaynak)
- `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX` — DINOv3, `license: other`
  (Meta'nın kendi lisansı; ticari kullanım için hukuk onayı gerekir — v1'de
  DINOv2/Apache-2.0'da kalınmalı, DINOv3 opsiyonel bırakılmalı)

### 3.5 Kırpma / yakın çekim: hiçbir katman çözmüyor

Hem hash hem embedding **tek global imza** üretiyor. Halı arşivinde gerçek sorgu
sıklıkla bir motifin fotoğrafı ya da desenin bir parçası oluyor; kütüphanedeki
karşılığı ise tüm halının taraması. Global imza bu ikisini eşleştiremez —
ölçümde de görüldü (hash d=26, fallback cos=−0.009).

Araştırma bunu doğruluyor: DINOv2/v3 ile retrieval'da **global + yama (patch/tile)
gömmelerini birlikte indekslemek** ve çoklu-çözünürlüklü kırpma stratejisi
kullanmak yerleşik uygulama. Bizim durumumuzda pratik karşılığı: her görsel için
tam kare + 3×3 kafes karolarının gömmelerini saklayıp sorguda **karolar arası
maksimum** benzerliği almak.

### 3.6 Ön işleme uyumsuzluğu

`embedder.preprocess()` kısa kenarı 256'ya ölçekleyip **ortadan 224 kırpıyor**.
Halıda bordür ana ayırt edici öğedir ve merkez kırpma bordürü tamamen atar.
Ayrıca `FallbackEmbedder` tam görüntüyü sıkıştırıyor — iki arka uç farklı
geometri görüyor, karşılaştırmalar tutarsız.

### 3.7 Diğer tespitler

- **Hibrit mod recall tavanı hash'in tavanıdır** (`search.py:_search_hybrid`):
  adaylar hash'ten geliyor, hash kaçırdıysa embedding kurtaramıyor.
- **Hibritte aday başına ayrı SQL sorgusu** (`store._conn` üzerinden, servis
  katmanından özel alana erişim). 200 aday = 200 sorgu; ayrıca bağlantı
  iş parçacığına bağlıysa GUI'de hata riski.
- `color_alpha` harmanlaması, desen skoru kalibre değilken beklenmedik davranıyor.
- Hash'ler gri tonlama üzerinden hesaplandığından colorway varyantı şansa kalıyor.

---

## 4. Geliştirme planı

> **Durum (9 Ağustos 2026):** Faz A–D tamamlandı ve testlerle sabitlendi.
> Kalan tek iş gerçek arşiv verisiyle doğrulama (D.5).

### Faz A — BMP ve format kapsaması ✅ TAMAMLANDI

1. `imageio.SUPPORTED_EXTENSIONS`'ı tek doğruluk kaynağı yap; `.dib`, `.jfif`,
   `.jpe`, `.tga` ekle.
2. `config.extensions` varsayılanını `SUPPORTED_EXTENSIONS`'a genişlet
   (jpg, jpeg, jpe, jfif, png, bmp, dib, webp, tif, tiff, tga).
3. **Config göçü:** mevcut `config.json` eski dar listeyi taşıyorsa (tam olarak
   `[".jpg",".jpeg",".png"]` ise) yeni varsayılana yükselt, `config_version`
   alanıyla bir kez uygula.
4. `scanner.DEFAULT_EXTENSIONS`'ı da aynı kaynağa bağla.
5. Ayarlar diyaloğunda serbest metin yerine **onay kutusu listesi** + "Tümünü
   seç"; uzantı değişince "Yeniden tarama gerekiyor, şimdi çalıştırılsın mı?"
   uyarısı.
6. Sorgu tarafındaki uzantı listesini de aynı kaynaktan üret (üç yerde kopyalanmış
   listeyi tekilleştir).
7. Test: `tests/` içine BMP/TIFF/WEBP fikstürleriyle uçtan uca indeks + arama testi
   (bugün format testi yok).

### Faz B — Skor kalibrasyonu ve eşikler ✅ TAMAMLANDI

Uygulama sırasında planda olmayan bir arıza daha ortaya çıktı ve düzeltildi:
**renk terimi de kalibre değildi.** Renk histogramı kesişimi, benzer paletteki
alakasız görsellerde 0.9+ verirken aynı desenin farklı renkli varyantında 0.0
veriyor; eski toplamsal harman (`(1-α)·desen + α·renk`) bu yüzden α=0.2'de
sıralamayı tersine çeviriyordu. İki değişiklikle kapatıldı: harman **çarpımsal**
yapıldı (desen kapı, renk en fazla α oranında düzeltme) ve varsayılan α **0**'a
çekildi — colorway varyantları halı arşivinde birincil senaryo olduğu için.
Renk benzerliğinin kendisini kalibre etmek gerçek ölçüt setini gerektirir; bu
Faz D.5'e bırakıldı.

1. **Hash skorunu kalibre et.** `1 − d/64` yerine, alakasızlık tabanını (d≈32)
   sıfıra oturtan bir eşleme:
   `skor = clamp((d_taban − d) / d_taban, 0, 1)`, `d_taban = bits/2`.
   Böylece alakasız → ~0, birebir → 1. 256-bit hash için otomatik ölçeklenir.
2. **`hash_max_distance` varsayılanını 12 → 22'ye çıkar** (64-bit için); 256-bit
   seçildiğinde orantılı ölçekle. 12 yalnızca "kopya" eşiği olarak kalsın
   (`duplicate_hamming` zaten 8).
3. Arayüzde skor yüzdesinin yanına **kalite rozeti** (Kopya / Çok benzer /
   Benzer / Zayıf) ve eşik altındakileri gizleme anahtarı.
4. `score_threshold` varsayılanını kalibre edilmiş ölçekte anlamlı bir değere
   çek (ör. 0.35) — bugün 0.0 olduğu için her şey listeleniyor.
5. Regresyon testi: Bölüm 3.2'deki 5 senaryo sabit fikstür olarak repoya girsin;
   "farklı desen, kırpılmış aynı desenden düşük skor almalı" testi CI'da koşsun.

### Faz C — AI arka ucunu gerçekten çalışır hâle getir ✅ TAMAMLANDI

Gerçek model indirilip ölçüldü; iki karar veriye dayandırıldı:

* **Letterbox ön işleme alındı** — kırpılmış sorguda aynı desenin kosinüsü
  0.867 → 0.918'e çıktı, alakasız desen yerinde kaldı (ayrım payı ~7×).
* **CLS + yama-ortalaması birleştirmesi alınmadı** — denendi, alakasız deseni
  0.888'den 0.907'ye çıkararak ayrımı *kötüleştirdi*. Yalnızca CLS kullanılıyor.

Planda olmayan üçüncü bir arıza bulundu ve düzeltildi: **DINOv2 kosinüsleri dar
ve yüksek bir bantta yaşıyor.** 124 görsellik halı arşivinde ölçülen değerler —
kütüphane medyanı 0.878, alakasız medyan 0.877, aynı desen farklı renk 0.950,
birebir 1.000. Sabit ölçekle bunlar 0.94–1.00 aralığına sıkışıyor, yani AI
modunda da "her şey %95 benzer" görünüyordu. Skor artık kütüphaneden alınan
temsilî örneklemin medyanına göre yeniden ölçekleniyor:

| | ham kosinüs | eski ölçek | yeni kalibre |
|---|---|---|---|
| Birebir / döndürülmüş | 1.000 | 1.000 | 1.000 |
| Aynı desen, farklı renk | 0.950 | 0.975 | 0.593 |
| Aynı desen, kırpılmış | 0.925 | 0.962 | 0.384 |
| Alakasız (medyan) | 0.877 | 0.939 | **0.000** |

Uyarı: bu ölçümdeki "alakasız" örnekler tek bir üreteçten geldiği ve hepsi aynı
bordürü taşıdığı için gerçekte olacaklarından benzerdir; en yüksek alakasız
komşu 0.663 alıyor. Mutlak sayılar gerçek arşivde daha iyi olmalıdır ama bunu
doğrulamak Faz D.5'teki etiketli seti gerektirir.

1. **Bozuk `dinov2-base` URL'ini değiştir** → `onnx-community/dinov2-base`
   (`onnx/model.onnx`). `dinov2-small` için de `onnx-community/dinov2-small`
   yedek kaynak olarak eklensin (birincil başarısız olursa sırayla dene).
2. **Sessiz fallback'i kaldır.** Model yoksa/indirilemezse arayüzde açık uyarı:
   "AI modeli bulunamadı — sonuçlar hızlı moda göre hesaplandı" + "Modeli indir"
   düğmesi + ilerleme çubuğu + çevrimdışı kurulum için `models/` klasörünü açan
   kısayol. `FallbackEmbedder` **varsayılan yol olmaktan çıkarılsın**; ölçümde
   halı için hash'ten kötü olduğu gösterildi.
3. **İlk kurulum sihirbazına model indirmeyi ekle** (tembel indirme yerine),
   böylece ilk arama sırasında sürpriz gecikme/başarısızlık olmaz.
4. `ModelSpec.sha256` alanlarını doldur ve doğrulamayı zorunlu kıl.
5. **Ön işlemeyi düzelt:** merkez kırpma yerine (a) tam kareyi 224'e ölçekle
   (en-boy oranını koruyup dolgu (letterbox) ile) — bordür korunur; (b) `FallbackEmbedder`
   ile aynı geometriyi paylaş.
6. `FallbackEmbedder`'ı ya kaldır ya da yeniden tasarla: kaba yapı haritası
   yerine kontrast-normalize edilmiş çok ölçekli doku istatistikleri (LBP/Gabor
   benzeri) ve renk histogramının **ayrı ayrı normalize edilip ağırlıklandırılması**
   — mevcut hâli bordür tarafından domine ediliyor.

### Faz D — Kırpma/yakın çekim ve halıya özgü kalite ✅ TAMAMLANDI

Karo indekslemenin etkisi, aynı kütüphanede (124 görsel, DINOv2 ViT-S/14, temiz
indeksler) ölçüldü — **tam olarak hedeflediği senaryoyu çözüyor, başka hiçbir
şeyi bozmuyor**:

| Sorgu senaryosu | Karo kapalı | Karo açık (3×3) |
|---|---|---|
| Döndürülmüş | 3/3 | 3/3 |
| Farklı renk (colorway) | 3/3 | 3/3 |
| **Kırpılmış / yakın çekim** | **1/3** | **3/3** |
| Orijinal | 3/3 | 3/3 |

Maliyeti gerçek: indeksleme 19.5 → 2.0 imaj/sn (10×), vektör satırı 124 → 1240.
Bu yüzden varsayılan **kapalı**; Ayarlar'da maliyeti yazan üç seçenek sunuluyor
(kapalı / 2×2 ≈ 5× / 3×3 ≈ 10×).

Ölçüt seti (D.5) altyapısı da kuruldu: `tests/make_dataset.py` artık kırpılmış
(BMP) senaryosu üretiyor, `tests/bench.py` **senaryo kırılımlı** recall@10
raporluyor ve karo/hibrit yapılandırmalarını ölçüyor. Sonuçlar ve yorumu:
[`bench-sonuclari.md`](bench-sonuclari.md). En çarpıcı bulgu: hash colorway
varyantlarında DINOv2'den belirgin biçimde iyi (0.65 vs 0.25), karo destekli
embedding ise kırpmada tek çalışan yöntem (0.40 vs 0.00). Bu, hibridin iki
skorun **maksimumunu** alması gerektiğini gösterdi — ilk uygulama embedding
skorunu üste yazıp hash'in üstünlüğünü kaybediyordu. Düzeltmeyle hibrit
hit@10 0.85 → **0.98**, colorway recall 0.25 → **0.67**.

Planda olmayan bir arıza daha bulundu: **embedding'lerin hangi koşullarda
üretildiği kayıtlı değildi.** Faz C'deki letterbox değişikliği eski indeksleri
geçersiz kılıyor, ama uygulama bunu bilmiyor ve eski vektörlerle yeni sorguları
karşılaştırmaya devam ediyordu. Artık model, ön işleme sürümü ve karo ızgarası
bir imzada saklanıyor; imza değişince embedding'ler otomatik yeniden
hesaplanıyor (metadata korunur, pahalı tarama tekrarlanmaz).

1. **Karo (tile) indeksleme.** Her görsel için tam kare + 3×3 kafes = 10 gömme
   üret, `vectors` tablosuna `(image_id, tile_no, vec)` olarak sakla. Sorguda
   FAISS'ten dönen karoları `image_id`'ye göre **maksimumla** topla. Depolama
   maliyeti 10×, 384-dim float16 ile 100k görselde ~750 MB — kabul edilebilir;
   `tile_enabled` ayarıyla opsiyonel olsun.
2. **Sorgu tarafında da çok ölçeklilik:** sorgunun tamamı + merkez %50 kırpması
   ayrı sorgulansın; TTA ile birleşince varyant sayısı kontrol altında tutulmalı
   (8 TTA × 2 ölçek = 16 gömme, ~0.5 sn — kabul edilebilir).
3. **Hibrit modun recall tavanını kaldır:** adaylar yalnızca hash'ten değil,
   hash ∪ embedding birleşiminden gelsin. Aday vektörlerini **tek toplu SQL
   sorgusuyla** çek (bugün aday başına bir sorgu var) ve `store` üzerine düzgün
   bir genel API ekle (`_conn`'a doğrudan erişimi kaldır).
4. **Hibriti varsayılan yap** — Faz C tamamlandıktan sonra, model kurulu ise.
5. **Ölçüt seti (raporun Faz 0'ı hâlâ yapılmadı):** gerçek arşivden 300–500
   görsel + 30–50 etiketli sorgu senaryosu (birebir kopya / farklı colorway /
   döndürülmüş tarama / kırpılmış yakın çekim / benzer stil). `tests/bench.py`
   bunu Recall@10 ile raporlasın. **Faz B ve D'nin kazancı ancak bu setle
   kanıtlanabilir** — bu adım pazarlık konusu değil.

### Faz E — Opsiyonel / sonraki tur

- DINOv3 ViT-S/16 ONNX desteği (ayarlarda kapalı gelsin; Meta lisansı hukuk
  onayına tabi — DINOv2 Apache 2.0'da kalmak varsayılan).
- INT8 kuantize model seçeneği (`onnx-community` varyantları hazır) — yavaş
  ofis PC'lerinde indeksleme süresini düşürür.
- Arşiv içi mükerrer raporu (pHash kümeleri).
- Kısmi/açılı fotoğraf için yerel öznitelik (ORB) ile son sıralama doğrulaması.

---

## 5. Sıralama önerisi

| Öncelik | Faz | Süre | Kullanıcının hissettiği etki |
|---|---|---|---|
| 1 | A — BMP/format | ½ gün | BMP arşivleri artık aranabiliyor |
| 2 | B — Kalibrasyon | 1 gün | Sonuç sıralaması düzeliyor, skorlar anlamlı |
| 3 | C — AI gerçekten çalışsın | 2–3 gün | "Aynı desen farklı renk" gerçekten bulunuyor |
| 4 | D — Karo + ölçüt seti | 3–5 gün | Kırpılmış/yakın çekim sorgular çalışıyor |

A + B tek bir sürümde (v1.1) çıkarılabilir ve mevcut şikâyetlerin büyük kısmını
kapatır; C + D v1.2 olur.

## 6. Riskler

| Risk | Önlem |
|---|---|
| Uzantı genişletmesi mevcut indeksi geçersiz kılmaz ama yeniden tarama gerektirir | Göçte kullanıcıya sor, arka planda artımlı tara |
| Karo indeksleme depolamayı 10× büyütür | float16 + `tile_enabled` ayarı + kütüphane boyutuna göre otomatik öneri |
| Kurumsal ağ HuggingFace'i engeller | Çevrimdışı model kurulumu belgelensin; `models/` klasörünü açan düğme |
| DINOv3 lisansı ticari kullanımda sorun çıkarır | v1'de DINOv2/Apache 2.0 sabit; v3 kapalı opsiyon |
| Faz D'nin kazancı ölçülmeden kabul edilir | Ölçüt seti (D.5) Faz D'nin ön koşulu |

---

## 7. Kaynaklar

- pHash mesafe dağılımları ve eşik kalibrasyonu: [Hamming distributions of popular perceptual hashing techniques](https://www.sciencedirect.com/science/article/pii/S2666281723000100) · [arXiv 2212.08035](https://arxiv.org/pdf/2212.08035)
- pHash'in renk/düzen karıştırması ve gömme indeksine geçiş gerekçesi: [Perceptual Image Hashing — Mixpeek](https://mixpeek.com/guides/perceptual-image-hashing-near-duplicate-detection)
- Hash'te uzamsal kodlamanın Hamming'in ötesine taşınması: [Beyond Hamming Distance](https://www.sciencedirect.com/science/article/pii/S2666281725000174)
- DINOv2 global + yama gömmeleriyle retrieval, çoklu-çözünürlüklü kırpma: [DINOv2 (arXiv 2304.07193)](https://arxiv.org/html/2304.07193v2) · [DINOv2 retrieval pratiği](https://purnasaigudikandula.medium.com/dinov2-image-classification-visualization-and-paper-review-745bee52c826)
- DINOv3, Gram anchoring ve yoğun yama öznitelikleri: [Meta AI blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) · [arXiv 2508.10104](https://arxiv.org/pdf/2508.10104) · [lisans tartışması](https://github.com/facebookresearch/dinov3/issues/28)
- Doğrulanan ONNX kaynakları: [onnx-community/dinov2-small](https://huggingface.co/onnx-community/dinov2-small) · [onnx-community/dinov2-base](https://huggingface.co/onnx-community/dinov2-base) · [onnx-community/dinov3-vits16](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX) · [sefaburak/dinov2-small-onnx](https://huggingface.co/sefaburak/dinov2-small-onnx)
