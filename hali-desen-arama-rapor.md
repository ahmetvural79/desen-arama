# Halı Deseni Görsel Benzerlik Arama Uygulaması — Araştırma Raporu ve Yol Haritası

**Tarih:** 22 Temmuz 2026
**Kapsam:** Windows masaüstü, yerel/ağ klasöründe JPEG–PNG halı deseni imajları arasında örnek görselle (query-by-example) benzerlik araması

---

## 1. Yönetici Özeti

Yapılan araştırma sonucunda önerilen çözüm şudur:

**Teknoloji yığını:** Python 3.11+ ve PySide6 (Qt) ile tek dilde, native bir Windows masaüstü uygulaması. Elektron yerine Python tercih edilmelidir; çünkü işin kalbi (embedding çıkarma, vektör arama, görüntü işleme) tamamen Python ekosisteminde yaşamaktadır ve Electron ikinci bir çalışma zamanı (Node.js) ile IPC katmanı ekleyerek karmaşıklığı ve kurulum boyutunu büyütür.

**Ana algoritma:** Meta'nın **DINOv2** (ViT-S/14 veya ViT-B/14) self-supervised görüntü embedding modeli, **ONNX Runtime** ile CPU'da (opsiyonel GPU/DirectML) çalıştırılır. DINOv2, metin etiketine ihtiyaç duymadan saf görsel benzerlik ve ince taneli doku/desen ayrımında CLIP'ten daha iyi performans gösterdiği için halı desenleri gibi "semantik değil görsel" bir problemde en doğru seçimdir. Yanına iki tamamlayıcı katman eklenir: birebir/yakın kopyaları yakalamak için **pHash (algısal hash)** ve renk ağırlığını kullanıcıya ayarlatmak için **HSV renk histogramı** ile yeniden sıralama (re-ranking).

**İndeksleme:** Metadata için **SQLite**, vektörler için başlangıçta brute-force (NumPy / sqlite-vec); kütüphane ~200k imajı aşarsa **FAISS HNSW** indeksine geçiş. Ağ klasörleri için artımlı (incremental) indeksleme, yerel küçük resim (thumbnail) önbelleği ve `watchdog` ile değişiklik izleme.

**Dağıtım:** PyInstaller ile paketleme + Inno Setup ile Windows kurulum sihirbazı. Tüm bileşenler ticari kullanıma uygun lisanslara sahiptir (DINOv2: Apache 2.0, FAISS: MIT, ONNX Runtime: MIT, PySide6: LGPL).

**Süre öngörüsü:** 1–2 geliştirici ile çalışan MVP ~6–8 hafta, paketlenmiş v1.0 ~10–13 hafta.

---

## 2. Problem Tanımı ve Gereksinimler

### 2.1 Fonksiyonel gereksinimler

1. Kullanıcı, yerel diskteki veya ağdaki (SMB/UNC yolu, örn. `\\sunucu\desenler`) bir "kütüphane" klasörü tanımlar; alt klasörler dahil taranır.
2. Uygulama kütüphanedeki tüm JPEG/PNG imajları indeksler (ilk indeksleme + sonradan eklenen/silinen dosyalar için artımlı güncelleme).
3. Kullanıcı bir sorgu imajı verir (dosya seç veya sürükle-bırak); uygulama kütüphanedeki en benzer imajları benzerlik skoruna göre sıralı listeler.
4. Sonuçlarda küçük resim, dosya yolu, skor gösterilir; çift tıkla dosyayı/klasörü açma.

### 2.2 Fonksiyonel olmayan gereksinimler ve alan (domain) özellikleri

- **Veri türü:** Ağırlıklı olarak halı desenleri — yoğun tekrar eden motifler, madalyonlar, bordürler, zemin dokuları. Bu, problemi genel "fotoğraf arama"dan ayırır: nesne semantiği değil, **desen/doku/kompozisyon benzerliği** aranır.
- **Aynı desenin varyantları:** Farklı renk kombinasyonları (colorway), farklı çözünürlük/kırpma, döndürülmüş taramalar, ürün fotoğrafı vs. dijital desen dosyası aynı desene işaret edebilir. Sistem hem "birebir aynı dosya/kopya"yı hem "aynı desen ailesi"ni bulabilmelidir.
- **Ölçek:** Tipik kurumsal desen arşivleri birkaç bin ile birkaç yüz bin imaj arasındadır. Tasarım hedefi: 100k imajda <1 sn sorgu, tek makinede.
- **Çevrimdışılık ve gizlilik:** Desen arşivleri ticari sır niteliğindedir; tüm işlem yerelde, buluta veri gönderilmeden yapılmalıdır.
- **Donanım:** Ortalama ofis PC'si (GPU'suz) baz alınmalı; GPU varsa hızlanma opsiyonel olmalıdır.

---

## 3. Görüntü Benzerliği Algoritmaları — Derin İnceleme

Görsel benzerlik araması (CBIR — Content-Based Image Retrieval) için dört ana yaklaşım ailesi vardır. Hepsi aynı temel şablonu izler: her imajdan sabit boyutlu bir **imza/vektör** çıkar, sorgu imajının imzasını kütüphanedekilerle karşılaştır, mesafeye/benzerliğe göre sırala.

### 3.1 Algısal Hash'ler (aHash, dHash, pHash, wHash)

İmajı küçültüp (örn. 8×8 veya 32×32 gri tonlama) parlaklık ilişkilerinden 64-bit gibi kompakt bir hash üretir; iki hash arasındaki Hamming mesafesi benzerliği verir. `imagededup` kütüphanesi PHash, DHash, WHash (wavelet) ve AHash'in dördünü de hazır sunar; czkawka gibi popüler masaüstü araçlarının "benzer imaj" modu da bu tekniğe dayanır.

- **Güçlü yanları:** Çok hızlı (saniyede binlerce imaj), bellekte imaj başına 8 bayt, yeniden boyutlandırma ve sıkıştırmaya dayanıklı, bağımlılık yükü sıfıra yakın.
- **Zayıf yanları:** Yalnızca *aynı imajın türevlerini* (yeniden kaydedilmiş, küçültülmüş, hafif kırpılmış kopyaları) bulur. "Farklı fotoğraflanmış aynı desen", "farklı renkte aynı motif" veya "benzer stilde başka desen" sorularına **cevap veremez**. Döndürmeye karşı da kırılgandır.
- **Bu projedeki rolü:** Ana arama motoru olamaz; ancak birebir/yakın kopyaları anında yakalayan ucuz bir ön katman ve "arşivdeki mükerrer dosyaları raporla" özelliği için idealdir.

### 3.2 Klasik Bilgisayarla Görü Öznitelikleri

- **Renk histogramları (HSV/Lab):** Küresel renk dağılımını yakalar; desenden bağımsızdır. Tekstil CBIR literatüründe fraktal doku + HSV renk kombinasyonlarının kumaş aramada kullanıldığı çalışmalar vardır. Tek başına zayıf, ama embedding skoruyla harmanlanınca "aynı desen + benzer renk" sıralamasını iyileştirir.
- **Doku betimleyicileri (LBP, Gabor filtre bankaları, GLCM):** Halı gibi dokulu yüzeyler için tarihsel olarak kullanılmış yöntemlerdir; döndürmeye dayanıklı varyantları vardır. Ancak modern derin özniteliklerin doğruluğuna ulaşamazlar; bugün ancak çok kısıtlı donanımda anlamlıdırlar.
- **Yerel öznitelikler (SIFT/ORB + Bag-of-Visual-Words / geometrik doğrulama):** Kısmi eşleme ve perspektif değişimlerine dayanıklıdır; "bir halının köşe fotoğrafından tüm deseni bul" gibi senaryolarda re-ranking aşamasında hâlâ değerlidir. Dezavantajı: imaj başına maliyetli, indeks yapısı karmaşık, tekrar eden motiflerde (halıda çok yaygın) eşleşme belirsizliği yüksek.
- **Bu projedeki rolü:** HSV histogram, opsiyonel renk-ağırlıklı re-ranking için alınır. SIFT/ORB v1 kapsamı dışında tutulur, "kısmi/açılı fotoğraf ile arama" ihtiyacı doğrulanırsa v1.x'te eklenir.

### 3.3 Derin Öğrenme Embedding'leri (günümüzün standardı)

Bir sinir ağının son katmanından alınan 384–1024 boyutlu vektör, imajın "görsel anlamını" temsil eder; kosinüs benzerliği ile karşılaştırılır. Üç ana model ailesi değerlendirilmiştir:

**a) Klasik CNN öznitelikleri (ResNet50, EfficientNet, MobileNet — ImageNet ağırlıklı):**
`imagededup`'ın CNN modu ve birçok GitHub projesi (örn. EfficientNet-B0 + ChromaDB tabanlı yerel ters imaj arama araçları) bu yaklaşımı kullanır. Kurulum kolaydır, CPU'da hızlıdır; ancak ImageNet sınıflandırma için eğitildiklerinden "nesne" odaklıdırlar ve soyut desen/doku ayrımında transformer tabanlı modern modellerin gerisinde kalırlar.

**b) CLIP / SigLIP (dil-görüntü kontrastif modeller — OpenAI/Google):**
CLIP + FAISS ikilisi, imajdan-imaja ve metinden-imaja arama motorları için fiilî standart hâline gelmiştir; çok sayıda açık kaynak örnek proje mevcuttur. Büyük artısı ileride **metinle arama** ("kırmızı madalyonlu klasik desen") kapısını açmasıdır. Ancak CLIP embedding'leri *semantiktir*: aynı kavramı taşıyan görselleri yakınlaştırır, birebir görsel/doku kimliğini garanti etmez — aynı kedinin farklı pozlarını bile uzak düşürebildiği pratikte raporlanmıştır. Ayrıca retrieval karşılaştırmalarında, görsel yapısı baskın sınıflarda (dövme deseni örneğindeki gibi) CLIP'in zayıf, DINOv2'nin güçlü olduğu gözlenmiştir.

**c) DINOv2 / DINOv3 (self-supervised ViT — Meta):**
DINOv2 etiketsiz 142M imajla eğitilmiş bir Vision Transformer'dır; patch-düzeyi hedefleri sayesinde **ince taneli görsel benzerlik, doku ve kenar detaylarını** yakalamada özellikle güçlüdür. Akademik karşılaştırmalar ve saha raporları, saf görüntü-tabanlı retrieval'da DINOv2'nin CLIP'e üstünlüğünü tekrarlar; bir arXiv çalışması aynen "DINOv2 excels at capturing fine-grained visual similarity, making it especially effective for purely image-based retrieval tasks" tespitini yapar. Doku/desen duyarlılığı ölçen bir başka çalışmada DINO ailesinin düşük-seviye görsel örüntüleri (grid, nokta, damalı desenler) temsil uzayında CLIP'ten daha iyi ayrıştırdığı raporlanmıştır — bu, halı deseni problemiyle birebir örtüşen bir bulgudur. Ağustos 2025'te yayımlanan **DINOv3**, 1.7B imaj ve 7B parametrelik öğretmen modelle bu aileyi daha da ileri taşımıştır (Gram anchoring, yüksek çözünürlük adaptasyonu); distile ViT-S/B varyantları mevcuttur. Dikkat: DINOv3 ağırlıkları Apache 2.0 değil, Meta'nın kendi "DINOv3 License" sözleşmesiyle dağıtılır; ticari üründe kullanmadan önce lisans metni hukuk onayından geçirilmelidir. DINOv2 ise Apache 2.0'dır ve ticari kullanımı nettir.

### 3.4 Halı/Tekstil Alanına Özgü Literatür

- Halı benzerlik denetimi üzerine yayımlanmış bir çalışma, Inception-V3 embedding + gözetimsiz kümeleme + genetik algoritma ile halıları benzerlik gruplarına ayırmıştır — derin özniteliklerin halı deseninde çalıştığının doğrudan kanıtı.
- Kumaş imajı retrieval'ı için 25.000 imajlık FIRD veri setiyle "focus ranking" adlı metric-learning yaklaşımı, ince taneli kumaş aramada genel CNN özniteliklerini geçmiştir. Bu, ileride kendi halı verinizle **fine-tuning** yapmanın ölçülebilir kazanç getireceğini gösterir.
- Tekstil arşivlerinde içerik tabanlı arama eksikliği literatürde açıkça belirtilir (mevcut dijital tekstil arşivlerinin çoğu yalnızca anahtar kelime araması sunar) — yani yapacağınız araç, sektörde gerçek bir boşluğu dolduruyor.
- Döndürme problemi: standart CNN/ViT retrieval hatları döndürmeye karşı değişmez (invariant) değildir; sorgu 90° döndürülünce sonuç kalitesinin ciddi düştüğü ölçülmüştür. Pratik ve ucuz çözüm, sorgu anında test-time augmentation'dır (aşağıda 6.5).

### 3.5 Karşılaştırma Tablosu ve Algoritma Kararı

| Kriter | pHash | Renk hist. | SIFT/ORB | CNN (ResNet/EffNet) | CLIP/SigLIP | **DINOv2 (öneri)** |
|---|---|---|---|---|---|---|
| "Aynı desen ailesi" doğruluğu | Zayıf | Zayıf | Orta | İyi | İyi | **Çok iyi** |
| Birebir kopya bulma | **Mükemmel** | Zayıf | İyi | İyi | İyi | Çok iyi |
| Doku/ince detay ayrımı | Zayıf | – | Orta | Orta | Orta | **Çok iyi** |
| Metinle arama potansiyeli | Yok | Yok | Yok | Yok | **Var** | Yok (v3 text-align hariç) |
| CPU hızı (imaj/sn, ~tahmini) | 1000+ | 500+ | 20–50 | 50–150 | 20–60 | 20–80 (ViT-S) |
| Bellek / imaj | 8 B | ~1 KB | 10–50 KB | 2–8 KB | 2–4 KB | 1.5–3 KB |
| Paketleme yükü | Yok | Yok | OpenCV | ONNX/TF | ONNX (~350 MB'a kadar) | ONNX (ViT-S ~90 MB) |
| Lisans | Serbest | Serbest | ORB serbest | Çoğu serbest | MIT/Apache | **Apache 2.0** |

**Karar:** Birincil motor **DINOv2 ViT-S/14 (384-dim)**; kalite yeterli gelmezse aynı hatta tek satır değişiklikle ViT-B/14 (768-dim) yükseltmesi. Yardımcı katmanlar: pHash (kopya tespiti) + HSV histogram (renk ağırlığı slider'ı). CLIP/SigLIP, "metinle arama" özelliği istenirse v1.x'te ikinci indeks olarak eklenir — mimari buna hazır tasarlanacaktır.

---

## 4. Vektör İndeksleme ve Arama Katmanı

Embedding'ler çıkarıldıktan sonra "en yakın k komşu" (kNN) aramasını yapacak bileşen gerekir. Masaüstü uygulama kısıtı nettir: **ayrı sunucu süreci gerektiren çözümler (Milvus, Qdrant server, Weaviate, Elasticsearch) elenir**; gömülü (in-process) çalışan seçenekler değerlendirilir:

| Seçenek | Tür | Artılar | Eksiler | Uygunluk |
|---|---|---|---|---|
| NumPy brute-force | Bellek içi tam arama | Sıfır bağımlılık, %100 doğruluk, 100k×384-dim'de ~on ms'ler | Her açılışta yükleme; milyonlarda yavaşlar | **MVP için ideal** |
| **FAISS (faiss-cpu)** | Kütüphane (Flat/HNSW/IVF-PQ) | Endüstri standardı, ölçeklenir, PQ ile bellek sıkıştırma, Windows wheel'i mevcut | İndeks dosyası + metadata'yı ayrı yönetmek gerekir | **200k+ için birincil** |
| hnswlib / USearch | Hafif ANN kütüphaneleri | Küçük, hızlı, tek header/tek dosya | Filtreleme/metadata yok | FAISS alternatifi |
| sqlite-vec | SQLite eklentisi | Metadata + vektör tek .db dosyasında, SQL ile filtreleme, SIMD'li brute-force | ANN yok (v0.1 itibarıyla tam tarama); çok büyük setlerde yavaş | Basitlik istenirse güçlü aday |
| LanceDB / Chroma (embedded) | Gömülü vektör DB | Zengin API | Bağımlılık ayak izi büyük, masaüstü paketlemede fazlalık | Gerek yok |

**Karar:** Metadata (yol, mtime, boyut, pHash, durum) **SQLite**'ta; vektörler ayrı bir binary blob/`.npy` + FAISS indeksi olarak tutulur. Faz 1–2'de FAISS `IndexFlatIP` (normalize vektörlerle kosinüs = iç çarpım, %100 doğru); kütüphane büyürse `IndexHNSWFlat`'e geçiş tek satırlık konfigürasyondur. Bellek hesabı: 100k imaj × 384 float32 ≈ 150 MB (float16 ile ~75 MB) — ofis PC'sinde sorunsuz.

---

## 5. Mevcut Çözümler Envanteri

### 5.1 Açık kaynak masaüstü uygulamaları

- **czkawka / Krokiet (Rust, MIT):** Windows dahil çok platformlu, çevrimdışı, "Similar Images" modu algısal hash tabanlı. Farklı çözünürlük/isim/sıkıştırmadaki kopyaları bulur; ancak sorgu-imajıyla-arama (query-by-example) iş akışı ve derin embedding yoktur. GTK arayüzü 12.0 ile emekliye ayrılmış, geliştirme Slint tabanlı Krokiet arayüzünde sürmektedir. Ayrıca dokümantasyonu ağ sürücülerinde taramanın belirgin biçimde yavaşladığını, mümkünse yerel kopyada çalışılmasını not eder — bizim ağ-klasörü tasarımımız için önemli bir ders.
- **digiKam (C++/Qt, GPL):** Fotoğraf yöneticisi; "fuzzy/similarity" araması Haar wavelet imzasıyla çalışır, bir referans imaja benzerleri bulabilir. Desen arşivi iş akışına (klasör kütüphanesi + skorlu sonuç listesi + kurumsal kullanım) uyarlanması zahmetlidir; embedding tabanlı değildir.
- **dupeGuru (Python, GPL):** Basit kopya bulucu; tek ayarlı benzer-imaj modu vardır, kalite ve esneklikte czkawka'nın gerisindedir.
- **Immich / PhotoPrism (self-hosted sunucular):** CLIP tabanlı "akıllı arama" sunarlar; ancak Docker/sunucu mimarisi ister, masaüstü kurulum hedefine uymaz. Yine de CLIP+vektör DB mimarilerinin olgun referans uygulamalarıdır.

### 5.2 Python kütüphaneleri (yeniden kullanılabilir yapı taşları)

- **imagededup (idealo, Apache 2.0):** PHash/DHash/WHash/AHash + CNN kodlayıcılarla kopya bulmayı tek pakette çözer; Python 3.9+, Windows destekli. pHash katmanımız için doğrudan ya da referans olarak kullanılabilir.
- **FAISS (Meta, MIT):** Vektör arama standardı.
- **imagehash, opencv-python, Pillow:** Hash, histogram ve I/O için.
- **DeepImageSearch, towhee, clip-retrieval:** Embedding+indeks hattını saran üst-düzey paketler; hızlı PoC için faydalı, üründe ince kontrol için kendi ince hattımız tercih edilir.
- **sefaburakokcu/dinov2_onnx + HuggingFace'teki hazır DINOv2 ONNX çevrimleri:** DINOv2'nin ONNX Runtime ile çalıştırılabildiğinin hazır kanıtı ve başlangıç kodu.

### 5.3 GitHub referans projeleri (mimari ilham)

- **abinthomasonline/clip-faiss, jarvisx17/OpenAI-Clip-Image-Search:** CLIP + FAISS ile indeks kur → sorgula akışının kompakt örnekleri (indeks dosyası + yol eşleme JSON deseni).
- **tikendraw/reverse-image-search:** EfficientNet-B0 + ChromaDB + Streamlit ile tamamen yerel ters imaj arama — "yerel klasörde benzer imaj" fikrinin çalışan kanıtı.
- **DINOv2/CLIP karşılaştırmalı retrieval projeleri ve yazıları:** Model seçim gerekçemizi destekleyen uygulamalı kaynaklar.

### 5.4 Ticari ürünler

- **Visual Similarity Duplicate Image Finder (MindGems):** Windows'ta görsel benzerlikle kopya bulan olgun ticari araç; 300+ format destekler. Kopya temizliğine odaklıdır, kurumsal desen arşivi + sorguyla arama senaryosunu hedeflemez.
- **Textronic Design Archive:** Tekstil sektörüne özel desen kütüphanesi + arama + 3B giydirme sunan sektörel SaaS/kurumsal ürün — pazarın bu ihtiyacı doğruladığının işareti; ancak bulut/tedarikçi bağımlılığı ve maliyet, yerel çözüm gerekçenizi güçlendirir.

### 5.5 Boşluk analizi

Mevcut hiçbir açık kaynak masaüstü araç şu üçünü birlikte sunmuyor: (1) tanımlı yerel/ağ kütüphanesi üzerinde **embedding tabanlı** benzerlik, (2) sorgu-imajıyla arama odaklı basit GUI, (3) Windows'a tek tıkla kurulum. Bu nedenle özel geliştirme doğru karardır; ama pHash, FAISS, ONNX-DINOv2 gibi hazır yapı taşları riski ve süreyi ciddi biçimde düşürür.

---

## 6. Önerilen Mimari ve Teknoloji Yığını

### 6.1 GUI/platform kararı: Python + PySide6 (Electron değil)

| Kriter | **Python + PySide6 (öneri)** | Electron + Python backend | Tauri + Python sidecar |
|---|---|---|---|
| Dil/çalışma zamanı sayısı | 1 (Python) | 2 (Node + Python) + IPC | 2 (Rust/JS + Python) |
| ML ekosistemine erişim | Doğrudan | Köprü üzerinden | Köprü üzerinden |
| Kurulum boyutu (tahmini) | ~250–450 MB (model dahil) | +150–250 MB Chromium | Küçük UI + Python paketi |
| Uzun listeler/thumbnail grid performansı | QListView model/view ile çok iyi | İyi (web grid) | İyi |
| Ekip öğrenme eğrisi | Qt widget'ları | Web bilgisi gerekir | En yüksek |
| Lisans | PySide6 LGPL (ticari serbest) | MIT | MIT/Apache |
| Paketleme olgunluğu | PyInstaller PySide6 ile kutudan çıktığı gibi çalışır; NSIS/Inno/InstallForge akışları belgeli | electron-builder + Python gömme zahmeti | Görece yeni |

Electron yalnızca ekip ağırlıklı web geliştiricilerinden oluşuyorsa ve zengin web-UI şartsa anlamlıdır; o durumda bile Python tarafı FastAPI yerel servisi olarak kalır (bkz. 6.7 B planı). Not: PyQt6 yerine **PySide6** seçilmelidir — işlevsel olarak eşdeğerdir ama PySide6 LGPL'dir, PyQt ticari kullanımda GPL/ücretli lisans ister.

### 6.2 Katmanlı mimari

```
┌──────────────────────────────────────────────────────────┐
│  GUI (PySide6)                                           │
│  Kütüphane ayarları · İndeks durumu · Sürükle-bırak      │
│  sorgu · Sonuç grid'i (thumbnail+skor) · Renk slider'ı   │
├──────────────────────────────────────────────────────────┤
│  Uygulama Servisleri (QThread / ProcessPool)             │
│  IndexerService        │  SearchService                  │
│  - klasör tarama       │  - sorgu embed (+TTA)           │
│  - artımlı fark        │  - FAISS kNN                    │
│  - batch embedding     │  - renk re-ranking              │
│  - thumbnail üretimi   │  - sonuç modeli                 │
├──────────────────────────────────────────────────────────┤
│  Çekirdek Motor (saf Python, GUI'den bağımsız = test edilebilir) │
│  embedder.py (ONNX Runtime · DINOv2 ViT-S/14)            │
│  hasher.py (pHash) · colorhist.py (HSV)                  │
│  store.py (SQLite) · vindex.py (FAISS Flat/HNSW)         │
│  scanner.py (os.scandir + watchdog)                      │
├──────────────────────────────────────────────────────────┤
│  Depolama (%LOCALAPPDATA%\DesenArama\)                   │
│  library.db (SQLite) · vectors.faiss · thumbs\ · models\ │
└──────────────────────────────────────────────────────────┘
```

### 6.3 Veri modeli (SQLite)

```sql
CREATE TABLE images(
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,            -- UNC dahil tam yol
  mtime REAL, size INTEGER,    -- artımlı indeks anahtarı
  phash BLOB,                  -- 64-bit algısal hash
  vec_row INTEGER,             -- FAISS satır no
  width INT, height INT,
  status TEXT,                 -- ok | missing | error
  indexed_at REAL
);
CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
```

### 6.4 İndeksleme akışı

1. `os.scandir` ile özyinelemeli tarama (uzantı filtresi: jpg/jpeg/png; ayarlardan genişletilebilir: webp/tif/bmp).
2. DB ile fark: yeni dosya → kuyruk; `mtime/size` değişen → yeniden işle; kayıp → `missing` işaretle (ağ kopması yüzünden silme yapma — sadece kullanıcı onayıyla temizle).
3. İşleme hattı (üretici-tüketici): **I/O havuzu** dosyayı okur + 224×224 ön-işler + 256px thumbnail'i yerel önbelleğe yazar → **CPU havuzu** 16–32'lik batch'lerle ONNX embedding çıkarır → normalize edip FAISS'e ekler, pHash'i hesaplar, DB'ye yazar.
4. Ağ klasörü önlemleri: küçük resim ve vektörler daima yerelde; okuma hataları yeniden dene + atla-ve-logla; büyük taramada ilerleme ve "duraklat/devam".
5. `watchdog` ile çalışma sırasında canlı değişiklik izleme (ağ paylaşımlarında event güvenilmezse periyodik yeniden tarama zamanlayıcısı).

### 6.5 Sorgu akışı ve halıya özel kalite hileleri

1. Sorgu imajı → aynı ön-işleme → DINOv2 embedding.
2. **Döndürme dayanıklılığı (TTA):** Sorgudan 8 varyant üret (0/90/180/270° × yatay ayna), 8 embedding ile ara, imaj başına en yüksek skoru al. Literatür, CNN/ViT retrieval'ının döndürmeye duyarlı olduğunu ve bunun sonucu ciddi bozduğunu gösteriyor; 8×TTA bunu ~%1'lik ek sorgu maliyetiyle çözer (8 embedding ≈ 0.2–0.5 sn CPU).
3. FAISS `IndexFlatIP` ile top-200 aday → **renk re-ranking:** `final = (1-α)·cos_embed + α·hist_benzerliği`; α GUI'de "Renk önemi" slider'ı (varsayılan 0.2). Böylece "aynı desen farklı renk" ile "aynı renk ailesi" arasında kullanıcı gezinebilir.
4. pHash mesafesi ≤ 8 olan sonuçlar "birebir kopya" rozetiyle işaretlenir.
5. Eşik altı sonuçlar gri gösterilir; FAISS her zaman k sonuç döndürdüğü için "gerçekten benzer yok" durumu skor eşiğiyle kullanıcıya dürüstçe yansıtılmalıdır.

### 6.6 Performans öngörüleri (8 çekirdekli, GPU'suz ofis PC — doğrulanacak tahminler)

- İlk indeksleme, ViT-S/14 ONNX INT8/FP32 batch: ~25–60 imaj/sn ⇒ 50.000 imaj ≈ 15–35 dk (ağ diskinde I/O darboğazıyla daha uzun; thumbnail üretimi dahil).
- Sorgu: 8×TTA embedding ~0.3 sn + 100k üzerinde Flat kNN ~30–60 ms + re-ranking ⇒ toplam **< 1 sn**.
- Disk: thumbnail önbelleği ~10–20 KB/imaj (50k imaj ≈ 0.5–1 GB), vektörler ~150 MB/100k (fp32).
- GPU opsiyonu: `onnxruntime-directml` ile herhangi bir DX12 GPU'da 5–15× embedding hızlanması; kod değişmeden provider seçimiyle.

### 6.7 Alternatif planlar

- **Plan B (web-UI istenirse):** Çekirdek motor aynı kalır; PySide6 yerine yerel FastAPI + tarayıcı/pywebview kabuğu ya da Electron ön yüzü. Çekirdeği GUI'den bağımsız yazmak bu geçişi ucuzlatır (mimari bunun için katmanlı kuruldu).
- **Plan C (çok kullanıcılı gelecek):** Aynı motor, şirket sunucusunda FastAPI + Qdrant ile servisleşir; masaüstü uygulama istemciye dönüşür. v1 mimarisinde `SearchService` arayüzü bu soyutlamayı şimdiden taşır.

### 6.8 Lisans özeti

DINOv2 (Apache 2.0) · ONNX Runtime (MIT) · FAISS (MIT) · PySide6/Qt (LGPL — dinamik bağlama koşuluyla ticari serbest) · Pillow (HPND) · imagehash (BSD) · SQLite (public domain) · PyInstaller (GPL + istisna: üretilen exe serbest). DINOv3 kullanılacaksa özel Meta lisansı hukuken incelenmeli; CLIP eklenirse MIT.

---

## 7. Geliştirme Yol Haritası

Varsayım: 1–2 geliştirici. Süreler takvim haftasıdır; tek kişilik ekipte üst sınırlar geçerlidir.

### Faz 0 — Hazırlık ve Ölçüt Tanımı (1 hafta)

Gerçek arşivden 500–2.000 imajlık temsilî bir test seti derlenir; içinden 30–50 sorgu senaryosu (birebir kopya, farklı renk aynı desen, döndürülmüş tarama, "benzer stil") elle etiketlenir. Başarı ölçütü tanımlanır: **Recall@10 ≥ %90** (doğru desen ilk 10 sonuçta) ve kullanıcı gözüyle "ilk ekran anlamlı" değerlendirmesi. Geliştirme ortamı, repo, CI iskeleti kurulur.
**Çıktı:** test seti + ölçüm scripti + proje iskeleti.

### Faz 1 — Algoritma PoC ve Karar Kapısı (1–2 hafta)

Komut satırı prototipi: klasörü tara → embedding çıkar → FAISS'e yaz → sorgu imajıyla top-k listele. Aynı hat üzerinde **DINOv2-S vs DINOv2-B vs CLIP ViT-B/32 vs pHash** dörtlüsü test setinde Recall@10 ile karşılaştırılır; 8×TTA'nın döndürülmüş sorgulardaki katkısı ölçülür; INT8 kuantizasyonun hız/doğruluk etkisi bakılır.
**Karar kapısı:** Model ve boyut seçimi verilere dayanarak kilitlenir. (Beklenti: DINOv2-S yeterli; değilse B.)
**Çıktı:** `bench.md` sonuç tablosu + seçilmiş ONNX model dosyası.

### Faz 2 — İndeksleme Motoru (2–3 hafta)

SQLite şeması, artımlı tarama (mtime/size), üretici-tüketici batch hattı, thumbnail önbelleği, pHash, hata toleransı (bozuk dosya, erişim reddi, ağ kopması), duraklat/devam, `watchdog` entegrasyonu, birim testleri. 50k imajlık yapay setle ve gerçek ağ paylaşımında dayanıklılık testi.
**Çıktı:** GUI'siz çalışan, testli `engine/` paketi + CLI.

### Faz 3 — Masaüstü Arayüz (2–3 hafta)

PySide6: ilk kurulum sihirbazı (kütüphane klasörü seç → indeksle), ana pencere (sürükle-bırak sorgu alanı, sonuç grid'i: thumbnail + skor + yol, sağ tık: dosyayı aç / klasörde göster / kopyala), ayarlar (klasörler, uzantılar, renk-önemi varsayılanı, eşik), durum çubuğunda indeks ilerlemesi, arka plan iş parçacıkları ile donmayan UI, TR arayüz metinleri (i18n altyapısıyla).
**Çıktı:** uçtan uca çalışan uygulama (geliştirici ortamında).

### Faz 4 — Paketleme ve Kurulum (1–2 hafta)

PyInstaller onedir yapılandırması (model dosyaları `models/` içinde), Inno Setup ile kurulum sihirbazı, ilk-açılış model doğrulaması, sürüm numaralama, opsiyonel kod imzalama, çökme günlükleri (%LOCALAPPDATA%\...\logs). Temiz Windows 10/11 VM'lerinde kurulum testi; Defender/SmartScreen davranışı kontrolü.
**Çıktı:** `DesenAramaSetup-1.0.0.exe`.

### Faz 5 — Pilot ve Ayar (1–2 hafta)

Gerçek arşivde 2–5 son kullanıcıyla pilot: eşik/α varsayılanları, ağ performansı, UI sürtünmeleri. Geri bildirimle düzeltme turu; kabul: Faz 0 ölçütleri gerçek arşivde tutuyor.
**Çıktı:** v1.0 yayını + kısa kullanım kılavuzu.

### Faz 6 — v1.x Geliştirmeleri (talebe göre, artımlı)

Öncelik sırası önerisi: (1) GPU/DirectML anahtarı, (2) arşiv içi mükerrer raporu (pHash kümeleri), (3) SigLIP ikinci indeksiyle **metinle arama**, (4) kendi verinizle fine-tuning — etiketli "aynı desen" çiftleriyle triplet/focus-ranking eğitimi literatürde ölçülebilir kazanç göstermiştir, (5) kısmi/açılı fotoğraf sorguları için yerel öznitelik re-ranking, (6) çok-kullanıcılı sunucu modu (Plan C).

**Toplam:** MVP (Faz 0–3) ≈ 6–9 hafta · v1.0 (Faz 0–5) ≈ 8–13 hafta.

---

## 8. Riskler ve Önlemler

| Risk | Olasılık/Etki | Önlem |
|---|---|---|
| Ağ (SMB/NAS) taraması yavaş | Yüksek/Orta | Thumbnail+vektör daima yerel; batch okuma; ilk indeksin gece çalışması; czkawka'nın da belgelediği gibi mümkünse yerel kopya senaryosu sunmak |
| CPU'da indeksleme süresi beklentiyi aşar | Orta/Orta | ViT-S + INT8; batch; DirectML opsiyonu; ilerleme + duraklat/devam ile algıyı yönetmek |
| "Benzerlik" tanımında kullanıcı beklentisi farklı (renk vs desen) | Orta/Yüksek | Renk-önemi slider'ı; Faz 0'da etiketli senaryolarla beklentiyi ölçüp varsayılanları veriye göre seçmek |
| Döndürülmüş/aynalanmış taramalar kaçar | Orta/Orta | 8×TTA (6.5); gerekirse indeks tarafında da 0/90° çift kayıt |
| Bozuk/dev boyutlu/egzotik dosyalar hattı düşürür | Orta/Düşük | Her dosya try/except + hata tablosu; boyut üst sınırı ve akıllı downscale; Pillow `MAX_IMAGE_PIXELS` yönetimi |
| PyInstaller çıktısında AV/SmartScreen uyarısı | Orta/Düşük | Kod imzalama sertifikası; onedir modu; kurulumda istisna dokümantasyonu |
| DINOv3'e geçme isteği lisans engeline takılır | Düşük/Orta | v1 DINOv2 (Apache 2.0) ile kilitli; v3 ancak hukuk onayıyla |
| Kütüphane 1M+ imaja büyür | Düşük/Orta | FAISS HNSW/IVF-PQ'ya geçiş; fp16/PQ sıkıştırma; mimari bunu tek modülde soyutlar |

---

## 9. Sonuç

Halı deseni arama problemi, 2026 itibarıyla çözülmüş yapı taşlarının doğru dizilimidir: **DINOv2 embedding (ONNX, yerel) + FAISS + SQLite + PySide6 + PyInstaller**. Bu dizilim (a) buluta veri çıkarmaz, (b) GPU'suz ofis PC'sinde saniye altı arama verir, (c) tamamı ticari-dostu lisanslıdır, (d) metinle arama, fine-tuning ve sunucu moduna evrilebilecek katmanlı bir mimari kurar. Kritik başarı faktörü teknoloji değil, **Faz 0'daki gerçek-veri test seti**dir: model ve eşik kararlarını bu set üzerinden vererek "kâğıt üstünde iyi, arşivde vasat" tuzağından kaçınılır.

---

## 10. Kaynaklar

**Model ve algoritma**
- DINOv2 (Meta, Apache 2.0): https://github.com/facebookresearch/dinov2
- DINOv3 makale/inceleme: https://arxiv.org/abs/2508.10104 · https://www.lightly.ai/blog/dinov3
- DINOv2'nin ince taneli görsel retrieval üstünlüğü: https://arxiv.org/pdf/2603.24480 · https://arxiv.org/pdf/2407.00592
- DINO ailesinin desen/doku duyarlılığı: https://arxiv.org/pdf/2510.11835
- CLIP vs DINOv2 uygulamalı karşılaştırmalar: https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6 · https://ai.gopubby.com/clip-vs-dinov2-which-one-is-better-for-image-retrieval-d68c03f51f0d
- CLIP+FAISS eğitimleri: https://blog.roboflow.com/clip-image-search-faiss/ · https://towardsdatascience.com/building-an-image-similarity-search-engine-with-faiss-and-clip-2211126d08fa/
- Döndürmeye duyarlılık ve çözümleri: https://arxiv.org/pdf/2006.13046

**Tekstil/halı alan literatürü**
- Halı benzerlik denetimi (Inception-V3 + kümeleme): https://digitalcommons.kennesaw.edu/facpubs/5486/
- Kumaş retrieval, FIRD veri seti ve focus ranking: https://arxiv.org/pdf/1712.10211
- Tekstil arşivlerinde içerik tabanlı arama boşluğu: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7936973/
- Halı doku/kalite veri seti örneği: https://arxiv.org/html/2407.18450v1

**Kütüphaneler ve araçlar**
- imagededup (idealo): https://github.com/idealo/imagededup
- FAISS: https://github.com/facebookresearch/faiss
- sqlite-vec ve gömülü vektör kıyasları: https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html
- ONNX Runtime: https://onnxruntime.ai/docs/get-started/with-python.html
- DINOv2 ONNX örnekleri: https://github.com/sefaburakokcu/dinov2_onnx · https://huggingface.co/sefaburak/dinov2-small-onnx
- PySide6 + PyInstaller paketleme rehberi: https://www.pythonguis.com/tutorials/packaging-pyside6-applications-windows-pyinstaller-installforge/

**Mevcut uygulamalar**
- czkawka/Krokiet: https://github.com/qarmin/czkawka · https://czkawka.net/
- Yerel ters imaj arama örneği (EfficientNet+Chroma): https://github.com/tikendraw/reverse-image-search
- CLIP-FAISS mini uygulama: https://github.com/abinthomasonline/clip-faiss
- Ticari örnekler: Visual Similarity Duplicate Image Finder (MindGems) · Textronic Design Archive: https://www.textronic.com/archive.html
