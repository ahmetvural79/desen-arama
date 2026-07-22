# Desen Arama — Halı Deseni Görsel Benzerlik Arama

Yerel diskteki veya **ağdaki (SMB/UNC)** bir halı deseni arşivinde, örnek bir
görselle (query-by-example) **aynı veya benzer deseni** bulan, tamamen çevrimdışı
çalışan bir Windows masaüstü uygulaması.

> Tüm işlem yereldedir; hiçbir görsel buluta gönderilmez. Desen arşivleri ticari
> sır niteliğinde olduğundan bu bir tasarım ilkesidir.

Bu proje [`hali-desen-arama-rapor.md`](hali-desen-arama-rapor.md) araştırma
raporundaki mimariyi uygular ve bir adım öteye taşır: **AI'sız hızlı hash modu**
birincil, DINOv2 (AI) opsiyonel katmandır.

---

## Neden iki katman?

"Aynı şekli veya ona çok benzeyeni" ararken pahalı AI etiketlemesi çoğu zaman
gereksizdir. Uygulama üç seçilebilir arama yöntemi sunar:

| Yöntem | Nasıl çalışır | En iyi olduğu senaryo | Hız | Model |
|---|---|---|---|---|
| **Hızlı (hash)** | Algısal hash (pHash/dHash/aHash/wHash) + BK-tree, Hamming mesafesi — czkawka/imagededup tarzı | Aynı / çok benzer şekil, kopyalar | 1000+ imaj/sn indeks | Gerekmez |
| **AI (DINOv2)** | DINOv2 ViT-S/14 embedding (ONNX) + FAISS kosinüs | Farklı renk / döndürülmüş / farklı çekim aynı desen | 20–80 imaj/sn (CPU) | ~90 MB |
| **Hibrit** | Hash ile ucuz aday üret → AI ile yeniden sırala | Hız + derinlik | Orta | ~90 MB |

Ortak kalite hileleri her yöntemde geçerlidir:

- **8×TTA (döndürme dayanıklılığı):** sorgudan 0/90/180/270° × yatay ayna ile 8
  varyant üretilir; döndürülmüş taramalar yakalanır. (Ölçümlerde hash modunda
  hit@1'i 0.04 → 0.50'ye çıkardı.)
- **Renk re-ranking:** "Renk önemi" slider'ıyla `final = (1-α)·desen + α·renk`.
  "Aynı desen farklı renk" ile "aynı renk ailesi" arasında gezinilir.
- **Kopya rozeti:** pHash Hamming mesafesi eşik altındaki sonuçlar "birebir
  kopya" işaretlenir.

---

## Windows & ağ (SMB/UNC) optimizasyonları

Uygulama, kurulduğu cihazın ağ paylaşımlarında çalışacağı varsayımıyla optimize
edildi:

- **Türetilmiş veri daima yereldedir:** SQLite, FAISS, thumbnail ve model
  `%LOCALAPPDATA%\DesenArama` altına yazılır; ağ paylaşımına asla yazılmaz.
- **Tek okuma stratejisi:** her dosya ağdan **bir kez** okunur; embedding, 4
  hash, renk histogramı ve thumbnail aynı bellekteki kopyadan türetilir (ağ
  gidiş-dönüşü minimuma iner).
- **`os.scandir` + DirEntry stat önbelleği:** `listdir`+`stat`'a göre çok daha
  az ağ trafiği.
- **UNC ve uzun yol desteği:** `\\sunucu\paylasim` yolları ve 260 karakteri
  aşan derin ağaçlar için `\\?\` / `\\?\UNC\` ön ekleri otomatik uygulanır.
- **Ağ-farkındalı ayar:** ağ paylaşımı tespit edilince I/O eşzamanlılığı otomatik
  artırılır (gecikmeyi gizlemek için).
- **Dayanıklılık:** erişim reddi / geçici ağ hatası olan dosya/dizin atlanıp
  loglanır; tarama düşmez. Kaybolan dosya **silinmez**, `missing` işaretlenir
  (ağ kopması gerçek silme sanılmaz).
- **Değişiklik izleme:** yerelde native olaylar, **ağda polling** veya periyodik
  yeniden tarama (SMB olayları güvenilmez olduğundan).

---

## Kurulum (geliştirici)

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Kullanım

### Masaüstü arayüz

```bash
python run_gui.py
# veya
python -m desenarama gui
```

Akış: **Kütüphane Klasörü Ekle** → **İndeksle** → sorgu görselini sürükle-bırak
→ **Ara**. Sonuç grid'inde thumbnail + skor + kopya rozeti; çift tık dosyayı
açar, sağ tık "klasörde göster / yolu kopyala".

### Komut satırı

```bash
# Ağ klasörünü hash moduyla indeksle
python -m desenarama index --root "\\sunucu\desenler" --backend hash

# AI moduyla indeksle (ilk çalıştırmada DINOv2 ONNX indirilir)
python -m desenarama index --root "D:\Desenler" --backend embedding

# Sorgu imajıyla ara (renk ağırlığı 0.3)
python -m desenarama search ornek.jpg --k 20 --alpha 0.3

# Durum
python -m desenarama status
```

---

## Mimari

```
desenarama/
  core/            saf Python, GUI'siz, test edilebilir çekirdek
    paths.py       Windows/UNC/uzun-yol + %LOCALAPPDATA% veri dizini
    imageio.py     dayanıklı tek-okuma görsel yükleme + thumbnail
    hasher.py      pHash/dHash/aHash/wHash (çoklu algısal hash)
    hashindex.py   BK-tree + doğrusal Hamming araması
    colorhist.py   HSV renk histogramı (re-ranking)
    embedder.py    ONNX DINOv2 + klasik fallback (model yoksa)
    models.py      DINOv2 ONNX kaydı + otomatik indirme
    vindex.py      FAISS Flat/HNSW soyutlaması
    store.py       SQLite (metadata + vektörler)
    scanner.py     os.scandir + artımlı fark
    osutil.py      dosya aç / klasörde göster
  services/        motor + indeksleyici + arama + izleyici
  gui/             PySide6 arayüz (arka plan iş parçacıklı)
  cli.py           komut satırı
tests/             birim + entegrasyon testleri + Recall@k benchmark
packaging/         PyInstaller spec + Inno Setup betiği
.github/workflows/ Windows exe/kurulum derleme (CI)
```

Katmanlı tasarım sayesinde çekirdek motor, GUI değişse (Plan B: web-UI) veya
sunucu moduna geçilse (Plan C) bile aynı kalır.

## Testler ve benchmark

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q      # 28 test
python tests/make_dataset.py test_dataset                 # sentetik set
DESENARAMA_DATA_DIR=/tmp/b python tests/bench.py --dataset test_dataset --out bench.md
```

## Windows paketi (GitHub Actions)

`main`'e push veya `v*` etiketiyle CI:

1. Linux'ta testleri çalıştırır,
2. Windows'ta PyInstaller (onedir) ile `DesenArama.exe` üretir,
3. Inno Setup ile `DesenAramaSetup-x.y.z.exe` kurulum sihirbazını derler,
4. Artifact olarak yükler; etiket push'unda GitHub Release'e ekler.

Yerelde (Windows) elle:

```powershell
pip install -r requirements-dev.txt
pyinstaller --noconfirm --clean packaging\desenarama.spec
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

## Lisanslar

Uygulama kodu MIT (bkz. [LICENSE](LICENSE)). Bileşenler: DINOv2 **Apache 2.0**,
ONNX Runtime **MIT**, FAISS **MIT**, PySide6/Qt **LGPL** (dinamik bağlama),
Pillow **HPND**, SQLite **kamu malı**. DINOv3 kullanılacaksa Meta'nın özel
lisansı ayrıca incelenmelidir.
