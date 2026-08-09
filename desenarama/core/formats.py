"""Desteklenen görsel formatları — **tek doğruluk kaynağı**.

Uzantı listesi daha önce üç ayrı yerde (config varsayılanı, tarayıcı filtresi,
arayüz dosya seçici) kopyalanmıştı ve birbirinden ayrışmıştı: dosya seçici BMP'yi
kabul ediyor, tarayıcı ise BMP dosyalarını hiç görmüyordu. Artık hepsi bu modülü
kullanır.

Buradaki her uzantı Pillow tarafından okunabilir olmalıdır; listeye ekleme
yapmadan önce ``tests/test_formats.py`` gerçek dosyayla doğrular.

Bu modül bilerek **ağır bağımlılık içermez** (numpy/PIL/cv2 yok), böylece
yapılandırma katmanı görüntü yığınını çekmeden uzantıları öğrenebilir.
"""

from __future__ import annotations

# Format ailesi -> uzantılar. Sıra arayüzde gösterim sırasıdır.
FORMAT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("JPEG", (".jpg", ".jpeg", ".jpe", ".jfif")),
    ("PNG", (".png",)),
    ("BMP", (".bmp", ".dib")),
    ("WebP", (".webp",)),
    ("TIFF", (".tif", ".tiff")),
    ("TGA", (".tga",)),
)

#: Uygulamanın okuyabildiği tüm uzantılar (sıralı).
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(
    ext for _, exts in FORMAT_FAMILIES for ext in exts
)

#: Yeni kurulumların varsayılanı — desteklenen her şey.
#:
#: Halı arşivlerinde tarama çıktıları sıklıkla BMP/TIFF olarak saklandığından
#: varsayılanı dar tutmak (eski davranış: yalnızca jpg/jpeg/png) arşivin büyük
#: kısmının sessizce indekslenmemesine yol açıyordu.
DEFAULT_EXTENSIONS: tuple[str, ...] = SUPPORTED_EXTENSIONS

#: v1.0'ın dar varsayılanı. Göç (migration) bunu tanıyıp yükseltmek için kullanır.
LEGACY_DEFAULT_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


def normalize(ext: str) -> str:
    """``JPG`` / ``.JPG`` / ``jpg`` → ``.jpg``."""
    e = ext.strip().lower()
    if not e:
        return ""
    return e if e.startswith(".") else "." + e


def normalize_all(exts) -> set[str]:
    """Bir uzantı dizisini normalize edip boşları eler."""
    return {n for n in (normalize(e) for e in exts) if n}


def is_supported(ext: str) -> bool:
    return normalize(ext) in SUPPORTED_EXTENSIONS


def qt_name_filter(label: str = "Görseller") -> str:
    """Qt dosya seçici için filtre dizesi (``"Görseller (*.jpg *.png ...)"``)."""
    patterns = " ".join("*" + e for e in SUPPORTED_EXTENSIONS)
    return f"{label} ({patterns})"
