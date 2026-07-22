"""Dayanıklı görsel okuma — ağ paylaşımları için tek-okuma stratejisi.

Ağ (SMB/UNC) klasörlerinde her dosya erişimi bir ağ gidiş-dönüşüdür.
Bu yüzden bir dosyayı **bir kez** bayt olarak okuyup, hem embedding hem
pHash hem renk histogramı hem thumbnail'i aynı bellekteki kopyadan
türetiriz. Modül ayrıca bozuk/dev/egzotik dosyalara karşı korumalıdır.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile

from . import paths

# Kısmen bozuk JPEG'lerin hattı düşürmemesi için toleranslı çözme.
ImageFile.LOAD_TRUNCATED_IMAGES = True
# "Decompression bomb" korumasını makul bir üst sınıra çekeriz (ör. 200 MP).
# None yaparsak dev dosyalar belleği tüketebilir; çok küçük yaparsak meşru
# yüksek çözünürlüklü taramalar reddedilir.
Image.MAX_IMAGE_PIXELS = 200_000_000

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageLoadError(Exception):
    """Dosya okunamadı / çözülemedi (bozuk, erişim reddi, desteklenmeyen)."""


@dataclass
class LoadedImage:
    """Bir dosyadan tek okumayla türetilen ham veriler."""

    rgb: np.ndarray  # (H, W, 3) uint8, RGB
    width: int
    height: int
    size_bytes: int


def read_bytes(path: str, retries: int = 2) -> bytes:
    """Dosyayı ikili okur; geçici ağ hatalarında yeniden dener.

    Windows'ta uzun/UNC yollar için ``\\\\?\\`` ön eki otomatik uygulanır.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with paths.open_binary(path) as f:
                return f.read()
        except (OSError, IOError) as e:  # ağ kopması, erişim reddi vb.
            last_err = e
    raise ImageLoadError(f"Okunamadı: {path}: {last_err}")


def load_from_bytes(data: bytes) -> LoadedImage:
    """Bellekteki baytlardan RGB numpy dizisi üretir."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            h, w = arr.shape[0], arr.shape[1]
            return LoadedImage(rgb=arr, width=w, height=h, size_bytes=len(data))
    except Exception as e:  # PIL geniş bir hata yelpazesi fırlatabilir
        raise ImageLoadError(f"Çözülemedi: {e}") from e


def load(path: str) -> LoadedImage:
    """Yoldan tek okumayla :class:`LoadedImage` döndürür."""
    return load_from_bytes(read_bytes(path))


def make_thumbnail(rgb: np.ndarray, max_side: int = 256, quality: int = 82) -> bytes:
    """RGB diziden JPEG thumbnail baytları üretir (yerel önbelleğe yazmak için)."""
    im = Image.fromarray(rgb)
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    # Şeffaf PNG'ler RGB'ye çevrildiği için JPEG güvenli ve küçüktür.
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
