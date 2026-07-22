"""Algısal hash'ler — czkawka / imagededup tarzı "aynı/çok benzer şekil" arama.

Bu katman uygulamanın **hızlı, AI'sız birincil arama motoru**dur. Bir imajı
küçültüp parlaklık ilişkilerinden kompakt bir hash üretir; iki hash arasındaki
**Hamming mesafesi** benzerliği verir. Yeniden boyutlandırma ve sıkıştırmaya
dayanıklıdır, imaj başına yalnızca birkaç bayt yer kaplar ve saniyede binlerce
imaj işlenebilir — ağ arşivleri için idealdir.

Dört algoritma sunulur (imagededup ile aynı aile):

* ``ahash`` — ortalama hash (average)
* ``dhash`` — fark hash (difference); yatay gradyan
* ``phash`` — DCT tabanlı algısal hash (en dengeli; **varsayılan**)
* ``whash`` — Haar dalgacık (wavelet) hash

``hash_size`` büyütülerek (8 → 16) czkawka'nın büyük hash boyutlarındaki gibi
daha ince ayrım elde edilebilir (64-bit → 256-bit).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dct

HashAlgo = str  # "phash" | "dhash" | "ahash" | "whash"
ALGOS: tuple[str, ...] = ("phash", "dhash", "ahash", "whash")
DEFAULT_ALGO = "phash"


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _grayscale(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """RGB diziyi verilen boyuta yüksek kaliteli küçülterek gri float dizi verir."""
    im = Image.fromarray(rgb).convert("L").resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def _bits_to_int(bits: np.ndarray) -> int:
    """Boolean bit dizisini (satır-öncelikli) tek bir tamsayıya paketler."""
    val = 0
    for b in bits.flatten():
        val = (val << 1) | int(b)
    return val


# --------------------------------------------------------------------------- #
# Algoritmalar (hepsi hash_size*hash_size bitlik bir tamsayı döndürür)
# --------------------------------------------------------------------------- #
def ahash(rgb: np.ndarray, hash_size: int = 8) -> int:
    px = _grayscale(rgb, (hash_size, hash_size))
    return _bits_to_int(px > px.mean())


def dhash(rgb: np.ndarray, hash_size: int = 8) -> int:
    # Yatay komşu farkı için bir sütun fazla örnekle.
    px = _grayscale(rgb, (hash_size + 1, hash_size))
    return _bits_to_int(px[:, 1:] > px[:, :-1])


def phash(rgb: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    img_size = hash_size * highfreq_factor
    px = _grayscale(rgb, (img_size, img_size))
    coef = dct(dct(px, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coef[:hash_size, :hash_size]
    med = np.median(low)
    return _bits_to_int(low > med)


def _haar2d(px: np.ndarray, level: int) -> np.ndarray:
    """Basit çok seviyeli 2B Haar dönüşümü (yalnızca yaklaşım/detay ayrıştırması)."""
    a = px.copy()
    n = px.shape[0]
    size = n
    for _ in range(level):
        half = size // 2
        # Satırlar
        tmp = np.empty_like(a[:size, :size])
        tmp[:, :half] = (a[:size, 0:size:2] + a[:size, 1:size:2]) / 2.0
        tmp[:, half:size] = (a[:size, 0:size:2] - a[:size, 1:size:2]) / 2.0
        # Sütunlar
        out = np.empty_like(tmp)
        out[:half, :] = (tmp[0:size:2, :] + tmp[1:size:2, :]) / 2.0
        out[half:size, :] = (tmp[0:size:2, :] - tmp[1:size:2, :]) / 2.0
        a[:size, :size] = out
        size = half
    return a


def whash(rgb: np.ndarray, hash_size: int = 8) -> int:
    # En yakın 2'nin kuvvetine göre görüntü boyutu seç (Haar için gereklidir).
    img_size = max(hash_size, 1)
    # hash_size 2'nin kuvveti değilse bir üst kuvvete yuvarla.
    p = 1
    while p < img_size:
        p <<= 1
    img_size = p * 4  # daha fazla frekans içeriği
    level = int(np.log2(img_size // hash_size))
    px = _grayscale(rgb, (img_size, img_size))
    px = px / 255.0
    coef = _haar2d(px, level)
    low = coef[:hash_size, :hash_size]
    med = np.median(low)
    return _bits_to_int(low > med)


_FUNCS = {"ahash": ahash, "dhash": dhash, "phash": phash, "whash": whash}


def compute(rgb: np.ndarray, algo: HashAlgo = DEFAULT_ALGO, hash_size: int = 8) -> int:
    fn = _FUNCS.get(algo)
    if fn is None:
        raise ValueError(f"Bilinmeyen hash algoritması: {algo}")
    return fn(rgb, hash_size)


# --------------------------------------------------------------------------- #
# Mesafe + serileştirme
# --------------------------------------------------------------------------- #
def hamming(a: int, b: int) -> int:
    """İki hash tamsayısı arasındaki Hamming mesafesi (farklı bit sayısı)."""
    return int(a ^ b).bit_count()


def to_blob(value: int, hash_size: int = 8) -> bytes:
    """Hash tamsayısını SQLite BLOB olarak sabit uzunlukta serileştirir."""
    nbytes = (hash_size * hash_size + 7) // 8
    return int(value).to_bytes(nbytes, "big")


def from_blob(blob: bytes) -> int:
    return int.from_bytes(blob, "big")


def max_bits(hash_size: int = 8) -> int:
    """Bir hash'teki toplam bit sayısı (mesafeyi benzerlik yüzdesine çevirmek için)."""
    return hash_size * hash_size
