"""HSV renk histogramı — renk-ağırlıklı yeniden sıralama (re-ranking) için.

Renk histogramı desenden bağımsız olarak küresel renk dağılımını yakalar.
Tek başına zayıftır; ancak hash/embedding skoruyla harmanlanınca "aynı desen
farklı renk" ile "aynı renk ailesi" arasında kullanıcının kaydırıcıyla
gezinmesini sağlar (bkz. SearchService renk-önemi α parametresi).
"""

from __future__ import annotations

import cv2
import numpy as np

# H:8, S:3, V:3 kovaları — 72 boyutlu kompakt, aydınlatmaya makul dayanıklı.
H_BINS, S_BINS, V_BINS = 8, 3, 3
DIM = H_BINS * S_BINS * V_BINS


def histogram(rgb: np.ndarray) -> np.ndarray:
    """RGB diziden L1-normalize edilmiş HSV histogramı (float32, DIM boyut)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki histogram arasında 0..1 benzerlik (histogram kesişimi)."""
    if a is None or b is None:
        return 0.0
    # Kesişim: min'lerin toplamı; her ikisi L1-normalize olduğundan 0..1.
    return float(np.minimum(a, b).sum())


def to_blob(hist: np.ndarray) -> bytes:
    return np.asarray(hist, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
