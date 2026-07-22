"""FAISS tabanlı vektör indeksi — AI (embedding) arama arka ucu için.

Kütüphane küçükken ``IndexFlatIP`` (normalize vektörlerle kosinüs = iç çarpım,
%100 doğru) kullanılır; koleksiyon büyürse (varsayılan > 200k) tek satırlık
konfigürasyonla ``IndexHNSWFlat``'e geçilir. İndeks diske FAISS'in kendi
biçiminde, satır→imaj eşlemesi ise SQLite'ta (vec_row) tutulur.

Vektörler yerel diske yazılır; ağ paylaşımına asla yazılmaz.
"""

from __future__ import annotations

import os

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int, hnsw: bool = False, hnsw_m: int = 32) -> None:
        self.dim = dim
        self.hnsw = hnsw
        if hnsw:
            index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
            self.index = index
        else:
            self.index = faiss.IndexFlatIP(dim)

    # -- yazma -------------------------------------------------------------- #
    @staticmethod
    def normalize(vecs: np.ndarray) -> np.ndarray:
        """Kosinüs benzerliği için L2 normalize (satır bazında), float32."""
        v = np.ascontiguousarray(vecs, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        faiss.normalize_L2(v)
        return v

    def add(self, vecs: np.ndarray) -> int:
        """Normalize edilmiş vektörleri ekler; eklemeden önceki satır sayısını döndürür.

        Dönen değer, eklenen ilk vektörün FAISS satır numarasıdır (vec_row).
        """
        start = self.index.ntotal
        self.index.add(self.normalize(vecs))
        return start

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    # -- arama -------------------------------------------------------------- #
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Kosinüs benzerliğine göre en yakın k; (skorlar, satır_no) döndürür.

        Skorlar iç çarpımdır: normalize vektörlerde -1..1, benzerlerde ~1.
        """
        q = self.normalize(query)
        k = min(k, max(self.index.ntotal, 1))
        scores, ids = self.index.search(q, k)
        return scores, ids

    # -- kalıcılık ---------------------------------------------------------- #
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        faiss.write_index(self.index, tmp)
        os.replace(tmp, path)  # atomik değiştirme (yarım yazma bırakmaz)

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorIndex":
        index = faiss.read_index(path)
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.index = index
        obj.hnsw = not isinstance(index, faiss.IndexFlatIP)
        return obj

    def reset(self) -> None:
        self.index.reset()
