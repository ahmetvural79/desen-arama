"""Motor — depo, hash indeksi, FAISS ve embedder'ı tek noktada birleştirir.

İndeksleyici ve arama servisleri bu motoru paylaşır. Motor açılışta:

* SQLite deposunu açar,
* seçili algoritma için hash'lerden bellek-içi BK-tree/doğrusal indeksi kurar
  (hash'ler küçük olduğundan bu her açılışta hızlıdır),
* AI modu açıksa FAISS vektör indeksini diskten yükler ve satır→imaj eşlemesini
  çıkarır,
* embedder'ı yalnızca gerektiğinde (tembel) yükler; model yoksa fallback'e düşer.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .. import config as cfg_mod
from ..core import hasher, hashindex, models, paths, store, vindex

# Kütüphane bu eşiği aşarsa FAISS Flat yerine HNSW (yaklaşık) indekse geçilir.
HNSW_THRESHOLD = 200_000

log = logging.getLogger("desenarama.engine")

_HASH_COL = {"phash": "phash", "dhash": "dhash", "ahash": "ahash", "whash": "whash"}


class Engine:
    def __init__(self, config: cfg_mod.AppConfig | None = None) -> None:
        self.config = config or cfg_mod.AppConfig.load()
        self.store = store.ImageStore()
        self.hash_index: hashindex.HashIndex | None = None
        self.vector_index: vindex.VectorIndex | None = None
        self.row_to_id: dict[int, int] = {}
        self._embedder = None

    # -- yollar ------------------------------------------------------------- #
    @property
    def faiss_path(self) -> str:
        return str(paths.data_dir() / "vectors.faiss")

    # -- açılış / indeks kurulumu ------------------------------------------ #
    def open(self) -> None:
        self._build_hash_index()
        if self.config.uses_embedding():
            self._load_vector_index()

    def _build_hash_index(self) -> None:
        algo = self.config.hash_algo
        bits = hasher.max_bits(self.config.hash_size)
        col = _HASH_COL[algo]
        idx = hashindex.HashIndex(hash_bits=bits)
        for rec in self.store.iter_records(only_ok=True):
            blob = getattr(rec, col)
            if blob is not None:
                idx.add(rec.id, hasher.from_blob(blob))
        idx.build()
        self.hash_index = idx
        log.info("Hash indeksi kuruldu: %d imaj (%s, %d-bit)", len(idx), algo, bits)

    def _load_vector_index(self) -> None:
        """FAISS indeksini ``vectors`` tablosundan yeniden kurar (doğruluk kaynağı).

        Vektörler DB'de tutulduğundan indeks her zaman temiz kurulur; değişmiş/
        silinmiş imajların eski vektörleri asla ortada kalmaz (orphan olmaz).
        """
        dim = int(self.store.get_meta("embed_dim", "0") or "0")
        if not dim or self.store.count_vectors() == 0:
            self.vector_index = None
            self.row_to_id = {}
            return
        n = self.store.count_vectors()
        use_hnsw = n > HNSW_THRESHOLD
        index = vindex.VectorIndex(dim, hnsw=use_hnsw)
        row_to_id: dict[int, int] = {}
        batch: list[np.ndarray] = []
        ids: list[int] = []

        def flush():
            if batch:
                start = index.add(np.stack(batch))
                for offset, img_id in enumerate(ids):
                    row_to_id[start + offset] = img_id
                batch.clear()
                ids.clear()

        for image_id, blob in self.store.iter_vectors():
            batch.append(np.frombuffer(blob, dtype=np.float32).copy())
            ids.append(image_id)
            if len(batch) >= 4096:
                flush()
        flush()
        self.vector_index = index
        self.row_to_id = row_to_id
        try:
            index.save(self.faiss_path)
        except Exception as e:  # disk dolu vb. — indeks bellekte yine çalışır
            log.warning("FAISS diske yazılamadı: %s", e)
        log.info("FAISS indeksi kuruldu: %d vektör (dim=%d, hnsw=%s)", index.ntotal, dim, use_hnsw)

    def rebuild_maps(self) -> None:
        """İndeksleme sonrası hash indeksini ve (varsa) vektör indeksini yeniden kur."""
        self._build_hash_index()
        if self.config.uses_embedding():
            self._load_vector_index()

    # -- embedder (tembel) -------------------------------------------------- #
    def get_embedder(self, allow_download: bool = True, progress=None):
        if self._embedder is not None:
            return self._embedder
        from ..core import embedder as emb_mod

        model_path = None
        spec = models.resolve(self.config.model_key)
        if models.is_available(spec):
            model_path = models.local_path(spec)
        elif allow_download:
            try:
                model_path = models.download(spec, progress=progress)
            except Exception as e:
                log.warning("Model indirilemedi: %s", e)
                model_path = None
        self._embedder = emb_mod.load_embedder(
            model_path, prefer_gpu=self.config.prefer_gpu
        )
        self.store.set_meta("embedder_name", self._embedder.name)
        self.store.set_meta("embed_dim", str(self._embedder.dim))
        return self._embedder

    def close(self) -> None:
        self.store.close()
