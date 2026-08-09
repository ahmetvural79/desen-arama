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

#: Skor kalibrasyonu için saklanan örneklem büyüklüğü. 2048 × 768 float32
#: ≈ 6 MB; medyan tahmini için fazlasıyla yeter.
BASELINE_SAMPLE_SIZE = 2048

#: Bu sayının altındaki kütüphanelerde örneklem medyanı güvenilir değildir;
#: kalibrasyon uygulanmaz.
BASELINE_MIN_VECTORS = 32

log = logging.getLogger("desenarama.engine")

_HASH_COL = {"phash": "phash", "dhash": "dhash", "ahash": "ahash", "whash": "whash"}


class Engine:
    def __init__(self, config: cfg_mod.AppConfig | None = None) -> None:
        self.config = config or cfg_mod.AppConfig.load()
        self.store = store.ImageStore()
        self.hash_index: hashindex.HashIndex | None = None
        self.vector_index: vindex.VectorIndex | None = None
        self.row_to_id: dict[int, int] = {}
        #: Skor kalibrasyonu için kütüphaneden alınan temsilî vektör örneklemi
        #: (normalize, en fazla :data:`BASELINE_SAMPLE_SIZE` satır).
        self.baseline_vectors: np.ndarray | None = None
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
            self.baseline_vectors = None
            return
        n = self.store.count_vectors()
        # Kalibrasyon örneklemi: kütüphane boyunca eşit aralıklarla seç.
        # Rastgele yerine adımlı seçim, tarama sırasının (klasör/tarih) yarattığı
        # kümelenmeye karşı daha temsilî ve tekrarlanabilirdir.
        sample_step = max(1, n // BASELINE_SAMPLE_SIZE)
        sample: list[np.ndarray] = []
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

        # Karo indekslemede aynı imaj birden çok FAISS satırına karşılık gelir;
        # ``row_to_id`` her satırı sahibi imaja eşler, arama tarafı da imaj
        # başına karolar arası maksimumu alır.
        for pos, (image_id, _tile, blob) in enumerate(self.store.iter_vectors()):
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            batch.append(vec)
            ids.append(image_id)
            if pos % sample_step == 0 and len(sample) < BASELINE_SAMPLE_SIZE:
                sample.append(vec)
            if len(batch) >= 4096:
                flush()
        flush()
        self.vector_index = index
        self.row_to_id = row_to_id
        self.baseline_vectors = (
            vindex.VectorIndex.normalize(np.stack(sample))
            if len(sample) >= BASELINE_MIN_VECTORS else None
        )
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
    def tiles_per_image(self) -> int:
        """İmaj başına indekslenen vektör sayısı (tam kare + karolar).

        Aday sayısını ölçeklemek için kullanılır: karo indekslemede top-k'yı tek
        bir imajın karoları doldurabilir.
        """
        grid = self.config.tile_grid
        return 1 + grid * grid if grid > 1 else 1

    def model_spec(self):
        return models.resolve(self.config.model_key)

    def model_available(self) -> bool:
        """Seçili AI modeli yerelde hazır mı? (indirme denemeden)"""
        return models.is_available(self.model_spec())

    def get_embedder(self, allow_download: bool = True, progress=None,
                     allow_fallback: bool = False):
        """Embedder'ı (tembel) yükler.

        Model yoksa ve indirilemiyorsa :class:`EmbedderUnavailable` yükselir —
        v1.0'daki gibi sessizce zayıf bir yedeğe düşülmez; çağıran katman
        durumu kullanıcıya bildirmekle yükümlüdür.
        """
        if self._embedder is not None:
            return self._embedder
        from ..core import embedder as emb_mod

        model_path = None
        spec = self.model_spec()
        if models.is_available(spec):
            model_path = models.local_path(spec)
        elif allow_download:
            try:
                model_path = models.download(spec, progress=progress)
            except Exception as e:
                log.warning("Model indirilemedi: %s", e)
                if not allow_fallback:
                    raise emb_mod.EmbedderUnavailable(str(e)) from e

        self._embedder = emb_mod.load_embedder(
            model_path, prefer_gpu=self.config.prefer_gpu,
            allow_fallback=allow_fallback,
        )
        self.store.set_meta("embedder_name", self._embedder.name)
        self.store.set_meta("embed_dim", str(self._embedder.dim))
        return self._embedder

    def close(self) -> None:
        self.store.close()
