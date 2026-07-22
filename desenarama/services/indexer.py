"""İndeksleme servisi — üretici-tüketici hattıyla dayanıklı, ağ-dostu tarama.

Akış:

1. Klasör(ler) taranır, DB ile fark alınır (yeni / değişmiş / kayıp).
2. Kayıp dosyalar ``missing`` işaretlenir — **silinmez** (ağ kopması gerçek
   silme sanılmasın; temizlik kullanıcı onayıyla yapılır).
3. İşlenecek dosyalar akışlı **batch'ler** hâlinde işlenir:
   * I/O iş parçacığı havuzu dosyayı **bir kez** okur, çözer, thumbnail üretir,
     4 algısal hash + renk histogramını hesaplar (ağdan tek okuma).
   * AI modu açıksa toplanan RGB'lerden embedding **batch** olarak çıkarılır ve
     DB'deki ``vectors`` tablosuna yazılır.
   * Her dosya kaydı DB'ye yazılır; hatalı dosya ``error`` işaretlenip atlanır.
4. Bittiğinde motorun hash/vektör indeksleri yeniden kurulur.

Bellek, batch boyutuyla sınırlıdır (tüm arşiv belleğe alınmaz). Duraklat/devam/
iptal desteklenir.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..core import colorhist, hasher, imageio, paths, scanner, store
from ..core.imageio import ImageLoadError
from .engine import Engine

log = logging.getLogger("desenarama.indexer")


@dataclass
class IndexProgress:
    phase: str = "scan"          # scan | process | finalize | done | error | cancelled
    total: int = 0
    done: int = 0
    errors: int = 0
    current: str = ""
    new: int = 0
    changed: int = 0
    missing: int = 0
    elapsed: float = 0.0
    rate: float = 0.0            # imaj/sn


@dataclass(eq=False)  # eq=False: ndarray alanları eşitlik karşılaştırmasını bozar
class _Processed:
    entry: scanner.ScanEntry
    width: int
    height: int
    hashes: dict[str, bytes]
    color: bytes
    thumb_name: str
    rgb: np.ndarray | None       # yalnızca AI modunda tutulur
    error: str | None = None


def _thumb_name(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest() + ".jpg"


class IndexerService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.config = engine.config
        self.store = engine.store
        self._cancel = threading.Event()
        self._pause = threading.Event()

    # -- kontrol ------------------------------------------------------------ #
    def cancel(self) -> None:
        self._cancel.set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def _wait_if_paused(self) -> None:
        while self._pause.is_set() and not self._cancel.is_set():
            time.sleep(0.1)

    # -- ana giriş ---------------------------------------------------------- #
    def reindex(self, progress_cb=None, mark_missing: bool = True) -> IndexProgress:
        """Kütüphaneyi (yeniden) indeksler. ``progress_cb(IndexProgress)`` çağrılır."""
        prog = IndexProgress(phase="scan")
        started = time.time()
        roots = self.config.library_roots
        exts = self.config.ext_set()
        self._auto_tune(roots)

        # 1) tarama + fark
        existing = self.store.existing_keys()
        diff = scanner.diff(roots, existing, exts)
        prog.new = len(diff.new)
        prog.changed = len(diff.changed)
        prog.missing = len(diff.missing)
        to_process = list(diff.to_process)

        # 1b) AI/hibrit modda: mevcut ama vektörü eksik imajları da işle
        # (hash modundan AI moduna geçişte embedding'leri tamamlar).
        if self.config.uses_embedding():
            have = {e.key for e in to_process}
            for key, path, mtime, size in self.store.keys_missing_vectors():
                if key not in have:
                    to_process.append(scanner.ScanEntry(path=path, key=key, mtime=mtime, size=size))

        prog.total = len(to_process)
        prog.phase = "process"
        if progress_cb:
            progress_cb(prog)

        # 2) kayıp dosyalar
        if mark_missing and diff.missing:
            for k in diff.missing:
                self.store.mark_status(k, "missing")

        # 3) işleme (akışlı batch)
        use_embed = self.config.uses_embedding()
        embedder = self.engine.get_embedder() if use_embed else None
        if use_embed and embedder is not None:
            # Vektör boyutunu kaydet — motorun FAISS'i yeniden kurabilmesi için şart.
            self.store.set_meta("embed_dim", str(embedder.dim))
            self.store.set_meta("embedder_name", embedder.name)
        batch_size = max(8, self.config.cpu_batch)
        io_workers = max(1, self.config.io_workers)

        with ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="io") as pool:
            for chunk in _chunks(to_process, batch_size):
                if self._cancel.is_set():
                    prog.phase = "cancelled"
                    break
                self._wait_if_paused()
                processed = list(pool.map(lambda e: self._process_one(e, keep_rgb=use_embed), chunk))

                # 3a) embedding batch (AI modu) — kimlik (id) tabanlı eşleme
                vec_items: list[tuple[int, bytes]] = []
                vec_for: dict[int, bytes] = {}
                good = [p for p in processed if p.error is None]
                if use_embed and good:
                    vecs = embedder.embed_batch([p.rgb for p in good])
                    for p, v in zip(good, vecs):
                        vec_for[id(p)] = np.ascontiguousarray(v, dtype=np.float32).tobytes()

                # 3b) DB yazımı
                for p in processed:
                    if p.error is not None:
                        self.store.upsert(
                            key=p.entry.key, path=p.entry.path, mtime=p.entry.mtime,
                            size=p.entry.size, status="error", error=p.error,
                            indexed_at=store.now(),
                        )
                        prog.errors += 1
                    else:
                        img_id = self.store.upsert(
                            key=p.entry.key, path=p.entry.path, mtime=p.entry.mtime,
                            size=p.entry.size, width=p.width, height=p.height,
                            phash=p.hashes["phash"], dhash=p.hashes["dhash"],
                            ahash=p.hashes["ahash"], whash=p.hashes["whash"],
                            color=p.color, thumb=p.thumb_name, status="ok",
                            error=None, indexed_at=store.now(),
                        )
                        if use_embed and id(p) in vec_for:
                            vec_items.append((img_id, vec_for[id(p)]))
                    prog.done += 1
                    prog.current = p.entry.path

                if vec_items:
                    self.store.upsert_vectors(vec_items)

                prog.elapsed = time.time() - started
                prog.rate = prog.done / prog.elapsed if prog.elapsed > 0 else 0.0
                if progress_cb:
                    progress_cb(prog)

        # 4) indeksleri yeniden kur
        if not self._cancel.is_set():
            prog.phase = "finalize"
            if progress_cb:
                progress_cb(prog)
            self.engine.rebuild_maps()
            prog.phase = "done"
        prog.elapsed = time.time() - started
        if progress_cb:
            progress_cb(prog)
        self.store.set_meta("last_index_at", str(store.now()))
        log.info("İndeksleme bitti: %d işlendi, %d hata, %.1f sn (%.1f imaj/sn)",
                 prog.done, prog.errors, prog.elapsed, prog.rate)
        return prog

    # -- tek dosya işleme --------------------------------------------------- #
    def _process_one(self, entry: scanner.ScanEntry, keep_rgb: bool) -> _Processed:
        hs = self.config.hash_size
        try:
            data = imageio.read_bytes(entry.path)
            img = imageio.load_from_bytes(data)
            rgb = img.rgb
            hashes = {
                a: hasher.to_blob(hasher.compute(rgb, a, hs), hs) for a in hasher.ALGOS
            }
            color = colorhist.to_blob(colorhist.histogram(rgb))
            thumb_name = _thumb_name(entry.key)
            try:
                thumb_bytes = imageio.make_thumbnail(rgb, self.config.thumb_size)
                with open(paths.thumbs_dir() / thumb_name, "wb") as f:
                    f.write(thumb_bytes)
            except Exception:
                thumb_name = ""  # thumbnail üretilemese de indeks sürer
            return _Processed(
                entry=entry, width=img.width, height=img.height, hashes=hashes,
                color=color, thumb_name=thumb_name, rgb=rgb if keep_rgb else None,
            )
        except ImageLoadError as e:
            return _Processed(entry=entry, width=0, height=0, hashes={}, color=b"",
                              thumb_name="", rgb=None, error=str(e))
        except Exception as e:  # beklenmedik — hattı düşürme
            return _Processed(entry=entry, width=0, height=0, hashes={}, color=b"",
                              thumb_name="", rgb=None, error=f"beklenmeyen: {e}")

    # -- ağ ayarı ----------------------------------------------------------- #
    def _auto_tune(self, roots: list[str]) -> None:
        """Ağ paylaşımı tespit edilirse I/O eşzamanlılığını artırır.

        Ağ diskinde darboğaz gecikmedir (bant genişliği değil); daha çok
        eşzamanlı okuma gecikmeyi gizler. Yerel diskte fazla iş parçacığı
        fayda etmez, hatta zarar verir.
        """
        if not self.config.auto_tune_network:
            return
        if any(paths.is_network_path(r) for r in roots):
            if self.config.io_workers < 16:
                log.info("Ağ paylaşımı tespit edildi — I/O eşzamanlılığı 16'ya çıkarıldı.")
                self.config.io_workers = 16


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
