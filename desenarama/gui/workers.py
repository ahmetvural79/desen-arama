"""Arka plan iş parçacıkları — UI'yi dondurmadan indeksleme ve arama.

Ağır işler (klasör tarama, embedding çıkarma, FAISS kurma, arama) ayrı bir
``QThread`` üzerinde çalışır; ilerleme ve sonuçlar Qt sinyalleriyle GUI'ye
taşınır. Böylece ana pencere her zaman yanıt verir (donmaz).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from ..services.indexer import IndexerService, IndexProgress
from ..services.search import SearchService


class IndexWorker(QObject):
    """İndekslemeyi arka planda çalıştırır; ilerlemeyi sinyalle bildirir."""

    progress = Signal(object)   # IndexProgress
    finished = Signal(object)   # IndexProgress
    failed = Signal(str)

    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        self._service = IndexerService(engine)

    def cancel(self) -> None:
        self._service.cancel()

    def pause(self) -> None:
        self._service.pause()

    def resume(self) -> None:
        self._service.resume()

    def run(self) -> None:
        try:
            result = self._service.reindex(progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:  # UI'yi çökertme — hatayı bildir
            self.failed.emit(str(e))


class SearchWorker(QObject):
    """Aramayı arka planda çalıştırır; sonuçları sinyalle döndürür."""

    results = Signal(list)      # list[SearchResult]
    failed = Signal(str)

    def __init__(self, engine, query_path: str, k: int, color_alpha: float) -> None:
        super().__init__()
        self.engine = engine
        self.query_path = query_path
        self.k = k
        self.color_alpha = color_alpha

    def run(self) -> None:
        try:
            svc = SearchService(self.engine)
            res = svc.search(query_path=self.query_path, k=self.k, color_alpha=self.color_alpha)
            self.results.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


def run_in_thread(worker: QObject, on_done=None):
    """Bir worker'ı yeni bir QThread'de çalıştırır; başvuruları döndürür.

    Çağıran, thread ve worker referanslarını canlı tutmalıdır (aksi halde GC
    onları toplar). Dönen (thread, worker) çifti pencere tarafında saklanır.
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def _cleanup(*_):
        thread.quit()

    if hasattr(worker, "finished"):
        worker.finished.connect(_cleanup)
    if hasattr(worker, "results"):
        worker.results.connect(_cleanup)
    if hasattr(worker, "failed"):
        worker.failed.connect(_cleanup)
    thread.finished.connect(thread.deleteLater)
    if on_done:
        thread.finished.connect(on_done)
    thread.start()
    return thread, worker
