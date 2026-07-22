"""Canlı değişiklik izleme — yerelde native olaylar, ağda polling.

Ağ paylaşımlarında (SMB) dosya sistemi olayları güvenilmezdir; bu yüzden ağ
kökleri için ``PollingObserver`` (periyodik tarama) kullanılır. Yerel kökler
için native ``Observer`` daha verimlidir. ``auto`` modu bunu köke göre seçer.

Değişiklikler bir "sessizlik" penceresiyle geciktirilerek (debounce) toplanır;
çok sayıda ardışık değişiklik tek bir yeniden-indeksleme tetikler.
"""

from __future__ import annotations

import logging
import threading

from ..core import paths

log = logging.getLogger("desenarama.watcher")


class LibraryWatcher:
    def __init__(self, debounce_sec: float = 3.0) -> None:
        self.debounce_sec = debounce_sec
        self._observers: list = []
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._on_change = None

    def start(self, roots: list[str], mode: str, on_change) -> bool:
        """İzlemeyi başlatır. watchdog yoksa ``False`` döner (GUI periyodik
        rescan'e düşebilir)."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
            from watchdog.observers.polling import PollingObserver
        except Exception:
            log.info("watchdog yok — canlı izleme devre dışı (periyodik tarama kullanın).")
            return False

        self._on_change = on_change

        class _Handler(FileSystemEventHandler):
            def __init__(self, outer):
                self._outer = outer

            def on_any_event(self, event):
                if event.is_directory:
                    return
                self._outer._schedule()

        for root in roots:
            use_polling = mode == "polling" or (mode == "auto" and paths.is_network_path(root))
            obs = PollingObserver(timeout=5) if use_polling else Observer()
            try:
                obs.schedule(_Handler(self), root, recursive=True)
                obs.start()
                self._observers.append(obs)
                log.info("İzleme başladı: %s (%s)", root, "polling" if use_polling else "native")
            except Exception as e:
                log.warning("İzlenemiyor: %s (%s)", root, e)
        return bool(self._observers)

    def _schedule(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_sec, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        if self._on_change:
            try:
                self._on_change()
            except Exception as e:
                log.warning("Değişiklik geri çağrımı hatası: %s", e)

    def stop(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
        for obs in self._observers:
            try:
                obs.stop()
                obs.join(timeout=2)
            except Exception:
                pass
        self._observers.clear()
