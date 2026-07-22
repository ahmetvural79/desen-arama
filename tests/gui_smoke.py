"""Offscreen GUI smoke testi — pencere kurulumu ve threaded arama akışı.

Gerçek görsel doğrulama kullanıcının Windows makinesinde yapılır; bu test
yalnızca pencerenin çöküp çökmediğini ve arama akışının grid'i doldurduğunu
denetler. ``QT_QPA_PLATFORM=offscreen`` ile başsız çalışır.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from desenarama import config as cfg_mod
from desenarama.gui.main_window import MainWindow
from desenarama.services.engine import Engine


def main() -> int:
    dataset = sys.argv[1]
    app = QApplication(sys.argv[:1])

    cfg = cfg_mod.AppConfig.load()
    cfg.backend = cfg_mod.BACKEND_HASH
    engine = Engine(cfg)
    engine.open()
    assert engine.store.count("ok") > 0, "kütüphane indekslenmemiş"

    win = MainWindow(engine)
    win.show()
    print("[1/3] MainWindow kuruldu; ok imaj:", engine.store.count("ok"))

    captured = {"n": None}
    original = win._on_results

    def hook(results):
        captured["n"] = len(results)
        original(results)

    win._on_results = hook
    win._set_query(os.path.join(dataset, "fam3_orig.png"))
    win._do_search()
    print("[2/3] Arama başlatıldı, sonuç bekleniyor…")

    t0 = time.time()
    # Sonuç gelene VE grid dolana kadar olay döngüsünü çevir (yarış koşulunu ele).
    while (captured["n"] is None or win.results.count() != captured["n"]) and time.time() - t0 < 15:
        app.processEvents(QEventLoop.AllEvents, 100)
        time.sleep(0.02)

    assert captured["n"] is not None, "arama zaman aşımına uğradı"
    assert captured["n"] > 0, "arama sonuç döndürmedi"
    assert win.results.count() == captured["n"], "grid sonuç sayısı uyuşmuyor"
    print(f"[3/3] Grid doldu: {win.results.count()} sonuç. GUI SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
