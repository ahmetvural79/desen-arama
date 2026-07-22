"""Ana pencere — sürükle-bırak sorgu, sonuç grid'i, indeksleme, ayarlar.

Tüm arayüz Türkçedir. Ağır işler arka plan iş parçacığında yürür; pencere
donmaz. Sonuçlar thumbnail + skor + kopya rozetiyle grid'de gösterilir; çift
tık dosyayı açar, sağ tık "klasörde göster / yolu kopyala" sunar.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QSize, QTimer, QObject, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QToolBar, QStatusBar,
)

from .. import __app_name__, __version__, config as cfg_mod
from ..core import osutil, paths
from ..services.engine import Engine
from .settings_dialog import SettingsDialog
from .workers import IndexWorker, SearchWorker, run_in_thread

THUMB_ICON = 190


class _WatchBridge(QObject):
    """watchdog iş parçacığından ana thread'e güvenli sinyal köprüsü."""

    changed = Signal()


class QueryDropLabel(QLabel):
    """Sorgu imajını sürükle-bırakla kabul eden önizleme alanı."""

    def __init__(self, on_image) -> None:
        super().__init__()
        self._on_image = on_image
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            "QLabel{border:2px dashed #888; border-radius:8px; color:#666;"
            "background:#fafafa;}"
        )
        self.setText("Sorgu görselini buraya\nsürükleyip bırakın\nveya 'Sorgu Seç'e tıklayın")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.splitext(path)[1].lower() in {
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
            }:
                self._on_image(path)
                break

    def show_image(self, path: str) -> None:
        pix = QPixmap(path)
        if not pix.isNull():
            self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine
        self.config = engine.config
        self.query_path: str | None = None
        self._threads: list = []  # thread/worker referanslarını canlı tut
        self._index_worker: IndexWorker | None = None

        self._watcher = None
        self._watch_bridge = _WatchBridge()
        self._watch_bridge.changed.connect(self._on_library_changed)
        self._rescan_timer: QTimer | None = None

        self.setWindowTitle(f"{__app_name__} {__version__}")
        self.resize(1100, 760)
        self._build_ui()
        self._refresh_status()
        self._setup_watch()

    # -- arayüz kurulumu ---------------------------------------------------- #
    def _build_ui(self) -> None:
        self._build_toolbar()

        central = QWidget()
        root = QHBoxLayout(central)

        # Sol panel: sorgu + kontroller
        left = QVBoxLayout()
        self.query_label = QueryDropLabel(self._set_query)
        left.addWidget(self.query_label)

        pick = QPushButton("Sorgu Seç…")
        pick.clicked.connect(self._pick_query)
        left.addWidget(pick)

        # Arka uç seçimi
        left.addWidget(QLabel("Arama yöntemi:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Hızlı (hash) — aynı/benzer şekil", cfg_mod.BACKEND_HASH)
        self.backend_combo.addItem("AI (DINOv2) — derin benzerlik", cfg_mod.BACKEND_EMBEDDING)
        self.backend_combo.addItem("Hibrit — hash + AI yeniden sıralama", cfg_mod.BACKEND_HYBRID)
        self.backend_combo.setCurrentIndex(
            [cfg_mod.BACKEND_HASH, cfg_mod.BACKEND_EMBEDDING, cfg_mod.BACKEND_HYBRID].index(self.config.backend)
        )
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        left.addWidget(self.backend_combo)

        self.tta_check = QCheckBox("Döndürme dayanıklılığı (8×TTA)")
        self.tta_check.setChecked(self.config.tta)
        self.tta_check.toggled.connect(self._on_tta_changed)
        left.addWidget(self.tta_check)

        # Renk önemi slider'ı
        left.addWidget(QLabel("Renk önemi:"))
        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setRange(0, 100)
        self.color_slider.setValue(int(self.config.color_alpha * 100))
        self.color_value = QLabel(f"{self.config.color_alpha:.2f}")
        self.color_slider.valueChanged.connect(lambda v: self.color_value.setText(f"{v/100:.2f}"))
        self.color_slider.sliderReleased.connect(self._maybe_research)
        crow = QHBoxLayout()
        crow.addWidget(self.color_slider)
        crow.addWidget(self.color_value)
        left.addLayout(crow)

        # Sonuç sayısı
        krow = QHBoxLayout()
        krow.addWidget(QLabel("Sonuç sayısı:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 500)
        self.k_spin.setValue(self.config.max_results)
        krow.addWidget(self.k_spin)
        left.addLayout(krow)

        self.search_btn = QPushButton("🔍  Ara")
        self.search_btn.clicked.connect(self._do_search)
        self.search_btn.setMinimumHeight(40)
        left.addWidget(self.search_btn)
        left.addStretch(1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(300)
        root.addWidget(left_widget)

        # Sağ panel: sonuç grid'i
        self.results = QListWidget()
        self.results.setViewMode(QListWidget.IconMode)
        self.results.setIconSize(QSize(THUMB_ICON, THUMB_ICON))
        self.results.setResizeMode(QListWidget.Adjust)
        self.results.setMovement(QListWidget.Static)
        self.results.setSpacing(10)
        self.results.setWordWrap(True)
        self.results.setUniformItemSizes(False)
        self.results.itemDoubleClicked.connect(self._open_item)
        self.results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results.customContextMenuRequested.connect(self._result_menu)
        root.addWidget(self.results, 1)

        self.setCentralWidget(central)

        # Durum çubuğu
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        sb.addPermanentWidget(self.progress)
        self.status_label = QLabel("Hazır")
        sb.addWidget(self.status_label)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Ana")
        tb.setMovable(False)
        self.addToolBar(tb)

        add_lib = QAction("Kütüphane Klasörü Ekle", self)
        add_lib.triggered.connect(self._add_library)
        tb.addAction(add_lib)

        self.index_action = QAction("İndeksle / Güncelle", self)
        self.index_action.triggered.connect(self._start_index)
        tb.addAction(self.index_action)

        self.cancel_action = QAction("İndekslemeyi Durdur", self)
        self.cancel_action.triggered.connect(self._cancel_index)
        self.cancel_action.setEnabled(False)
        tb.addAction(self.cancel_action)

        tb.addSeparator()
        settings = QAction("Ayarlar", self)
        settings.triggered.connect(self._open_settings)
        tb.addAction(settings)

        about = QAction("Hakkında", self)
        about.triggered.connect(self._about)
        tb.addAction(about)

    # -- sorgu -------------------------------------------------------------- #
    def _pick_query(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Sorgu görseli seç", "",
            "Görseller (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)",
        )
        if path:
            self._set_query(path)

    def _set_query(self, path: str) -> None:
        self.query_path = path
        self.query_label.show_image(path)
        self.status_label.setText(f"Sorgu: {os.path.basename(path)}")

    # -- ayar değişiklikleri ------------------------------------------------ #
    def _on_backend_changed(self, _idx: int) -> None:
        self.config.backend = self.backend_combo.currentData()
        self.config.save()
        # AI/hibrit moduna geçildiyse ve vektör yoksa kullanıcıyı uyar
        if self.config.uses_embedding():
            self.engine.open()  # vektör indeksini yüklemeyi dene
            if self.engine.store.count_vectors() == 0 and self.engine.store.count("ok") > 0:
                self.status_label.setText(
                    "AI modu seçildi — 'İndeksle / Güncelle' ile embedding'leri üretin."
                )

    def _on_tta_changed(self, checked: bool) -> None:
        self.config.tta = checked
        self.config.save()

    def _maybe_research(self) -> None:
        self.config.color_alpha = self.color_slider.value() / 100.0
        self.config.save()
        if self.query_path:
            self._do_search()

    # -- kütüphane / indeksleme -------------------------------------------- #
    def _add_library(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Kütüphane klasörü seç (ağ/UNC desteklenir)")
        if folder:
            if folder not in self.config.library_roots:
                self.config.library_roots.append(folder)
                self.config.save()
            self._refresh_status()
            reply = QMessageBox.question(
                self, "İndeksleme", f"'{folder}' eklendi. Şimdi indekslensin mi?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._start_index()

    def _start_index(self) -> None:
        if not self.config.library_roots:
            QMessageBox.warning(self, "Kütüphane yok", "Önce bir kütüphane klasörü ekleyin.")
            return
        self.index_action.setEnabled(False)
        self.cancel_action.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # belirsiz (tarama)

        worker = IndexWorker(self.engine)
        worker.progress.connect(self._on_index_progress)
        worker.finished.connect(self._on_index_finished)
        worker.failed.connect(self._on_index_failed)
        self._index_worker = worker
        thread, _ = run_in_thread(worker)
        self._threads.append((thread, worker))

    def _cancel_index(self) -> None:
        if self._index_worker:
            self._index_worker.cancel()
            self.status_label.setText("İndeksleme durduruluyor…")

    def _on_index_progress(self, prog) -> None:
        if prog.phase == "process" and prog.total:
            self.progress.setRange(0, prog.total)
            self.progress.setValue(prog.done)
            self.status_label.setText(
                f"İndeksleniyor: {prog.done}/{prog.total} — hata:{prog.errors} "
                f"hız:{prog.rate:.0f}/sn  (yeni:{prog.new} değişen:{prog.changed})"
            )
        elif prog.phase == "scan":
            self.status_label.setText("Klasör taranıyor…")
        elif prog.phase == "finalize":
            self.status_label.setText("İndeksler kuruluyor…")

    def _on_index_finished(self, prog) -> None:
        self.index_action.setEnabled(True)
        self.cancel_action.setEnabled(False)
        self.progress.setVisible(False)
        self._refresh_status()
        st = self.engine.store.stats()
        self.status_label.setText(
            f"İndeksleme bitti: {prog.done} işlendi, {prog.errors} hata, "
            f"{prog.elapsed:.0f} sn — toplam {st['ok']} imaj."
        )

    def _on_index_failed(self, msg: str) -> None:
        self.index_action.setEnabled(True)
        self.cancel_action.setEnabled(False)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "İndeksleme hatası", msg)

    # -- arama -------------------------------------------------------------- #
    def _do_search(self) -> None:
        if not self.query_path:
            QMessageBox.information(self, "Sorgu yok", "Önce bir sorgu görseli seçin.")
            return
        if self.engine.store.count("ok") == 0:
            QMessageBox.information(self, "Boş kütüphane", "Önce bir kütüphaneyi indeksleyin.")
            return
        self.search_btn.setEnabled(False)
        self.status_label.setText("Aranıyor…")
        alpha = self.color_slider.value() / 100.0
        worker = SearchWorker(self.engine, self.query_path, self.k_spin.value(), alpha)
        worker.results.connect(self._on_results)
        worker.failed.connect(self._on_search_failed)
        thread, _ = run_in_thread(worker)
        self._threads.append((thread, worker))

    def _on_results(self, results: list) -> None:
        self.search_btn.setEnabled(True)
        self.results.clear()
        if not results:
            self.status_label.setText("Sonuç bulunamadı.")
            return
        for r in results:
            item = QListWidgetItem()
            if r.thumb_path and os.path.exists(r.thumb_path):
                item.setIcon(QIcon(QPixmap(r.thumb_path)))
            badge = "  🟢KOPYA" if r.is_duplicate else ""
            name = os.path.basename(r.path)
            item.setText(f"%{r.score*100:.0f}{badge}\n{name}")
            tip = (f"{r.path}\nSkor: {r.score:.3f} (desen {r.pattern_score:.3f}, "
                   f"renk {r.color_sim:.3f})")
            if r.hamming is not None:
                tip += f"\npHash mesafesi: {r.hamming}"
            tip += f"\nBoyut: {r.width}×{r.height}"
            item.setToolTip(tip)
            item.setData(Qt.UserRole, r.path)
            if r.below_threshold:
                item.setForeground(Qt.gray)
            self.results.addItem(item)
        self.status_label.setText(f"{len(results)} sonuç bulundu.")

    def _on_search_failed(self, msg: str) -> None:
        self.search_btn.setEnabled(True)
        QMessageBox.critical(self, "Arama hatası", msg)

    # -- sonuç etkileşimi --------------------------------------------------- #
    def _open_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            try:
                osutil.open_path(path)
            except Exception as e:
                QMessageBox.warning(self, "Açılamadı", str(e))

    def _result_menu(self, pos) -> None:
        item = self.results.itemAt(pos)
        if not item:
            return
        path = item.data(Qt.UserRole)
        menu = QMenu(self)
        act_open = menu.addAction("Dosyayı aç")
        act_reveal = menu.addAction("Klasörde göster")
        act_copy = menu.addAction("Yolu kopyala")
        act_use = menu.addAction("Bunu sorgu yap")
        chosen = menu.exec(self.results.mapToGlobal(pos))
        if chosen == act_open:
            osutil.open_path(path)
        elif chosen == act_reveal:
            osutil.reveal_in_folder(path)
        elif chosen == act_copy:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(path)
        elif chosen == act_use:
            self._set_query(path)
            self._do_search()

    # -- ayarlar / durum ---------------------------------------------------- #
    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.config = dlg.updated_config
            self.engine.config = self.config
            self.config.save()
            self.engine.open()  # indeksleri yeni ayarlara göre yeniden kur
            self._sync_controls()
            self._refresh_status()
            self._setup_watch()  # izleme ayarları değişmiş olabilir

    def _sync_controls(self) -> None:
        idx = [cfg_mod.BACKEND_HASH, cfg_mod.BACKEND_EMBEDDING, cfg_mod.BACKEND_HYBRID].index(self.config.backend)
        self.backend_combo.setCurrentIndex(idx)
        self.tta_check.setChecked(self.config.tta)
        self.color_slider.setValue(int(self.config.color_alpha * 100))

    def _refresh_status(self) -> None:
        st = self.engine.store.stats()
        roots = len(self.config.library_roots)
        self.setWindowTitle(
            f"{__app_name__} {__version__} — {st['ok']} imaj / {roots} klasör"
        )

    # -- canlı izleme / periyodik tarama ------------------------------------ #
    def _setup_watch(self) -> None:
        """Yapılandırmaya göre canlı izleme ve/veya periyodik yeniden tarama kurar."""
        self._teardown_watch()
        if not self.config.library_roots:
            return
        # Canlı izleme (native/polling)
        if self.config.watch_mode != "off":
            from ..services.watcher import LibraryWatcher

            self._watcher = LibraryWatcher()
            self._watcher.start(
                self.config.library_roots, self.config.watch_mode,
                on_change=self._watch_bridge.changed.emit,  # thread-güvenli
            )
        # Periyodik yeniden tarama (ağ paylaşımları için önerilen)
        if self.config.rescan_interval_sec > 0:
            self._rescan_timer = QTimer(self)
            self._rescan_timer.timeout.connect(self._on_library_changed)
            self._rescan_timer.start(self.config.rescan_interval_sec * 1000)

    def _teardown_watch(self) -> None:
        if self._watcher:
            self._watcher.stop()
            self._watcher = None
        if self._rescan_timer:
            self._rescan_timer.stop()
            self._rescan_timer = None

    def _on_library_changed(self) -> None:
        """Değişiklik algılandı — indeksleme çalışmıyorsa artımlı güncelle."""
        if self.index_action.isEnabled():  # indeksleme sürmüyorsa
            self.status_label.setText("Değişiklik algılandı — indeks güncelleniyor…")
            self._start_index()

    def _about(self) -> None:
        QMessageBox.about(
            self, f"{__app_name__} hakkında",
            f"<b>{__app_name__}</b> v{__version__}<br><br>"
            "Halı deseni görsel benzerlik arama.<br>"
            "Algısal hash (czkawka/imagededup tarzı) + DINOv2 (ONNX) + FAISS.<br><br>"
            f"Veri dizini: {paths.data_dir()}<br>"
            "Tüm işlem yereldedir; buluta veri gönderilmez.",
        )

    def closeEvent(self, e) -> None:
        if self._index_worker:
            self._index_worker.cancel()
        self._teardown_watch()
        e.accept()
