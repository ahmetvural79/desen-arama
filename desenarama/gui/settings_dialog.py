"""Ayarlar diyaloğu — kütüphane, arka uç, model ve ağ/performans parametreleri."""

from __future__ import annotations

import copy

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QFileDialog, QWidget,
)

from .. import config as cfg_mod
from ..core import formats


class SettingsDialog(QDialog):
    def __init__(self, config: cfg_mod.AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.resize(560, 620)
        self.updated_config = copy.deepcopy(config)
        self.extensions_widened = False
        self._build(config)

    def _build(self, cfg: cfg_mod.AppConfig) -> None:
        layout = QVBoxLayout(self)

        # -- kütüphane klasörleri -- #
        layout.addWidget(QLabel("<b>Kütüphane klasörleri</b> (ağ/UNC desteklenir):"))
        self.roots_list = QListWidget()
        self.roots_list.addItems(cfg.library_roots)
        layout.addWidget(self.roots_list)
        row = QHBoxLayout()
        add_btn = QPushButton("Ekle…")
        add_btn.clicked.connect(self._add_root)
        rm_btn = QPushButton("Kaldır")
        rm_btn.clicked.connect(self._remove_root)
        row.addWidget(add_btn)
        row.addWidget(rm_btn)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(self._build_formats_group(cfg))

        form = QFormLayout()

        self.backend_combo = QComboBox()
        for label, val in [("Hızlı (hash)", cfg_mod.BACKEND_HASH),
                           ("AI (DINOv2)", cfg_mod.BACKEND_EMBEDDING),
                           ("Hibrit", cfg_mod.BACKEND_HYBRID)]:
            self.backend_combo.addItem(label, val)
        self.backend_combo.setCurrentIndex(
            [cfg_mod.BACKEND_HASH, cfg_mod.BACKEND_EMBEDDING, cfg_mod.BACKEND_HYBRID].index(cfg.backend)
        )
        form.addRow("Arama yöntemi:", self.backend_combo)

        self.hash_algo = QComboBox()
        self.hash_algo.addItems(["phash", "dhash", "ahash", "whash"])
        self.hash_algo.setCurrentText(cfg.hash_algo)
        form.addRow("Hash algoritması:", self.hash_algo)

        self.hash_size = QComboBox()
        self.hash_size.addItem("64-bit (hızlı)", 8)
        self.hash_size.addItem("256-bit (ince ayrım)", 16)
        self.hash_size.setCurrentIndex(0 if cfg.hash_size == 8 else 1)
        form.addRow("Hash boyutu:", self.hash_size)

        self.model_combo = QComboBox()
        self.model_combo.addItem("DINOv2-S (384, hızlı)", "dinov2-small")
        self.model_combo.addItem("DINOv2-B (768, kaliteli)", "dinov2-base")
        self.model_combo.setCurrentIndex(0 if cfg.model_key == "dinov2-small" else 1)
        form.addRow("AI modeli:", self.model_combo)

        self.gpu_check = QCheckBox("GPU (DirectML/CUDA) kullan — varsa")
        self.gpu_check.setChecked(cfg.prefer_gpu)
        form.addRow("", self.gpu_check)

        self.dup_spin = QSpinBox()
        self.dup_spin.setRange(0, 40)
        self.dup_spin.setValue(cfg.duplicate_hamming)
        form.addRow("Kopya eşiği (Hamming):", self.dup_spin)

        self.maxdist_spin = QSpinBox()
        self.maxdist_spin.setRange(0, 64)
        self.maxdist_spin.setValue(cfg.hash_max_distance)
        form.addRow("Hash arama eşiği (0=sınırsız):", self.maxdist_spin)

        self.io_spin = QSpinBox()
        self.io_spin.setRange(1, 64)
        self.io_spin.setValue(cfg.io_workers)
        form.addRow("I/O iş parçacığı (ağ için ↑):", self.io_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(cfg.cpu_batch)
        form.addRow("Embedding batch:", self.batch_spin)

        self.autotune_check = QCheckBox("Ağ paylaşımı tespitinde I/O eşzamanlılığını otomatik artır")
        self.autotune_check.setChecked(cfg.auto_tune_network)
        form.addRow("", self.autotune_check)

        self.watch_combo = QComboBox()
        self.watch_combo.addItems(["auto", "native", "polling", "off"])
        self.watch_combo.setCurrentText(cfg.watch_mode)
        form.addRow("Değişiklik izleme:", self.watch_combo)

        self.rescan_spin = QSpinBox()
        self.rescan_spin.setRange(0, 86400)
        self.rescan_spin.setValue(cfg.rescan_interval_sec)
        form.addRow("Periyodik yeniden tarama (sn, 0=kapalı):", self.rescan_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- dosya formatları --------------------------------------------------- #
    def _build_formats_group(self, cfg: cfg_mod.AppConfig):
        """Uzantı seçimi — serbest metin yerine onay kutuları.

        Eskiden virgülle ayrılmış bir metin kutusuydu; kullanıcının ``.bmp``
        yazması gerektiğini bilmesi gerekiyordu ve yazım hatası sessizce
        yutuluyordu. Artık desteklenen her format görünür ve tıklanabilir.
        """
        group = QGroupBox("Aranacak dosya formatları")
        grid = QGridLayout(group)
        self._initial_exts = formats.normalize_all(cfg.extensions)
        self.ext_checks: dict[str, QCheckBox] = {}

        for col, (family, exts) in enumerate(formats.FORMAT_FAMILIES):
            grid.addWidget(QLabel(f"<b>{family}</b>"), 0, col)
            for row, ext in enumerate(exts, start=1):
                box = QCheckBox(ext)
                box.setChecked(ext in self._initial_exts)
                self.ext_checks[ext] = box
                grid.addWidget(box, row, col)

        buttons = QHBoxLayout()
        all_btn = QPushButton("Tümünü seç")
        all_btn.clicked.connect(lambda: self._set_all_exts(True))
        none_btn = QPushButton("Hiçbiri")
        none_btn.clicked.connect(lambda: self._set_all_exts(False))
        buttons.addWidget(all_btn)
        buttons.addWidget(none_btn)
        buttons.addStretch(1)
        grid.addLayout(buttons, len(max(formats.FORMAT_FAMILIES, key=lambda f: len(f[1]))[1]) + 1,
                       0, 1, len(formats.FORMAT_FAMILIES))
        return group

    def _set_all_exts(self, checked: bool) -> None:
        for box in self.ext_checks.values():
            box.setChecked(checked)

    def _selected_exts(self) -> list[str]:
        return [e for e, box in self.ext_checks.items() if box.isChecked()]

    def _add_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Klasör seç")
        if folder:
            self.roots_list.addItem(folder)

    def _remove_root(self) -> None:
        for item in self.roots_list.selectedItems():
            self.roots_list.takeItem(self.roots_list.row(item))

    def _accept(self) -> None:
        c = self.updated_config
        selected = self._selected_exts()
        if not selected:
            QMessageBox.warning(
                self, "Format seçilmedi",
                "En az bir dosya formatı seçmelisiniz; aksi hâlde hiçbir görsel "
                "indekslenmez.",
            )
            return
        #: Yeni format eklendiyse mevcut indeks o dosyaları içermez — arayüz
        #: bu bayrağa bakıp yeniden tarama önerir.
        self.extensions_widened = bool(set(selected) - self._initial_exts)

        c.library_roots = [self.roots_list.item(i).text() for i in range(self.roots_list.count())]
        c.extensions = selected
        c.backend = self.backend_combo.currentData()
        c.hash_algo = self.hash_algo.currentText()
        c.hash_size = self.hash_size.currentData()
        c.model_key = self.model_combo.currentData()
        c.prefer_gpu = self.gpu_check.isChecked()
        c.duplicate_hamming = self.dup_spin.value()
        c.hash_max_distance = self.maxdist_spin.value()
        c.io_workers = self.io_spin.value()
        c.cpu_batch = self.batch_spin.value()
        c.auto_tune_network = self.autotune_check.isChecked()
        c.watch_mode = self.watch_combo.currentText()
        c.rescan_interval_sec = self.rescan_spin.value()
        self.accept()
