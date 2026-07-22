# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — Desen Arama Windows masaüstü paketi (onedir).

onedir modu seçildi: tek exe (onefile) yerine bir klasör; ilk açılış daha
hızlıdır ve AV/SmartScreen ile daha az sorun çıkarır (rapordaki öneri). ONNX
modeli pakete GÖMÜLMEZ; ilk çalıştırmada indirilir (ya da models/ klasörüne
elle konur). İnternetsiz kurulum için modeli aşağıdaki `datas`'a ekleyin.
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH: bu .spec dosyasının bulunduğu dizin (packaging/). Yollar çalışma
# dizininden bağımsız olsun diye repo köküne buradan gideriz.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

datas, binaries, hiddenimports = [], [], []

# Ağır/ikili bağımlılıklar için tüm alt modül ve veri dosyalarını topla.
for pkg in ("PySide6", "onnxruntime", "faiss", "cv2", "scipy", "PIL"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += collect_submodules("desenarama")

# İsteğe bağlı: modeli çevrimdışı paketlemek için (dosyayı models/ altına koyup açın):
# datas += [("models/dinov2_vits14.onnx", "models")]

block_cipher = None

a = Analysis(
    [os.path.join(ROOT, "run_gui.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "streamlit", "notebook"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DesenArama",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI uygulaması — konsol penceresi açma
    icon=os.path.join(SPECPATH, "app.ico") if os.path.exists(os.path.join(SPECPATH, "app.ico")) else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="DesenArama",
)
