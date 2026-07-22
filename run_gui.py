"""Masaüstü uygulama giriş noktası (PyInstaller bunu paketler)."""

import multiprocessing

from desenarama.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows'ta donmuş exe için gerekli
    raise SystemExit(main())
