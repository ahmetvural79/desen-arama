"""GUI giriş noktası."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from .. import __app_name__, config as cfg_mod
from ..core import paths
from ..services.engine import Engine


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(paths.logs_dir() / "desenarama.log", encoding="utf-8")],
    )
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(__app_name__)

    engine = Engine(cfg_mod.AppConfig.load())
    engine.open()

    from .main_window import MainWindow

    win = MainWindow(engine)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
