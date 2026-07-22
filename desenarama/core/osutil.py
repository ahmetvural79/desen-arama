"""İşletim sistemi entegrasyonu — dosya açma ve dosya yöneticisinde gösterme.

Windows birincil hedeftir (Explorer ``/select`` ile dosyayı seçili açar); macOS
ve Linux geliştirme/taşınabilirlik için desteklenir.
"""

from __future__ import annotations

import os
import subprocess
import sys

from . import paths


def open_path(path: str) -> None:
    """Dosyayı/klasörü varsayılan uygulamayla açar."""
    p = paths.long_path(path) if paths.IS_WINDOWS else path
    if paths.IS_WINDOWS:
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def reveal_in_folder(path: str) -> None:
    """Dosyayı içeren klasörü açar ve dosyayı seçili gösterir."""
    if paths.IS_WINDOWS:
        # Explorer'ın /select argümanı normal (uzun-yol olmayan) biçim ister.
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        folder = os.path.dirname(path) or "."
        subprocess.Popen(["xdg-open", folder])
