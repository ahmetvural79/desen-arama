"""Klasör tarama ve artımlı fark hesaplama.

Ağ (SMB/UNC) klasörlerinde her ``stat`` çağrısı bir ağ gidiş-dönüşüdür.
``os.scandir`` döndürdüğü ``DirEntry`` nesnelerinde stat bilgisini önbelleğe
aldığından, ``os.listdir`` + ``os.stat`` kombinasyonuna göre çok daha az ağ
trafiği üretir — bu, ağ arşivi taramasında en önemli optimizasyondur.

Tarayıcı ayrıca:
* erişim reddi / geçici ağ hatalarında dizini atlar ve loglar (taramayı düşürmez),
* uzantı filtresi uygular,
* mevcut DB durumuyla karşılaştırıp yeni / değişmiş / kayıp dosyaları ayırır.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from . import paths
from .imageio import DEFAULT_EXTENSIONS

log = logging.getLogger("desenarama.scanner")


@dataclass
class ScanEntry:
    path: str
    key: str
    mtime: float
    size: int


@dataclass
class ScanDiff:
    """Artımlı indeks için tarama farkı."""

    new: list[ScanEntry]        # DB'de olmayan yeni dosyalar
    changed: list[ScanEntry]    # mtime/size değişmiş dosyalar
    missing: list[str]          # tarama sırasında görülmeyen DB anahtarları
    unchanged: int              # değişmemiş dosya sayısı (bilgi amaçlı)

    @property
    def to_process(self) -> list[ScanEntry]:
        return self.new + self.changed


# Ağ dosya sistemlerinde mtime çözünürlüğü kaba olabilir (FAT ~2 sn). Küçük
# farkları "değişmedi" saymak gereksiz yeniden indekslemeyi önler.
MTIME_TOLERANCE = 2.0


def iter_images(root: str, extensions: set[str] | None = None, follow_symlinks: bool = False):
    """``root`` altında özyinelemeli olarak imaj dosyalarını verir (ScanEntry).

    Erişilemeyen dizinler atlanır ve loglanır; tarama kesintisiz sürer.
    """
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            it = os.scandir(paths.long_path(current))
        except (OSError, PermissionError) as e:
            log.warning("Dizin taranamadı, atlanıyor: %s (%s)", current, e)
            continue
        with it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=follow_symlinks):
                        stack.append(entry.path)
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in exts:
                        continue
                    st = entry.stat(follow_symlinks=follow_symlinks)
                    yield ScanEntry(
                        path=_clean_path(entry.path),
                        key=paths.normalize_key(entry.path),
                        mtime=st.st_mtime,
                        size=st.st_size,
                    )
                except (OSError, PermissionError) as e:
                    log.warning("Dosya atlandı: %s (%s)", getattr(entry, "path", "?"), e)
                    continue


def _clean_path(path: str) -> str:
    """Uzun-yol ön ekini görüntüleme/DB için geri çıkarır."""
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def diff(roots: list[str], existing: dict[str, tuple[float, int]],
         extensions: set[str] | None = None) -> ScanDiff:
    """Bir veya daha fazla kök klasörü tarayıp DB durumuyla farkı hesaplar."""
    new: list[ScanEntry] = []
    changed: list[ScanEntry] = []
    seen: set[str] = set()
    remaining = dict(existing)

    for root in roots:
        for entry in iter_images(root, extensions):
            if entry.key in seen:
                continue
            seen.add(entry.key)
            prev = remaining.pop(entry.key, None)
            if prev is None:
                new.append(entry)
            else:
                prev_mtime, prev_size = prev
                if entry.size != prev_size or abs(entry.mtime - prev_mtime) > MTIME_TOLERANCE:
                    changed.append(entry)

    missing = list(remaining.keys())
    unchanged = len(seen) - len(new) - len(changed)
    return ScanDiff(new=new, changed=changed, missing=missing, unchanged=unchanged)
