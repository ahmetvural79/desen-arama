"""Format kapsaması testleri.

v1.0'da BMP dosyaları hiç indekslenmiyordu: okuma katmanı ve arayüz BMP'yi
destekliyordu, ancak tarayıcı filtresi ``.jpg/.jpeg/.png`` ile sınırlıydı.
Bu testler o regresyonu kalıcı olarak kapatır — desteklenen **her** uzantı
gerçek bir dosyayla yazılıp okunur ve uçtan uca indekslenip aranır.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from desenarama import config as cfg_mod
from desenarama.core import formats, imageio, scanner

# Pillow'un yazma adı her uzantı için türetilemez; açıkça eşleriz.
_WRITE_FORMAT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".jpe": "JPEG", ".jfif": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP", ".dib": "DIB",
    ".webp": "WEBP",
    ".tif": "TIFF", ".tiff": "TIFF",
    ".tga": "TGA",
}


def _write(path, ext, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path), _WRITE_FORMAT[ext])
    return arr


def test_every_supported_extension_has_a_writer():
    """Kayıt listesi ile test eşlemesi ayrışmasın."""
    assert set(formats.SUPPORTED_EXTENSIONS) == set(_WRITE_FORMAT)


@pytest.mark.parametrize("ext", formats.SUPPORTED_EXTENSIONS)
def test_supported_extension_is_readable(tmp_path, ext):
    """Desteklenen her uzantı gerçekten okunabilmeli (Pillow desteği doğrulaması)."""
    path = tmp_path / f"desen{ext}"
    _write(path, ext)
    loaded = imageio.load(str(path))
    assert (loaded.width, loaded.height) == (64, 48)
    assert loaded.rgb.shape == (48, 64, 3)


@pytest.mark.parametrize("ext", formats.SUPPORTED_EXTENSIONS)
def test_scanner_finds_every_supported_extension(tmp_path, ext):
    """Tarayıcı varsayılan filtresi desteklenen her formatı görmeli."""
    _write(tmp_path / f"desen{ext}", ext)
    found = list(scanner.iter_images(str(tmp_path)))
    assert [f.path.endswith(ext) for f in found] == [True], f"{ext} taranmadı"


def test_bmp_is_indexed_by_default(tmp_path):
    """Regresyon: BMP varsayılan yapılandırmayla taranmalı."""
    _write(tmp_path / "hali.bmp", ".bmp")
    cfg = cfg_mod.AppConfig()
    found = list(scanner.iter_images(str(tmp_path), cfg.ext_set()))
    assert len(found) == 1 and found[0].path.endswith(".bmp")


def test_scanner_ignores_unsupported_extension(tmp_path):
    (tmp_path / "notlar.txt").write_text("desen değil", encoding="utf-8")
    assert list(scanner.iter_images(str(tmp_path))) == []


@pytest.mark.parametrize("written,configured", [
    (".JPG", ".jpg"),      # dosya adı büyük harfli
    (".bmp", "bmp"),       # ayarda nokta unutulmuş
    (".png", ".PNG"),      # ayarda büyük harfli
])
def test_extension_matching_is_forgiving(tmp_path, written, configured):
    """Büyük/küçük harf ve eksik nokta sessizce eşleşmemeye yol açmamalı."""
    _write(tmp_path / f"desen{written}", written.lower())
    found = list(scanner.iter_images(str(tmp_path), {configured}))
    assert len(found) == 1


# --------------------------------------------------------------------------- #
# Yapılandırma göçü
# --------------------------------------------------------------------------- #
def test_fresh_config_includes_bmp():
    cfg = cfg_mod.AppConfig()
    assert ".bmp" in cfg.ext_set()
    assert cfg.ext_set() == set(formats.SUPPORTED_EXTENSIONS)


def test_migration_widens_legacy_extension_list(fresh_env):
    """v1.0 yapılandırması (jpg/jpeg/png) yeni varsayılana yükseltilmeli."""
    import json

    path = fresh_env / "config.json"
    path.write_text(json.dumps({"extensions": [".jpg", ".jpeg", ".png"],
                                "library_roots": ["C:/desenler"]}), encoding="utf-8")

    cfg = cfg_mod.AppConfig.load()
    assert ".bmp" in cfg.ext_set()
    assert cfg.library_roots == ["C:/desenler"]      # diğer ayarlar korunur
    assert cfg.config_version == cfg_mod.CONFIG_VERSION
    assert cfg.migration_notes                        # arayüze bildirilir

    # Göç diske yazılmalı ve tekrar çalışmamalı
    again = cfg_mod.AppConfig.load()
    assert not again.migration_notes


def test_migration_respects_user_customized_list(fresh_env):
    """Kullanıcı listeyi bilerek daralttıysa (eski varsayılan değilse) dokunma."""
    import json

    (fresh_env / "config.json").write_text(
        json.dumps({"extensions": [".png"]}), encoding="utf-8")

    cfg = cfg_mod.AppConfig.load()
    assert cfg.ext_set() == {".png"}
    assert not cfg.migration_notes


def test_qt_name_filter_covers_bmp():
    assert "*.bmp" in formats.qt_name_filter()
