"""Windows ve ağ (UNC) farkındalı yol yardımcıları.

Bu modülün temel görevleri:

* Türetilmiş verinin (SQLite, FAISS, thumbnail, model) yazılacağı yerel
  veri dizinini bulmak. Windows'ta ``%LOCALAPPDATA%\\DesenArama``; diğer
  platformlarda (geliştirme/test) ``~/.local/share/DesenArama`` benzeri
  bir yer. **Türetilmiş veri asla ağ paylaşımına yazılmaz.**
* UNC yollarını (``\\\\sunucu\\paylasim\\...``) ve 260 karakter (MAX_PATH)
  sınırını aşan uzun yolları güvenle işlemek. Windows'ta ``\\\\?\\`` uzun
  yol ön eki uygulanır; UNC için ``\\\\?\\UNC\\sunucu\\paylasim\\`` biçimi
  kullanılır.
* Bir yolun ağ paylaşımı olup olmadığını tahmin ederek indeksleyicinin
  eşzamanlılık ve olay-izleme stratejisini ayarlamasına yardımcı olmak.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .. import __app_id__

IS_WINDOWS = os.name == "nt"


# --------------------------------------------------------------------------- #
# Yerel veri dizini
# --------------------------------------------------------------------------- #
def data_dir() -> Path:
    """Türetilmiş verinin tutulacağı yerel dizini döndürür (yoksa oluşturur).

    * Windows: ``%LOCALAPPDATA%\\DesenArama``
    * macOS:   ``~/Library/Application Support/DesenArama``
    * Linux:   ``$XDG_DATA_HOME/DesenArama`` veya ``~/.local/share/DesenArama``

    Ortam değişkeni ``DESENARAMA_DATA_DIR`` ile geçersiz kılınabilir
    (test ve taşınabilir kurulum için).
    """
    override = os.environ.get("DESENARAMA_DATA_DIR")
    if override:
        base = Path(override)
    elif IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / __app_id__
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / __app_id__
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".local" / "share") / __app_id__
    base.mkdir(parents=True, exist_ok=True)
    return base


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumbs_dir() -> Path:
    d = data_dir() / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# UNC / uzun yol işleme
# --------------------------------------------------------------------------- #
def is_unc(path: str) -> bool:
    """Yol bir UNC ağ yolu mu? (``\\\\sunucu\\paylasim`` biçimi)."""
    p = str(path)
    return p.startswith("\\\\") or p.startswith("//")


def is_network_path(path: str) -> bool:
    """Yolun ağ paylaşımı olma olasılığını tahmin eder.

    * UNC yolları kesinlikle ağdır.
    * Windows'ta eşlenmiş sürücü harfleri (Z: vb.) da ağ olabilir; bunu
      kesin bilmek ``GetDriveType`` gerektirir. Kütüphane hafif kalsın
      diye burada yalnızca UNC'yi ağ kabul ederiz, sürücü harfi kontrolü
      :func:`drive_is_remote` içinde opsiyoneldir.
    """
    if is_unc(path):
        return True
    if IS_WINDOWS:
        return drive_is_remote(path)
    return False


def drive_is_remote(path: str) -> bool:
    """Windows'ta bir sürücü harfinin ağ sürücüsü olup olmadığını kontrol eder.

    Windows dışında her zaman ``False`` döner. Hata durumunda güvenli
    tarafta kalıp ``False`` döner (yerel varsayımı taramayı bozmaz).
    """
    if not IS_WINDOWS:
        return False
    try:
        import ctypes

        drive = os.path.splitdrive(os.path.abspath(path))[0]
        if not drive:
            return False
        if not drive.endswith("\\"):
            drive += "\\"
        # DRIVE_REMOTE == 4
        return ctypes.windll.kernel32.GetDriveTypeW(drive) == 4
    except Exception:
        return False


def long_path(path: str) -> str:
    """MAX_PATH sınırını aşan Windows yolları için ``\\\\?\\`` ön eki uygular.

    * Windows dışında yol olduğu gibi döner.
    * Zaten ``\\\\?\\`` ile başlayan yollara dokunulmaz.
    * UNC yolları ``\\\\?\\UNC\\sunucu\\paylasim\\...`` biçimine çevrilir.
    * Göreli yollar önce mutlaklaştırılır (uzun yol ön eki mutlak yol ister).

    Ağ paylaşımlarındaki derin klasör ağaçları 260 karakteri kolayca
    aşabildiğinden, dosya açmadan hemen önce bu fonksiyon kullanılmalıdır.
    """
    if not IS_WINDOWS:
        return path
    p = str(path)
    if p.startswith("\\\\?\\"):
        return p
    # Mutlaklaştır ve ayırıcıları normalize et
    p = os.path.abspath(p)
    if p.startswith("\\\\"):
        # UNC: \\sunucu\paylasim -> \\?\UNC\sunucu\paylasim
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def open_binary(path: str):
    """Bir dosyayı ikili okuma için açar; Windows'ta uzun yol ön ekini uygular.

    Ağdaki derin/uzun yollarda ``FileNotFoundError`` alınmasını önler.
    """
    return open(long_path(path), "rb")


def normalize_key(path: str) -> str:
    """DB anahtarı olarak kullanılacak kararlı yol biçimi.

    Windows'ta dosya sistemi büyük/küçük harf duyarsız olduğundan aynı
    dosyanın farklı yazımlarla iki kez indekslenmesini önlemek için yol
    normalize edilir. Ağ paylaşımı adları da tutarlı kalsın diye ters
    eğik çizgi standardı korunur.
    """
    p = os.path.normpath(str(path))
    if IS_WINDOWS:
        # Büyük/küçük harf farkını gidermek için normcase; ayrıca \\?\ önekini at
        if p.startswith("\\\\?\\UNC\\"):
            p = "\\\\" + p[8:]
        elif p.startswith("\\\\?\\"):
            p = p[4:]
        p = os.path.normcase(p)
    return p
