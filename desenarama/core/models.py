"""DINOv2 ONNX model kaydı ve otomatik indirme.

Modeller yalnızca AI modu seçildiğinde ve yerelde yoksa indirilir; indirilen
dosya yerel ``models/`` dizinine (``%LOCALAPPDATA%\\DesenArama\\models``)
yazılır. İnternet erişimi olmayan kurulumlar için model dosyası elle bu dizine
kopyalanabilir. Ticari kullanım için DINOv2 (Apache 2.0) tercih edilir;
DINOv3'ün özel lisansı ayrıca değerlendirilmelidir.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass

from . import paths

log = logging.getLogger("desenarama.models")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    urls: tuple[str, ...]     # sırayla denenir (ayna desteği)
    dim: int
    input_size: int
    size_bytes: int           # beklenen dosya boyutu (yarım indirmeyi yakalar)
    sha256: str               # zorunlu bütünlük doğrulaması
    note: str = ""

    @property
    def url(self) -> str:
        """Birincil kaynak (geriye dönük uyumluluk / günlük mesajları için)."""
        return self.urls[0]


def _hf(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


# Bilinen modeller. Kurum ağı HuggingFace'i engelliyorsa dosya elle
# ``models/`` dizinine ``filename`` adıyla bırakılabilir.
#
# ``sha256`` değerleri HuggingFace'in LFS ``oid``'inden alınmıştır (LFS oid'i
# dosyanın sha256'sıdır) ve indirilen dosyayla doğrulanmıştır.
REGISTRY: dict[str, ModelSpec] = {
    "dinov2-small": ModelSpec(
        key="dinov2-small",
        filename="dinov2_vits14.onnx",
        # Birincil: onnx-community (transformers dışa aktarımı, dinamik batch ve
        # dinamik girdi çözünürlüğü). Ayna: sefaburak (sabit dışa aktarım).
        urls=(_hf("onnx-community/dinov2-small", "onnx/model.onnx"),),
        dim=384,
        input_size=224,
        size_bytes=88_532_934,
        sha256="f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
        note="DINOv2 ViT-S/14, 384-dim, Apache 2.0",
    ),
    "dinov2-base": ModelSpec(
        key="dinov2-base",
        filename="dinov2_vitb14.onnx",
        # v1.0'daki ``sefaburak/dinov2-base-onnx`` deposu **mevcut değil**
        # (HTTP 401): bu seçenek her zaman başarısız olup sessizce zayıf
        # fallback embedder'a düşüyordu.
        urls=(_hf("onnx-community/dinov2-base", "onnx/model.onnx"),),
        dim=768,
        input_size=224,
        size_bytes=346_627_111,
        sha256="320d1012a6fc65b101fc85ca30ee7a47b2e4f6a2e8bd78fb9d7036def0e30cb0",
        note="DINOv2 ViT-B/14, 768-dim, Apache 2.0",
    ),
}

DEFAULT_MODEL = "dinov2-small"


def local_path(spec: ModelSpec) -> str:
    return str(paths.models_dir() / spec.filename)


def is_available(spec: ModelSpec) -> bool:
    """Model yerelde ve makul görünüyor mu?

    Yalnızca boyut kontrolü yapılır: SHA256'yı her açılışta hesaplamak 88–350 MB
    dosyada gereksiz gecikme yaratır. Tam doğrulama indirme sırasında
    (:func:`verify`) yapılır.
    """
    try:
        return os.path.getsize(local_path(spec)) == spec.size_bytes
    except OSError:
        return False


def resolve(key: str | None) -> ModelSpec:
    return REGISTRY[key or DEFAULT_MODEL]


class ModelDownloadError(RuntimeError):
    """Model indirilemedi (ağ engeli, bozuk dosya, disk dolu…)."""


def verify(spec: ModelSpec, path: str | None = None) -> bool:
    """Yerel model dosyasının boyutunu ve SHA256'sını doğrular.

    Yarım kalmış indirmeler ve kurum proxy'sinin döndürdüğü HTML hata sayfaları
    v1.0'da sessizce "model dosyası" olarak kabul edilebiliyordu; bu kontrol
    onları yakalar. Boyut önce bakılır (ucuz), sonra hash.
    """
    p = path or local_path(spec)
    try:
        if os.path.getsize(p) != spec.size_bytes:
            return False
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest() == spec.sha256
    except OSError:
        return False


def download(spec: ModelSpec, progress=None, timeout: int = 60,
             cancel=None) -> str:
    """Modeli yerel dizine indirir; zaten varsa mevcut yolu döndürür.

    ``progress(indirilen_bayt, toplam_bayt)`` arayüzdeki ilerleme çubuğu için,
    ``cancel()`` ise kullanıcının indirmeyi durdurabilmesi için çağrılır.
    Kayıtlı her ayna sırayla denenir; hepsi başarısız olursa
    :class:`ModelDownloadError` yükseltilir — **sessizce zayıf bir yedeğe
    düşülmez**.
    """
    dest = local_path(spec)
    if os.path.exists(dest):
        if verify(spec, dest):
            return dest
        log.warning("Yerel model bozuk/eksik, yeniden indiriliyor: %s", dest)
        os.remove(dest)

    errors: list[str] = []
    for url in spec.urls:
        try:
            _download_one(spec, url, dest, progress, timeout, cancel)
            log.info("Model indirildi ve doğrulandı: %s", dest)
            return dest
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning("Model kaynağı başarısız (%s): %s", url, e)
            errors.append(f"{url}: {e}")
    raise ModelDownloadError(
        "Model indirilemedi. Denenen kaynaklar:\n  " + "\n  ".join(errors)
        + f"\n\nÇevrimdışı kurulum: dosyayı '{spec.filename}' adıyla şu dizine "
          f"kopyalayın:\n  {paths.models_dir()}"
    )


def _download_one(spec: ModelSpec, url: str, dest: str, progress, timeout, cancel) -> None:
    tmp = dest + ".part"
    log.info("Model indiriliyor: %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "DesenArama/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or spec.size_bytes
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    if cancel is not None and cancel():
                        raise ModelDownloadError("İndirme kullanıcı tarafından iptal edildi")
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total)
        if not verify(spec, tmp):
            raise ModelDownloadError(
                "İndirilen dosya doğrulanamadı (boyut/SHA256 uyuşmuyor) — "
                "ağ üzerinde bir vekil sunucu içeriği değiştirmiş olabilir"
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
