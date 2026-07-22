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
    url: str
    dim: int
    input_size: int
    sha256: str | None = None  # doğrulama için (opsiyonel)
    note: str = ""


# Bilinen modeller. URL'ler HuggingFace "resolve" bağlantılarıdır; kurum ağı
# bunları engelliyorsa dosya elle models/ dizinine bırakılabilir.
REGISTRY: dict[str, ModelSpec] = {
    "dinov2-small": ModelSpec(
        key="dinov2-small",
        filename="dinov2_vits14.onnx",
        url="https://huggingface.co/sefaburak/dinov2-small-onnx/resolve/main/dinov2_vits14.onnx",
        dim=384,
        input_size=224,
        note="DINOv2 ViT-S/14, 384-dim, Apache 2.0",
    ),
    "dinov2-base": ModelSpec(
        key="dinov2-base",
        filename="dinov2_vitb14.onnx",
        url="https://huggingface.co/sefaburak/dinov2-base-onnx/resolve/main/dinov2_vitb14.onnx",
        dim=768,
        input_size=224,
        note="DINOv2 ViT-B/14, 768-dim, Apache 2.0",
    ),
}

DEFAULT_MODEL = "dinov2-small"


def local_path(spec: ModelSpec) -> str:
    return str(paths.models_dir() / spec.filename)


def is_available(spec: ModelSpec) -> bool:
    return os.path.exists(local_path(spec))


def resolve(key: str | None) -> ModelSpec:
    return REGISTRY[key or DEFAULT_MODEL]


def download(spec: ModelSpec, progress=None, timeout: int = 60) -> str:
    """Modeli yerel dizine indirir; zaten varsa mevcut yolu döndürür.

    ``progress(indirilen_bayt, toplam_bayt)`` opsiyonel geri çağrımı UI'de
    ilerleme çubuğu için kullanılabilir.
    """
    dest = local_path(spec)
    if os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    log.info("Model indiriliyor: %s -> %s", spec.url, dest)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "DesenArama/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        h = hashlib.sha256()
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)
    if spec.sha256 and h.hexdigest() != spec.sha256:
        os.remove(tmp)
        raise RuntimeError("Model SHA256 doğrulaması başarısız")
    os.replace(tmp, dest)
    log.info("Model indirildi: %s (%d bayt)", dest, downloaded)
    return dest
