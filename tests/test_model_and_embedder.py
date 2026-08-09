"""AI (embedding) arka ucu testleri — model kaydı, indirme, ön işleme, kalibrasyon.

v1.0'da AI modu pratikte hiç çalışmıyordu:

* ``dinov2-base`` indirme adresi mevcut değildi (HTTP 401),
* model bulunamayınca **sessizce** çok daha zayıf bir yedek çıkarıcıya
  düşülüyordu ve kullanıcı bunu öğrenmiyordu,
* SHA256 doğrulaması tanımsızdı, yarım/bozuk dosya kabul ediliyordu.

Bu testler üçünü de kapatır. Gerçek model dosyası gerektirmezler (ağ erişimi
olmayan CI'da da koşarlar).
"""

from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pytest

from desenarama.core import embedder as emb_mod
from desenarama.core import models, paths


# --------------------------------------------------------------------------- #
# Model kaydı
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(models.REGISTRY))
def test_every_model_has_verifiable_metadata(key):
    """Her kayıt indirilebilir ve doğrulanabilir olmalı."""
    spec = models.REGISTRY[key]
    assert spec.urls, "en az bir kaynak gerekli"
    assert all(u.startswith("https://") for u in spec.urls)
    assert len(spec.sha256) == 64, "SHA256 zorunlu (v1.0'da tanımsızdı)"
    assert spec.size_bytes > 1_000_000
    assert spec.dim in (384, 768)


def test_base_model_no_longer_points_at_dead_repo():
    """Regresyon: v1.0'ın dinov2-base adresi mevcut değildi (HTTP 401)."""
    assert all("sefaburak/dinov2-base-onnx" not in u
               for u in models.REGISTRY["dinov2-base"].urls)


# --------------------------------------------------------------------------- #
# İndirme ve bütünlük
# --------------------------------------------------------------------------- #
class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _spec_for(payload: bytes, urls=("https://ornek/model.onnx",)) -> models.ModelSpec:
    return models.ModelSpec(
        key="test", filename="test_model.onnx", urls=urls, dim=384,
        input_size=224, size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


@pytest.fixture
def fake_net(monkeypatch):
    """``urlopen``'ı taklit eder; URL -> yük eşlemesi verilir."""
    def install(mapping: dict[str, bytes | Exception]):
        def fake_urlopen(req, timeout=None):
            value = mapping[req.full_url]
            if isinstance(value, Exception):
                raise value
            return _FakeResponse(value)
        monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    return install


def test_download_verifies_and_stores(fresh_env, fake_net):
    payload = b"ONNX" * 500
    spec = _spec_for(payload)
    fake_net({spec.urls[0]: payload})

    path = models.download(spec)
    assert os.path.exists(path)
    assert models.verify(spec, path)
    assert models.is_available(spec)


def test_download_rejects_corrupted_payload(fresh_env, fake_net):
    """Vekil sunucunun döndürdüğü HTML hata sayfası model sanılmamalı."""
    spec = _spec_for(b"ONNX" * 500)
    fake_net({spec.urls[0]: b"<html>403 Forbidden</html>"})

    with pytest.raises(models.ModelDownloadError):
        models.download(spec)
    # Bozuk dosya diskte bırakılmamalı
    assert not os.path.exists(models.local_path(spec))


def test_download_falls_back_to_mirror(fresh_env, fake_net):
    payload = b"ONNX" * 500
    spec = _spec_for(payload, urls=("https://birincil/m.onnx", "https://ayna/m.onnx"))
    fake_net({
        "https://birincil/m.onnx": OSError("ağ engellendi"),
        "https://ayna/m.onnx": payload,
    })

    assert models.verify(spec, models.download(spec))


def test_download_error_mentions_offline_install(fresh_env, fake_net):
    """Kurumsal ağ engelinde kullanıcı ne yapacağını bilmeli."""
    spec = _spec_for(b"ONNX" * 500)
    fake_net({spec.urls[0]: OSError("engellendi")})

    with pytest.raises(models.ModelDownloadError) as err:
        models.download(spec)
    assert str(paths.models_dir()) in str(err.value)
    assert spec.filename in str(err.value)


def test_existing_corrupt_file_is_replaced(fresh_env, fake_net):
    payload = b"ONNX" * 500
    spec = _spec_for(payload)
    with open(models.local_path(spec), "wb") as f:
        f.write(b"yarim indirme")
    fake_net({spec.urls[0]: payload})

    assert models.verify(spec, models.download(spec))


def test_download_can_be_cancelled(fresh_env, fake_net):
    spec = _spec_for(b"ONNX" * 500)
    fake_net({spec.urls[0]: b"ONNX" * 500})

    with pytest.raises(models.ModelDownloadError):
        models.download(spec, cancel=lambda: True)
    assert not os.path.exists(models.local_path(spec))


# --------------------------------------------------------------------------- #
# Sessiz fallback kaldırıldı
# --------------------------------------------------------------------------- #
def test_missing_model_raises_instead_of_silent_fallback():
    """Regresyon: v1.0 burada sessizce FallbackEmbedder döndürüyordu."""
    with pytest.raises(emb_mod.EmbedderUnavailable):
        emb_mod.load_embedder(None)


def test_broken_model_file_raises(tmp_path):
    bad = tmp_path / "bozuk.onnx"
    bad.write_bytes(b"bu bir onnx dosyasi degil")
    with pytest.raises(emb_mod.EmbedderUnavailable):
        emb_mod.load_embedder(str(bad))


def test_fallback_requires_explicit_opt_in():
    """Yedek çıkarıcı yalnızca çevrimdışı test/CI için, açık tercihle."""
    e = emb_mod.load_embedder(None, allow_fallback=True)
    assert isinstance(e, emb_mod.FallbackEmbedder)


def test_error_message_tells_user_what_to_do():
    with pytest.raises(emb_mod.EmbedderUnavailable) as err:
        emb_mod.load_embedder(None)
    assert "hash" in str(err.value).lower()


# --------------------------------------------------------------------------- #
# Ön işleme (letterbox)
# --------------------------------------------------------------------------- #
def test_preprocess_shape_and_range():
    rgb = np.full((300, 500, 3), 128, np.uint8)
    t = emb_mod.preprocess(rgb, size=224)
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == np.float32


def test_preprocess_preserves_borders():
    """Regresyon: merkez kırpma halının bordürünü tamamen atıyordu.

    Kenarları belirgin bir görselde, letterbox sonrası kenar bilgisi tensörde
    kalmalı — yani kenar bölgesi merkezden ayırt edilebilir olmalı.
    """
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[:40, :] = rgb[-40:, :] = 255      # üst/alt bordür
    rgb[:, :40] = rgb[:, -40:] = 255
    t = emb_mod.preprocess(rgb, size=224)[0, 0]
    kenar = t[:12, :].mean()
    merkez = t[100:124, 100:124].mean()
    assert kenar > merkez + 1.0, "bordür ön işlemede kayboldu"


def test_preprocess_keeps_aspect_ratio():
    """Geniş bir görselin içeriği ezilmemeli; dolgu simetrik olmalı."""
    rgb = np.full((100, 400, 3), 200, np.uint8)
    t = emb_mod.preprocess(rgb, size=224)[0, 0]
    ust_dolgu = t[:20, :].mean()
    alt_dolgu = t[-20:, :].mean()
    assert ust_dolgu == pytest.approx(alt_dolgu, abs=1e-4)


# --------------------------------------------------------------------------- #
# Kütüphaneye göre skor kalibrasyonu
# --------------------------------------------------------------------------- #
def test_cosine_calibration_against_library_baseline():
    """DINOv2 kosinüsleri dar ve yüksek bir bantta yaşar.

    Ölçülen gerçek değerler (124 görsellik halı arşivi, DINOv2 ViT-S/14):
    kütüphane medyanı 0.878, birebir eşleşme 1.000, aynı desen farklı renk
    0.950. Eski ``(cos+1)/2`` eşlemesi bunları 0.939 / 1.000 / 0.975'e
    sıkıştırıyordu — yani her şey "%95 benzer" görünüyordu.
    """
    from desenarama.services.search import _cosine_to_score

    baseline = 0.8778
    assert _cosine_to_score(0.8771, baseline) == pytest.approx(0.0, abs=0.01)
    assert _cosine_to_score(1.0, baseline) == pytest.approx(1.0)
    orta = _cosine_to_score(0.9503, baseline)
    assert 0.4 < orta < 0.8, "aynı desen/farklı renk ayırt edilebilir olmalı"


def test_calibration_is_skipped_without_baseline():
    """Küçük kütüphanede taban güvenilir değildir; ham ölçek kullanılır."""
    from desenarama.services.search import _cosine_to_score

    assert _cosine_to_score(0.9, 0.0) == pytest.approx(0.9)


def test_calibration_handles_degenerate_baseline():
    """Taban 1'e dayanırsa sıfıra bölme olmamalı."""
    from desenarama.services.search import _cosine_to_score

    assert _cosine_to_score(1.0, 1.0) == 1.0
    assert _cosine_to_score(0.5, 1.0) == 0.0
