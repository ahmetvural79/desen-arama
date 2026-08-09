"""Skor kalibrasyonu regresyon testleri.

v1.0'da benzerlik skoru ``1 - d/bits`` idi. Algısal hash'lerde alakasız iki
görselin beklenen mesafesi ``bits/2`` olduğundan, **tamamen alakasız görseller
%50 skor alıyordu**; sıralama bozuluyor ve skor eşiği anlamsızlaşıyordu.

Bu dosya hem formülü hem de gerçek görseller üzerindeki uçtan uca davranışı
sabitler.
"""

from __future__ import annotations

import numpy as np
import pytest

from desenarama import config as cfg_mod
from desenarama.core import hasher, hashindex


# --------------------------------------------------------------------------- #
# Formül
# --------------------------------------------------------------------------- #
def test_identical_scores_one():
    assert hasher.similarity(0, 64) == 1.0


def test_unrelated_scores_zero():
    """Alakasız görsellerin beklenen mesafesi (bits/2) tam olarak 0 vermeli."""
    assert hasher.similarity(32, 64) == 0.0
    assert hasher.similarity(40, 64) == 0.0  # tabanın ötesi kırpılır


def test_score_is_monotonic_and_bounded():
    scores = [hasher.similarity(d, 64) for d in range(0, 64)]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_calibration_is_hash_size_independent():
    """Aynı *oransal* mesafe, 64-bit ve 256-bit hash'te aynı skoru vermeli."""
    assert hasher.similarity(8, 64) == pytest.approx(hasher.similarity(32, 256))


def test_scale_distance():
    assert hasher.scale_distance(22, 64) == 22
    assert hasher.scale_distance(22, 256) == 88


def test_hash_index_uses_calibrated_similarity():
    idx = hashindex.HashIndex(hash_bits=64)
    idx.add(1, 0)                       # birebir
    idx.add(2, (1 << 32) - 1)           # 32 bit farklı → alakasız
    by_id = {r.image_id: r for r in idx.search(0, k=10)}
    assert by_id[1].similarity == 1.0
    assert by_id[2].similarity == 0.0


# --------------------------------------------------------------------------- #
# Varsayılanlar
# --------------------------------------------------------------------------- #
def test_default_thresholds_admit_recolored_and_cropped_variants():
    """Eski varsayılan (12) aynı desenin varyantlarını eliyordu."""
    cfg = cfg_mod.AppConfig()
    assert cfg.effective_max_distance() >= 20
    assert cfg.score_threshold > 0.0


def test_thresholds_scale_with_hash_size():
    cfg = cfg_mod.AppConfig(hash_size=16)
    assert cfg.hash_bits() == 256
    assert cfg.effective_max_distance() == 88
    assert cfg.effective_duplicate_hamming() == 32


def test_unlimited_max_distance_is_preserved():
    cfg = cfg_mod.AppConfig(hash_max_distance=0)
    assert cfg.effective_max_distance() == 0


def test_migration_fixes_legacy_thresholds(fresh_env):
    """v1.0 eşikleri yeni skor ölçeğinde yanlış davranır; taşınmalı."""
    import json

    (fresh_env / "config.json").write_text(json.dumps({
        "hash_max_distance": 12, "score_threshold": 0.0,
    }), encoding="utf-8")

    cfg = cfg_mod.AppConfig.load()
    assert cfg.hash_max_distance == 22
    assert cfg.score_threshold == 0.35
    assert len(cfg.migration_notes) >= 2


def test_migration_keeps_customized_thresholds(fresh_env):
    import json

    (fresh_env / "config.json").write_text(json.dumps({
        "hash_max_distance": 30, "score_threshold": 0.6,
    }), encoding="utf-8")

    cfg = cfg_mod.AppConfig.load()
    assert cfg.hash_max_distance == 30
    assert cfg.score_threshold == 0.6


def test_threshold_migration_does_not_trigger_reindex(fresh_env):
    """Eşik değişikliği yeniden tarama gerektirmez — kullanıcıyı boşuna meşgul etme."""
    import json

    (fresh_env / "config.json").write_text(json.dumps({
        "extensions": [".png"],          # özelleştirilmiş → genişletilmeyecek
        "hash_max_distance": 12,
    }), encoding="utf-8")

    cfg = cfg_mod.AppConfig.load()
    assert cfg.migration_notes
    assert cfg.migration_needs_reindex is False


def test_extension_migration_triggers_reindex(fresh_env):
    import json

    (fresh_env / "config.json").write_text(
        json.dumps({"extensions": [".jpg", ".jpeg", ".png"]}), encoding="utf-8")

    cfg = cfg_mod.AppConfig.load()
    assert cfg.migration_needs_reindex is True


# --------------------------------------------------------------------------- #
# Embedding tarafı
# --------------------------------------------------------------------------- #
def test_cosine_score_mapping():
    from desenarama.services.search import _cosine_to_score

    assert _cosine_to_score(1.0) == 1.0
    assert _cosine_to_score(0.0) == 0.0     # eski eşleme burada 0.5 veriyordu
    assert _cosine_to_score(-0.4) == 0.0
    assert _cosine_to_score(0.8) == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Renk harmanlaması
# --------------------------------------------------------------------------- #
def test_color_never_outranks_pattern():
    """Regresyon: renk terimi sıralamayı tersine çeviremez.

    Ölçülen gerçek durum: aynı desenin farklı renkli varyantı desen=1.00 /
    renk=0.00 alırken, alakasız ama benzer paletli bir görsel desen=0.63 /
    renk=0.90 alıyordu. Eski toplamsal harmanda (α=0.2) alakasız olan öne
    geçiyordu.
    """
    from desenarama.services.search import _blend

    same_pattern_other_color = _blend(1.00, 0.00, 0.2)
    unrelated_same_palette = _blend(0.63, 0.90, 0.2)
    assert same_pattern_other_color > unrelated_same_palette


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2, 0.5, 1.0])
def test_blend_is_bounded_and_monotonic_in_pattern(alpha):
    from desenarama.services.search import _blend

    assert 0.0 <= _blend(1.0, 0.0, alpha) <= 1.0
    assert _blend(0.9, 0.5, alpha) > _blend(0.4, 0.5, alpha)


def test_blend_alpha_zero_is_pattern_only():
    from desenarama.services.search import _blend

    assert _blend(0.7, 0.0, 0.0) == 0.7


def test_blend_without_color_data_is_neutral():
    """Renk bilgisi olmayan eski kayıtlar cezalandırılmamalı."""
    from desenarama.services.search import _blend

    assert _blend(0.7, None, 0.5) == 0.7


# --------------------------------------------------------------------------- #
# Kalite bandı
# --------------------------------------------------------------------------- #
def _result(score, dup=False):
    from desenarama.services.search import SearchResult

    return SearchResult(image_id=1, path="x", thumb_path=None, score=score,
                        pattern_score=score, color_sim=0.0, hamming=0 if dup else 30,
                        is_duplicate=dup, width=1, height=1, below_threshold=False)


@pytest.mark.parametrize("score,expected", [
    (0.95, "Çok benzer"), (0.60, "Benzer"), (0.40, "Zayıf"), (0.10, "Çok zayıf"),
])
def test_quality_bands(score, expected):
    assert _result(score).quality() == expected


def test_duplicate_band_wins():
    assert _result(0.9, dup=True).quality() == "Kopya"


# --------------------------------------------------------------------------- #
# Uçtan uca: gerçek görsellerle sıralama
# --------------------------------------------------------------------------- #
def _pattern(seed: int, hue_shift: float = 0.0, size: int = 512) -> np.ndarray:
    """Halıya benzer sentetik desen: tekrarlayan motif + bordür."""
    import cv2

    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    m = np.zeros((size, size), np.float32)
    for _ in range(6):
        fx, fy = r.uniform(4, 14, 2)
        m += (np.sin(2 * np.pi * fx * x / size + r.uniform(0, 6))
              * np.sin(2 * np.pi * fy * y / size + r.uniform(0, 6)))
    m = (m - m.min()) / (np.ptp(m) + 1e-9)
    b = 40
    m[:b, :] = m[-b:, :] = 0.9
    m[:, :b] = m[:, -b:] = 0.9
    h = np.full((size, size), int(((0.05 + hue_shift) % 1.0) * 179), np.uint8)
    s = np.full((size, size), 178, np.uint8)
    v = ((0.3 + 0.7 * m) * 255).astype(np.uint8)
    return cv2.cvtColor(np.stack([h, s, v], -1), cv2.COLOR_HSV2RGB)


def test_unrelated_pattern_scores_near_zero():
    """Regresyon: farklı desen artık %50 değil, ~0 almalı."""
    base, other = _pattern(1), _pattern(99)
    d = hasher.hamming(hasher.compute(base, "phash", 8),
                       hasher.compute(other, "phash", 8))
    assert hasher.similarity(d, 64) < 0.30, "alakasız desen hâlâ yüksek skor alıyor"


def test_same_pattern_outranks_unrelated_pattern():
    """Regresyon: aynı desenin varyantı, alakasız desenden yüksek skor almalı.

    v1.0'da bu koşul kırpılmış varyantta bozuluyordu (farklı desen 0.59 vs
    kırpılmış aynı desen 0.50).
    """
    base = _pattern(1)
    q = hasher.compute(base, "phash", 8)

    def score(img):
        return hasher.similarity(
            hasher.hamming(q, hasher.compute(img, "phash", 8)), 64)

    unrelated = score(_pattern(99))
    for name, variant in [
        ("yeniden boyutlandırılmış", np.ascontiguousarray(base[::2, ::2])),
        ("farklı renk", _pattern(1, hue_shift=0.5)),
    ]:
        assert score(variant) > unrelated, f"{name} varyantı alakasızın altında kaldı"


def test_recolored_variant_survives_default_threshold():
    """Aynı desenin farklı renkli hâli varsayılan eşikte elenmemeli."""
    cfg = cfg_mod.AppConfig()
    d = hasher.hamming(hasher.compute(_pattern(1), "phash", 8),
                       hasher.compute(_pattern(1, hue_shift=0.5), "phash", 8))
    assert d <= cfg.effective_max_distance()
    assert hasher.similarity(d, 64) >= cfg.score_threshold
