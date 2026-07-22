"""Algısal hash birim testleri."""

from __future__ import annotations

import numpy as np
import pytest

from desenarama.core import hasher


def _img(seed, size=128):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


@pytest.mark.parametrize("algo", hasher.ALGOS)
def test_all_algos_deterministic(algo):
    img = _img(1)
    a = hasher.compute(img, algo)
    b = hasher.compute(img, algo)
    assert a == b  # aynı girdi → aynı hash


@pytest.mark.parametrize("algo", hasher.ALGOS)
def test_blob_roundtrip(algo):
    img = _img(2)
    v = hasher.compute(img, algo, hash_size=8)
    blob = hasher.to_blob(v, 8)
    assert len(blob) == 8  # 64-bit
    assert hasher.from_blob(blob) == v


def test_hash_size_16_is_256bit():
    img = _img(3)
    v = hasher.compute(img, "phash", hash_size=16)
    blob = hasher.to_blob(v, 16)
    assert len(blob) == 32  # 256-bit
    assert hasher.from_blob(blob) == v
    assert hasher.max_bits(16) == 256


def test_identical_images_zero_distance():
    img = _img(4)
    assert hasher.hamming(hasher.phash(img), hasher.phash(img)) == 0


def test_resize_robustness():
    """Küçültülüp geri büyütülen imaj (kopya senaryosu) düşük Hamming vermeli."""
    from PIL import Image

    img = _img(5, size=256)
    small = np.asarray(Image.fromarray(img).resize((128, 128)).resize((256, 256)))
    d = hasher.hamming(hasher.phash(img), hasher.phash(small))
    assert d <= 10  # yeniden boyutlandırmaya dayanıklı


def test_different_images_larger_distance():
    d_same = hasher.hamming(hasher.phash(_img(6)), hasher.phash(_img(6)))
    d_diff = hasher.hamming(hasher.phash(_img(6)), hasher.phash(_img(7)))
    assert d_same == 0
    assert d_diff > d_same


def test_unknown_algo_raises():
    with pytest.raises(ValueError):
        hasher.compute(_img(8), "bilinmeyen")
