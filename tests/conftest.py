"""Test yardımcıları ve izolasyon fikstürleri.

Her test kendi geçici veri dizininde çalışır (gerçek %LOCALAPPDATA% verisine
dokunulmaz). ``DESENARAMA_DATA_DIR`` ortam değişkeni test başına ayarlanır.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Proje kökünü içe aktarma yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def fresh_env(tmp_path, monkeypatch):
    """Her teste temiz, izole bir yerel veri dizini verir."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DESENARAMA_DATA_DIR", str(data))
    return data


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_dataset(tmp_path):
    """make_dataset ile küçük bir sentetik desen seti üretir; dizini döndürür."""
    from make_dataset import build

    out = tmp_path / "ds"
    info = build(str(out), n_families=6)
    return info
