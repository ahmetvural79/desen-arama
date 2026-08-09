"""Karo (tile) indeksleme ve kısmi eşleşme testleri.

v1.0'da hem hash hem embedding tek bir **global** imza üretiyordu. Halı
arşivinde sorgu sıklıkla desenin bir parçasıdır (motif fotoğrafı, kısmi tarama);
karşılığı ise tüm halının taramasıdır. Ölçümde kırpılmış sorgular alakasız
desenlerin altına düşüyordu.
"""

from __future__ import annotations

import numpy as np
import pytest

from desenarama import config as cfg_mod
from desenarama.core import embedder as emb_mod
from desenarama.core import store


# --------------------------------------------------------------------------- #
# Karo üretimi
# --------------------------------------------------------------------------- #
def test_tiles_disabled_returns_whole_image_only():
    rgb = np.zeros((90, 60, 3), np.uint8)
    assert len(emb_mod.tiles(rgb, 0)) == 1
    assert len(emb_mod.tiles(rgb, 1)) == 1


@pytest.mark.parametrize("grid,expected", [(2, 5), (3, 10), (4, 17)])
def test_tile_count(grid, expected):
    """Tam kare + grid×grid karo."""
    rgb = np.zeros((120, 120, 3), np.uint8)
    assert len(emb_mod.tiles(rgb, grid)) == expected


def test_tiles_cover_image_without_overlap():
    """Karolar görseli tam kaplamalı; piksel ne atlanmalı ne iki kez sayılmalı."""
    rgb = np.arange(60 * 90 * 3, dtype=np.uint8).reshape(60, 90, 3)
    crops = emb_mod.tiles(rgb, 3)[1:]  # tam kareyi atla
    assert sum(c.shape[0] * c.shape[1] for c in crops) == 60 * 90


def test_tiles_handle_non_divisible_sizes():
    """Kenar uzunluğu ızgaraya bölünmüyorsa da karo üretimi çökmemeli."""
    rgb = np.zeros((17, 23, 3), np.uint8)
    crops = emb_mod.tiles(rgb, 3)
    assert len(crops) == 10
    assert all(c.size > 0 for c in crops)


def test_tiny_image_produces_valid_tiles():
    rgb = np.zeros((2, 2, 3), np.uint8)
    assert all(c.size > 0 for c in emb_mod.tiles(rgb, 3))


# --------------------------------------------------------------------------- #
# Depo: karo vektörleri
# --------------------------------------------------------------------------- #
def _vec(fill: float, dim: int = 8) -> bytes:
    return np.full(dim, fill, dtype=np.float32).tobytes()


def test_store_roundtrips_multiple_tiles(fresh_env):
    st = store.ImageStore()
    iid = st.upsert(key="k", path="/x.png", mtime=1.0, size=1, status="ok")
    st.upsert_vectors([(iid, 0, _vec(1)), (iid, 1, _vec(2)), (iid, 2, _vec(3))])

    assert st.count_vectors() == 3
    assert st.count_vector_images() == 1
    assert [t for _, t, _ in st.iter_vectors()] == [0, 1, 2]
    st.close()


def test_reindexing_with_smaller_grid_drops_stale_tiles(fresh_env):
    """Izgara küçültülünce eski karolar ortada kalmamalı (orphan)."""
    st = store.ImageStore()
    iid = st.upsert(key="k", path="/x.png", mtime=1.0, size=1, status="ok")
    st.upsert_vectors([(iid, t, _vec(t)) for t in range(10)])
    assert st.count_vectors() == 10

    st.upsert_vectors([(iid, 0, _vec(99))])  # karo kapatıldı
    assert st.count_vectors() == 1
    st.close()


def test_get_vectors_for_batches_all_tiles(fresh_env):
    st = store.ImageStore()
    a = st.upsert(key="a", path="/a.png", mtime=1.0, size=1, status="ok")
    b = st.upsert(key="b", path="/b.png", mtime=1.0, size=1, status="ok")
    st.upsert_vectors([(a, 0, _vec(1)), (a, 1, _vec(2)), (b, 0, _vec(3))])

    got = st.get_vectors_for([a, b])
    assert len(got[a]) == 2 and len(got[b]) == 1
    assert st.get_vectors_for([]) == {}
    st.close()


def test_clear_vectors_keeps_metadata(fresh_env):
    """Embedding'ler geçersizleşince metadata (pahalı tarama) korunmalı."""
    st = store.ImageStore()
    iid = st.upsert(key="k", path="/x.png", mtime=1.0, size=1,
                    phash=b"\x00" * 8, status="ok")
    st.upsert_vectors([(iid, 0, _vec(1))])

    st.clear_vectors()
    assert st.count_vectors() == 0
    assert st.count("ok") == 1
    assert st.get_by_id(iid).phash == b"\x00" * 8
    st.close()


def test_schema_migration_drops_v1_vectors(fresh_env):
    """v1 şemasındaki vektörler karo sütunu olmadığı için yeniden üretilmeli."""
    import sqlite3

    db = str(fresh_env / "library.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE vectors(image_id INTEGER PRIMARY KEY, vec BLOB NOT NULL);
        INSERT INTO meta VALUES('schema_version','1');
        INSERT INTO vectors VALUES(1, x'00');
    """)
    conn.commit()
    conn.close()

    st = store.ImageStore(db)
    assert st.count_vectors() == 0
    assert st.get_meta("schema_version") == str(store.SCHEMA_VERSION)
    # Yeni şema karo sütununu kabul etmeli
    st.upsert_vectors([(1, 3, _vec(1))])
    assert [t for _, t, _ in st.iter_vectors()] == []  # imaj kaydı yok (status filtresi)
    assert st.count_vectors() == 1
    st.close()


# --------------------------------------------------------------------------- #
# Embedding imzası
# --------------------------------------------------------------------------- #
def test_signature_changes_with_preprocessing_and_grid():
    """Regresyon: ön işleme değişince eski vektörler geçersiz sayılmalı.

    v1.0'da böyle bir kontrol yoktu; Faz C'deki letterbox geçişi eski indeksleri
    uyumsuz hâle getirdiği hâlde uygulama bunu fark etmezdi.
    """
    a = emb_mod.signature("onnx-dinov2", "dinov2-small", 384, 0)
    b = emb_mod.signature("onnx-dinov2", "dinov2-small", 384, 3)
    c = emb_mod.signature("onnx-dinov2", "dinov2-base", 768, 0)
    assert a != b != c and a != c
    assert f"prep{emb_mod.PREPROCESS_VERSION}" in a


def test_signature_is_stable_for_same_conditions():
    args = ("onnx-dinov2", "dinov2-small", 384, 3)
    assert emb_mod.signature(*args) == emb_mod.signature(*args)


# --------------------------------------------------------------------------- #
# Yapılandırma
# --------------------------------------------------------------------------- #
def test_tiles_per_image_matches_config():
    from desenarama.services.engine import Engine

    for grid, expected in [(0, 1), (1, 1), (2, 5), (3, 10)]:
        cfg = cfg_mod.AppConfig(tile_grid=grid)
        engine = Engine.__new__(Engine)
        engine.config = cfg
        assert engine.tiles_per_image() == expected


def test_tiling_is_off_by_default():
    """3×3 indekslemeyi ~10× yavaşlatır; kullanıcı bilerek açmalı."""
    assert cfg_mod.AppConfig().tile_grid == 0


def test_query_multiscale_is_on_by_default():
    """Yalnızca sorgu maliyetidir, indeks büyümez."""
    assert cfg_mod.AppConfig().query_multiscale is True


# --------------------------------------------------------------------------- #
# Sorgu görünümleri
# --------------------------------------------------------------------------- #
def _service(**overrides):
    from desenarama.services.search import SearchService

    svc = SearchService.__new__(SearchService)
    svc.config = cfg_mod.AppConfig(**overrides)
    return svc


def test_hash_views_exclude_crops():
    """Hash bir yakın-kopya aracıdır; kırpma varyantı gürültü getirir."""
    rgb = np.zeros((40, 40, 3), np.uint8)
    assert len(_service(tta=True, query_multiscale=True)._hash_views(rgb)) == 8
    assert len(_service(tta=False)._hash_views(rgb)) == 1


def test_embedding_views_add_center_crop():
    rgb = np.zeros((40, 40, 3), np.uint8)
    assert len(_service(tta=False, query_multiscale=True)._embedding_views(rgb)) == 2
    assert len(_service(tta=True, query_multiscale=True)._embedding_views(rgb)) == 16
    assert len(_service(tta=True, query_multiscale=False)._embedding_views(rgb)) == 8


def test_center_crop_is_centered():
    from desenarama.services.search import _center_crop

    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[40:60, 40:60] = 255
    crop = _center_crop(rgb, 0.5)
    assert crop.shape[:2] == (50, 50)
    assert crop.mean() > rgb.mean(), "merkez kırpma merkezi almalı"
