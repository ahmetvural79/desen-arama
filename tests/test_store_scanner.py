"""Depo, tarayıcı ve yol yardımcıları testleri."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from desenarama.core import paths, scanner, store


def test_store_roundtrip(fresh_env):
    st = store.ImageStore()
    iid = st.upsert(key="k1", path="/x/a.jpg", mtime=1.0, size=100, width=10, height=10,
                    phash=b"\x00" * 8, status="ok", indexed_at=store.now())
    rec = st.get_by_key("k1")
    assert rec.id == iid and rec.status == "ok" and rec.phash == b"\x00" * 8
    # upsert güncelleme
    st.upsert(key="k1", path="/x/a.jpg", size=200, mtime=1.0, status="ok")
    assert st.get_by_key("k1").size == 200
    st.close()


def test_store_vectors_and_missing(fresh_env):
    st = store.ImageStore()
    iid = st.upsert(key="k2", path="/x/b.jpg", mtime=1.0, size=1, status="ok", indexed_at=store.now())
    vec = np.ones(8, dtype=np.float32).tobytes()
    st.upsert_vector(iid, vec)
    assert st.count_vectors() == 1
    got = list(st.iter_vectors())
    assert got[0][0] == iid and got[0][1] == vec
    # silme vektörü de siler
    st.delete_keys(["k2"])
    assert st.count_vectors() == 0
    st.close()


def test_existing_keys_for_diff(fresh_env):
    st = store.ImageStore()
    st.upsert(key="a", path="a", mtime=5.0, size=10, status="ok")
    ek = st.existing_keys()
    assert ek["a"] == (5.0, 10)
    st.close()


def test_scanner_incremental_diff(tmp_path, fresh_env):
    # 3 imaj oluştur
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(d / f"i{i}.png")

    diff1 = scanner.diff([str(d)], existing={})
    assert len(diff1.new) == 3 and len(diff1.changed) == 0

    # DB durumunu simüle et
    existing = {e.key: (e.mtime, e.size) for e in diff1.new}
    diff2 = scanner.diff([str(d)], existing=existing)
    assert len(diff2.new) == 0 and diff2.unchanged == 3

    # bir dosyayı büyüt (size değişir)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(d / "i0.png")
    diff3 = scanner.diff([str(d)], existing=existing)
    assert len(diff3.changed) == 1

    # bir dosyayı sil → missing
    os.remove(d / "i1.png")
    diff4 = scanner.diff([str(d)], existing=existing)
    assert any(k.endswith("i1.png") or "i1" in k for k in diff4.missing)


def test_scanner_extension_filter(tmp_path):
    d = tmp_path / "mix"
    d.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(d / "ok.png")
    (d / "not.txt").write_text("selam")
    entries = list(scanner.iter_images(str(d), extensions={".png"}))
    assert len(entries) == 1 and entries[0].path.endswith("ok.png")


def test_normalize_key_stable():
    k1 = paths.normalize_key("/a/b/../b/c.jpg")
    k2 = paths.normalize_key("/a/b/c.jpg")
    assert k1 == k2
