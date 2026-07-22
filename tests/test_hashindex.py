"""BK-tree ve HashIndex birim testleri."""

from __future__ import annotations

from desenarama.core import hashindex
from desenarama.core.hashindex import BKTree, HashIndex


def test_bktree_finds_within_distance():
    tree = BKTree()
    data = {1: 0b0000, 2: 0b0011, 3: 0b1111, 4: 0b0001}
    for img_id, h in data.items():
        tree.add(h, img_id)
    res = tree.query(0b0000, max_distance=1)
    ids = {img_id for img_id, _ in res}
    assert ids == {1, 4}  # 0 ve 1-bit uzaklıktakiler


def test_bktree_handles_duplicate_hashes():
    tree = BKTree()
    tree.add(0b1010, 1)
    tree.add(0b1010, 2)  # aynı hash farklı id
    res = tree.query(0b1010, max_distance=0)
    assert {img_id for img_id, _ in res} == {1, 2}


def test_hashindex_linear_and_tree_agree():
    """Doğrusal ve BK-tree yolları eşik içindeki AYNI kümeyi bulmalı.

    (Berabere mesafeler top-k dilimlemede farklı sıralanabileceğinden, eşik
    içindeki tüm sonuçların kümesi karşılaştırılır — bu, iki yolun
    denkliğinin doğru testidir.)
    """
    items = [(i, (i * 2654435761) & 0xFFFFFFFFFFFFFFFF) for i in range(200)]
    query = items[0][1]

    lin = HashIndex(hash_bits=64, linear_threshold=10_000)  # doğrusal kalır
    for img_id, h in items:
        lin.add(img_id, h)
    lin.build()
    lin_res = {(r.image_id, r.distance) for r in lin.search(query, k=1000, max_distance=20)}

    tree = HashIndex(hash_bits=64, linear_threshold=10)  # BK-tree'ye geçer
    for img_id, h in items:
        tree.add(img_id, h)
    tree.build()
    tree_res = {(r.image_id, r.distance) for r in tree.search(query, k=1000, max_distance=20)}

    assert lin_res == tree_res
    assert (0, 0) in lin_res  # sorgu kendisi mesafe 0


def test_similarity_scaling():
    idx = HashIndex(hash_bits=64)
    idx.add(1, 0)
    idx.build()
    r = idx.search(0, k=1)[0]
    assert r.distance == 0
    assert r.similarity == 1.0
