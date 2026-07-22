"""Uçtan uca entegrasyon: indeksleme + arama (hash ve fallback-embedding)."""

from __future__ import annotations

import os

from desenarama import config as cfg_mod
from desenarama.core import embedder as emb_mod
from desenarama.services.engine import Engine
from desenarama.services.indexer import IndexerService
from desenarama.services.search import SearchService


def _index(engine):
    prog = IndexerService(engine).reindex()
    assert prog.errors == 0
    return prog


def test_hash_backend_end_to_end(fresh_env, sample_dataset):
    cfg = cfg_mod.AppConfig()
    cfg.library_roots = [sample_dataset["dir"]]
    cfg.backend = cfg_mod.BACKEND_HASH
    cfg.save()
    engine = Engine(cfg)
    engine.open()
    prog = _index(engine)
    assert prog.done == len(sample_dataset["labels"])
    assert engine.store.count("ok") == prog.done

    # Sorgu: bir orijinal imaj kendisini tam skorla ve kopya rozetiyle bulmalı
    query = os.path.join(sample_dataset["dir"], "fam0_orig.png")
    results = SearchService(engine).search(query_path=query, k=10)
    assert results, "sonuç boş olmamalı"
    top = results[0]
    assert top.path.endswith("fam0_orig.png")
    assert top.hamming == 0 and top.is_duplicate and top.score > 0.99
    engine.close()


def test_tta_finds_rotated_variant(fresh_env, sample_dataset):
    cfg = cfg_mod.AppConfig()
    cfg.library_roots = [sample_dataset["dir"]]
    cfg.backend = cfg_mod.BACKEND_HASH
    cfg.tta = True
    cfg.save()
    engine = Engine(cfg)
    engine.open()
    _index(engine)

    # fam0_rot90 kütüphanede; fam0_orig ile TTA açıkken düşük Hamming (döndürme yakalanır)
    query = os.path.join(sample_dataset["dir"], "fam0_orig.png")
    results = SearchService(engine).search(query_path=query, k=48)
    by_name = {os.path.basename(r.path): r for r in results}
    assert "fam0_rot90.png" in by_name
    assert by_name["fam0_rot90.png"].hamming == 0  # TTA döndürmeyi eşledi
    engine.close()


def test_color_reranking_moves_recolor_down(fresh_env, sample_dataset):
    """Aynı desen farklı renk: renk ağırlığı artınca skor düşmeli."""
    cfg = cfg_mod.AppConfig()
    cfg.library_roots = [sample_dataset["dir"]]
    cfg.backend = cfg_mod.BACKEND_HASH
    cfg.save()
    engine = Engine(cfg)
    engine.open()
    _index(engine)
    query = os.path.join(sample_dataset["dir"], "fam0_orig.png")

    svc = SearchService(engine)
    low = {os.path.basename(r.path): r.score for r in svc.search(query_path=query, k=48, color_alpha=0.0)}
    high = {os.path.basename(r.path): r.score for r in svc.search(query_path=query, k=48, color_alpha=0.8)}
    # recolor: desen aynı ama renk farklı → renk ağırlığı artınca skoru düşer
    assert high["fam0_recolor.png"] < low["fam0_recolor.png"]
    engine.close()


def test_embedding_backend_with_fallback(fresh_env, sample_dataset, monkeypatch):
    """AI arka ucu, model indirmeden fallback embedder ile uçtan uca çalışmalı."""
    cfg = cfg_mod.AppConfig()
    cfg.library_roots = [sample_dataset["dir"]]
    cfg.backend = cfg_mod.BACKEND_EMBEDDING
    cfg.save()
    engine = Engine(cfg)
    # İnternet/model indirmeyi engelle: fallback embedder'ı önceden ata
    engine._embedder = emb_mod.FallbackEmbedder()
    engine.open()
    _index(engine)
    assert engine.store.count_vectors() == len(sample_dataset["labels"])
    assert engine.vector_index is not None and engine.vector_index.ntotal == engine.store.count_vectors()

    query = os.path.join(sample_dataset["dir"], "fam2_orig.png")
    results = SearchService(engine).search(query_path=query, k=10)
    assert results
    assert results[0].path.endswith("fam2_orig.png")  # kendini bulur
    assert results[0].score > 0.9
    engine.close()


def test_incremental_reindex_no_duplicates(fresh_env, sample_dataset):
    """İkinci indeksleme aynı dosyaları yeniden eklememeli (idempotent)."""
    cfg = cfg_mod.AppConfig()
    cfg.library_roots = [sample_dataset["dir"]]
    cfg.backend = cfg_mod.BACKEND_HASH
    cfg.save()
    engine = Engine(cfg)
    engine.open()
    p1 = _index(engine)
    n = engine.store.count()
    p2 = IndexerService(engine).reindex()  # değişiklik yok
    assert engine.store.count() == n  # kayıt sayısı değişmedi
    assert p2.new == 0 and p2.changed == 0
    engine.close()
