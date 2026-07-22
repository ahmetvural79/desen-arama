"""Arama servisi — sorgu imajıyla benzerlik araması (query-by-example).

Arka uçlar:

* **hash** — algısal hash + BK-tree; Hamming mesafesinden benzerlik. Hızlı,
  AI'sız. "Aynı / çok benzer şekil" için birincil.
* **embedding** — DINOv2 + FAISS kosinüs; "farklı renk / döndürülmüş / farklı
  çekim aynı desen" için.
* **hybrid** — hash ile ucuz aday üret, embedding ile yeniden sırala.

Ortak kalite hileleri:

* **8×TTA (döndürme dayanıklılığı):** sorgudan 0/90/180/270° × yatay ayna ile
  8 varyant üretilir; her aday için varyantlar arası en iyi skor alınır. CNN/ViT
  ve hash'ler döndürmeye duyarlı olduğundan bu, döndürülmüş taramaları yakalar.
* **Renk re-ranking:** ``final = (1-α)·desen_skoru + α·renk_benzerliği``.
* **Kopya rozeti:** pHash Hamming mesafesi eşik altındaki sonuçlar "birebir
  kopya" işaretlenir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .. import config as cfg_mod
from ..core import colorhist, hasher, imageio, paths
from .engine import Engine

log = logging.getLogger("desenarama.search")


@dataclass
class SearchResult:
    image_id: int
    path: str
    thumb_path: str | None
    score: float            # 0..1 nihai (renk re-rank sonrası)
    pattern_score: float    # desen/hash/embedding skoru
    color_sim: float        # renk benzerliği
    hamming: int | None     # sorgu pHash'ine Hamming mesafesi (varsa)
    is_duplicate: bool
    width: int
    height: int
    below_threshold: bool


def _tta_variants(rgb: np.ndarray) -> list[np.ndarray]:
    """0/90/180/270° × yatay ayna → 8 bitişik (contiguous) varyant."""
    out = []
    for k in range(4):
        r = np.rot90(rgb, k)
        out.append(np.ascontiguousarray(r))
        out.append(np.ascontiguousarray(np.fliplr(r)))
    return out


class SearchService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.config = engine.config

    # -- ana giriş ---------------------------------------------------------- #
    def search(
        self,
        query_path: str | None = None,
        query_rgb: np.ndarray | None = None,
        k: int | None = None,
        color_alpha: float | None = None,
    ) -> list[SearchResult]:
        if query_rgb is None:
            if query_path is None:
                raise ValueError("query_path veya query_rgb verilmeli")
            query_rgb = imageio.load(query_path).rgb
        k = k or self.config.max_results
        alpha = self.config.color_alpha if color_alpha is None else color_alpha

        backend = self.config.backend
        if backend == cfg_mod.BACKEND_HASH:
            hits = self._search_hash(query_rgb, k)
        elif backend == cfg_mod.BACKEND_EMBEDDING:
            hits = self._search_embedding(query_rgb, k)
        else:  # hibrit
            hits = self._search_hybrid(query_rgb, k)

        return self._finalize(query_rgb, hits, alpha, k)

    # -- hash arka ucu ------------------------------------------------------ #
    def _search_hash(self, rgb: np.ndarray, k: int) -> dict[int, float]:
        """{image_id: pattern_score(0..1)} — en iyi (varyant içi) benzerlik."""
        idx = self.engine.hash_index
        if idx is None or len(idx) == 0:
            return {}
        hs = self.config.hash_size
        algo = self.config.hash_algo
        variants = _tta_variants(rgb) if self.config.tta else [rgb]
        max_dist = self.config.hash_max_distance
        best: dict[int, float] = {}
        # Aday sayısını geniş tut (renk re-rank için) — k'nın birkaç katı.
        cand_k = max(k * 3, 100)
        for v in variants:
            qh = hasher.compute(v, algo, hs)
            for r in idx.search(qh, k=cand_k, max_distance=max_dist if max_dist > 0 else None):
                s = r.similarity
                if s > best.get(r.image_id, -1.0):
                    best[r.image_id] = s
        return best

    # -- embedding arka ucu ------------------------------------------------- #
    def _search_embedding(self, rgb: np.ndarray, k: int) -> dict[int, float]:
        vi = self.engine.vector_index
        if vi is None or vi.ntotal == 0:
            return {}
        embedder = self.engine.get_embedder()
        variants = _tta_variants(rgb) if self.config.tta else [rgb]
        qvecs = embedder.embed_batch(variants)  # (V, dim)
        cand_k = max(self.config.rerank_candidates, k)
        scores, ids = vi.search(qvecs, cand_k)  # (V, cand_k)
        best: dict[int, float] = {}
        for vi_row in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                row = int(ids[vi_row, j])
                if row < 0:
                    continue
                img_id = self.engine.row_to_id.get(row)
                if img_id is None:
                    continue
                s = float(scores[vi_row, j])
                if s > best.get(img_id, -2.0):
                    best[img_id] = s
        # kosinüs -1..1 → 0..1 normalize
        return {i: (s + 1.0) / 2.0 for i, s in best.items()}

    # -- hibrit ------------------------------------------------------------- #
    def _search_hybrid(self, rgb: np.ndarray, k: int) -> dict[int, float]:
        """Hash ile ucuz aday üret; adayları embedding ile yeniden sırala."""
        hash_hits = self._search_hash(rgb, max(k * 5, 200))
        if not self.engine.vector_index or not hash_hits:
            return hash_hits
        embedder = self.engine.get_embedder()
        variants = _tta_variants(rgb) if self.config.tta else [rgb]
        qv = self.engine.vector_index.normalize(embedder.embed_batch(variants))  # (V, dim)
        # Yalnızca hash adaylarını embedding ile yeniden sırala (ucuz).
        result: dict[int, float] = {}
        for img_id, hash_score in hash_hits.items():
            cur = self.engine.store._conn.execute(
                "SELECT vec FROM vectors WHERE image_id=?", (img_id,)
            ).fetchone()
            if cur is None:
                result[img_id] = hash_score  # embedding yoksa hash skoruyla bırak
                continue
            cand = np.frombuffer(cur["vec"], dtype=np.float32).copy()
            cand /= (np.linalg.norm(cand) + 1e-9)
            sim = float(np.max(qv @ cand))  # varyantlar arası en iyi kosinüs
            result[img_id] = (sim + 1.0) / 2.0
        return result

    # -- ortak son işleme --------------------------------------------------- #
    def _finalize(self, rgb: np.ndarray, hits: dict[int, float], alpha: float, k: int
                  ) -> list[SearchResult]:
        if not hits:
            return []
        q_color = colorhist.histogram(rgb)
        q_phash = hasher.compute(rgb, "phash", self.config.hash_size)
        # TTA: kopya rozeti için sorgu pHash varyantlarının en yakınını kullan
        q_phash_variants = (
            [hasher.compute(v, "phash", self.config.hash_size) for v in _tta_variants(rgb)]
            if self.config.tta else [q_phash]
        )
        dup_thr = self.config.duplicate_hamming
        thr = self.config.score_threshold

        results: list[SearchResult] = []
        for img_id, pat in hits.items():
            rec = self.engine.store.get_by_id(img_id)
            if rec is None or rec.status != "ok":
                continue
            color_sim = 0.0
            if rec.color:
                color_sim = colorhist.similarity(q_color, colorhist.from_blob(rec.color))
            final = (1.0 - alpha) * pat + alpha * color_sim
            ham = None
            is_dup = False
            if rec.phash:
                cand_phash = hasher.from_blob(rec.phash)
                ham = min(hasher.hamming(qv, cand_phash) for qv in q_phash_variants)
                is_dup = ham <= dup_thr
            thumb = str(paths.thumbs_dir() / rec.thumb) if rec.thumb else None
            results.append(SearchResult(
                image_id=img_id, path=rec.path, thumb_path=thumb, score=final,
                pattern_score=pat, color_sim=color_sim, hamming=ham, is_duplicate=is_dup,
                width=rec.width or 0, height=rec.height or 0,
                below_threshold=final < thr,
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
