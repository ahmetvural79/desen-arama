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
* **Renk re-ranking:** ``final = desen_skoru · (1-α + α·renk_benzerliği)`` —
  desen kapıdır, renk en fazla α oranında düzeltme yapar (bkz. :func:`_blend`).
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

    def quality(self) -> str:
        """Skoru kullanıcının okuyabileceği bir kalite bandına çevirir.

        Ham yüzde tek başına yanıltıcıydı; bant, skorun kalibre edilmiş
        ölçekte ne anlama geldiğini açık eder.
        """
        if self.is_duplicate:
            return "Kopya"
        if self.score >= 0.75:
            return "Çok benzer"
        if self.score >= 0.50:
            return "Benzer"
        if self.score >= 0.35:
            return "Zayıf"
        return "Çok zayıf"


def _cosine_to_score(cos: float, baseline: float = 0.0) -> float:
    """Kosinüs benzerliğini kalibre edilmiş 0..1 skora çevirir.

    İki katmanlı bir düzeltmedir.

    **1. Negatif tarafı boşa harcama.** Eski eşleme ``(cos + 1) / 2`` idi;
    embedding uzayında alakasız iki görselin kosinüsü 0 civarında olduğundan
    alakasız her şey %50 skor alıyordu.

    **2. Kütüphaneye göre taban alma.** DINOv2 kosinüsleri dar ve yüksek bir
    bantta yaşar; hepsi halı olan bir arşivde **alakasız** iki desen bile ~0.89
    kosinüs verir (ölçüldü). Sabit bir ölçek bu yüzden yine "her şey %90 benzer"
    üretir. ``baseline`` kütüphaneden alınan temsilî örneklemin medyanıdır;
    skor bu tabanın üstündeki paya göre yeniden ölçeklenir:

        skor = (cos - taban) / (1 - taban)

    Böylece tipik bir kütüphane üyesi 0, birebir eşleşme 1 alır ve ölçek
    koleksiyonun kendi dağılımına uyarlanır (fotoğraf arşivi ile halı arşivi
    aynı formülle doğru davranır).
    """
    if baseline > 0.0:
        span = 1.0 - baseline
        if span <= 1e-6:
            return 1.0 if cos >= baseline else 0.0
        cos = (cos - baseline) / span
    return max(0.0, min(1.0, cos))


def _blend(pattern: float, color_sim: float | None, alpha: float) -> float:
    """Desen skorunu renk benzerliğiyle harmanlar — **çarpımsal** modülasyon.

    Eski formül toplamsaldı: ``(1-α)·desen + α·renk``. Renk histogramı kesişimi
    kalibre değildir; benzer paletteki **alakasız** görseller 0.9+ alırken, aynı
    desenin farklı renkli (colorway) varyantı 0.0 alır. Toplamsal harmanda bu,
    α=0.2'de bile sıralamayı tersine çeviriyordu: alakasız görseller aynı desenin
    farklı renkli hâlinin üstüne çıkıyordu (ölçümle doğrulandı).

    Çarpımsal formda desen **kapı**, renk ise en fazla ``α`` oranında bir
    düzeltmedir; renk asla desen eşleşmesi olmayan bir sonucu yukarı taşıyamaz:

    * ``α = 0``   → yalnızca desen
    * ``α = 0.2`` → renk uyuşmazlığı skoru en çok %20 düşürür
    * ``α = 1``   → renk uyuşmayan desen eşleşmeleri tamamen elenir

    Renk bilgisi olmayan kayıtlarda (eski indeks) modülasyon uygulanmaz.
    """
    if color_sim is None or alpha <= 0.0:
        return pattern
    return pattern * (1.0 - alpha + alpha * color_sim)


def _tta_variants(rgb: np.ndarray) -> list[np.ndarray]:
    """0/90/180/270° × yatay ayna → 8 bitişik (contiguous) varyant."""
    out = []
    for k in range(4):
        r = np.rot90(rgb, k)
        out.append(np.ascontiguousarray(r))
        out.append(np.ascontiguousarray(np.fliplr(r)))
    return out


def _center_crop(rgb: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Görselin merkezinden ``fraction`` oranında bir kare kırpar."""
    h, w = rgb.shape[0], rgb.shape[1]
    ch, cw = max(1, int(h * fraction)), max(1, int(w * fraction))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return np.ascontiguousarray(rgb[y0:y0 + ch, x0:x0 + cw])


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

    # -- sorgu görünümleri -------------------------------------------------- #
    def _hash_views(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Hash araması için sorgu varyantları — yalnızca döndürme/ayna.

        Hash bir *yakın kopya* aracıdır; kırpma varyantları eklemek recall'dan
        çok gürültü getirir (kırpılmış hash zaten tamamen farklı bir hash'tir).
        """
        return _tta_variants(rgb) if self.config.tta else [rgb]

    def _embedding_views(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Embedding araması için sorgu varyantları — döndürme + çok ölçeklilik.

        Sorgunun tamamına ek olarak merkez %50 kırpması da aranır: kütüphanedeki
        görsel sorgunun bir *yakın çekimi* olduğunda eşleşmeyi sağlar. Yalnızca
        sorgu maliyetidir, indeks büyümez. Ters yön (sorgu kütüphane görselinin
        parçası) indeks tarafındaki karolarla çözülür (bkz. ``tile_grid``).
        """
        views = [rgb]
        if self.config.query_multiscale:
            views.append(_center_crop(rgb))
        if not self.config.tta:
            return views
        return [v for view in views for v in _tta_variants(view)]

    # -- hash arka ucu ------------------------------------------------------ #
    def _search_hash(self, rgb: np.ndarray, k: int) -> dict[int, float]:
        """{image_id: pattern_score(0..1)} — en iyi (varyant içi) benzerlik."""
        idx = self.engine.hash_index
        if idx is None or len(idx) == 0:
            return {}
        hs = self.config.hash_size
        algo = self.config.hash_algo
        variants = self._hash_views(rgb)
        # Eşik 64-bit referansıyla saklanır; seçili hash boyutuna ölçeklenir.
        max_dist = self.config.effective_max_distance()
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
        variants = self._embedding_views(rgb)
        qvecs = embedder.embed_batch(variants)  # (V, dim)
        # Karo indekslemede aynı imaj birden çok satır kaplar; aday sayısını
        # karo başına ölçekle, aksi hâlde top-k'yı tek bir imajın karoları
        # doldurabilir.
        cand_k = max(self.config.rerank_candidates, k) * max(1, self.engine.tiles_per_image())
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
        baseline = self._embedding_baseline(qvecs)
        return {i: _cosine_to_score(s, baseline) for i, s in best.items()}

    def _embedding_baseline(self, qvecs: np.ndarray) -> float:
        """Sorgunun kütüphanedeki **tipik** benzerlik düzeyi (medyan).

        Motorun tuttuğu temsilî örnekleme karşı hesaplanır. TTA varyantları
        adaylarda olduğu gibi burada da maksimumla birleştirilir; aksi hâlde
        taban sistematik olarak düşük çıkar ve kalibrasyon şişer.

        Kütüphane çok küçükse (örneklem yoksa) 0 döner — kalibrasyon uygulanmaz.
        """
        sample = self.engine.baseline_vectors
        if sample is None or len(sample) == 0:
            return 0.0
        q = self.engine.vector_index.normalize(qvecs)  # (V, dim)
        per_item = np.max(q @ sample.T, axis=0)        # (N,) varyantlar arası en iyi
        return float(np.median(per_item))

    # -- hibrit ------------------------------------------------------------- #
    def _search_hybrid(self, rgb: np.ndarray, k: int) -> dict[int, float]:
        """Hash **ve** embedding adaylarını birleştirip embedding ile sırala.

        Önceki sürümde adaylar yalnızca hash'ten geliyordu; bu, hibridin
        recall'ını hash'in recall'ıyla sınırlıyordu — hash kaçırdığında (farklı
        renk, kısmi eşleşme) embedding'in kurtarma şansı yoktu. Artık iki aday
        havuzu birleştirilir: hash ucuz ve kesin, embedding geniş ve derin.
        """
        hash_hits = self._search_hash(rgb, max(k * 5, 200))
        if not self.engine.vector_index:
            return hash_hits

        embed_hits = self._search_embedding(rgb, k)
        if not hash_hits and not embed_hits:
            return {}

        # İki skorun **maksimumu** alınır, embedding'inki tek başına değil.
        # Ölçümde iki yöntemin güçlü olduğu senaryolar farklı çıktı: hash
        # colorway varyantlarında belirgin biçimde daha iyi (recall@10 0.65 vs
        # 0.25 — gri tonlama üzerinden çalıştığı için renk değişimine dayanıklı),
        # karo destekli embedding ise kırpılmış sorgularda tek çalışan yöntem
        # (0.40 vs 0.00). Embedding skorunu üste yazmak hash'in colorway
        # üstünlüğünü çöpe atıyordu; maksimum ikisini de korur.
        result = {i: max(s, hash_hits.get(i, 0.0)) for i, s in embed_hits.items()}
        missing = [i for i in hash_hits if i not in result]
        if missing:
            embedder = self.engine.get_embedder()
            qv = self.engine.vector_index.normalize(
                embedder.embed_batch(self._embedding_views(rgb)))
            baseline = self._embedding_baseline(qv)
            vectors = self.engine.store.get_vectors_for(missing)  # tek sorgu
            for img_id in missing:
                blobs = vectors.get(img_id)
                if not blobs:
                    # Embedding'i olmayan kayıt: hash skoruyla bırak.
                    result[img_id] = hash_hits[img_id]
                    continue
                cand = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
                cand = self.engine.vector_index.normalize(cand)
                # Sorgu varyantları × aday karoları arasındaki en iyi eşleşme.
                embed_score = _cosine_to_score(float(np.max(qv @ cand.T)), baseline)
                result[img_id] = max(embed_score, hash_hits[img_id])
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
        dup_thr = self.config.effective_duplicate_hamming()
        thr = self.config.score_threshold

        results: list[SearchResult] = []
        for img_id, pat in hits.items():
            rec = self.engine.store.get_by_id(img_id)
            if rec is None or rec.status != "ok":
                continue
            color_sim = 0.0
            has_color = bool(rec.color)
            if has_color:
                color_sim = colorhist.similarity(q_color, colorhist.from_blob(rec.color))
            final = _blend(pat, color_sim if has_color else None, alpha)
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
