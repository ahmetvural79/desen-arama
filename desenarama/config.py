"""Uygulama yapılandırması — JSON olarak yerel veri dizininde kalıcı tutulur.

Yapılandırma arama arka ucunu (hash / embedding / hibrit), hash algoritmasını,
AI modelini, ağ optimizasyon parametrelerini ve arama davranışını (renk ağırlığı,
döndürme dayanıklılığı) belirler. Varsayılanlar **hash tabanlı hızlı mod**a
ayarlıdır: model gerektirmez, ağ arşivinde hızlı çalışır.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from .core import paths

# Arama arka uçları
BACKEND_HASH = "hash"        # algısal hash + BK-tree (hızlı, AI'sız — varsayılan)
BACKEND_EMBEDDING = "embedding"  # DINOv2 + FAISS
BACKEND_HYBRID = "hybrid"    # hash ön-eleme + embedding yeniden sıralama
BACKENDS = (BACKEND_HASH, BACKEND_EMBEDDING, BACKEND_HYBRID)


@dataclass
class AppConfig:
    # -- kütüphane -- #
    library_roots: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])

    # -- arama arka ucu -- #
    backend: str = BACKEND_HASH
    hash_algo: str = "phash"          # phash | dhash | ahash | whash
    hash_size: int = 8                # 8 => 64-bit; 16 => 256-bit (daha ince)
    hash_max_distance: int = 12       # bu Hamming eşiği üstü "benzemez" sayılır

    # -- AI (embedding) -- #
    model_key: str = "dinov2-small"   # dinov2-small | dinov2-base
    prefer_gpu: bool = False          # DirectML/CUDA varsa kullan
    tta: bool = True                  # döndürme dayanıklılığı (8x test-time augmentation)

    # -- yeniden sıralama -- #
    color_alpha: float = 0.2          # 0=sadece desen, 1=sadece renk
    rerank_candidates: int = 200      # embedding aramasında yeniden sıralanacak aday sayısı

    # -- performans / ağ -- #
    io_workers: int = 8               # dosya okuma iş parçacığı (ağda daha yüksek yardımcı olur)
    cpu_batch: int = 16               # embedding batch boyutu
    thumb_size: int = 256
    max_results: int = 100
    auto_tune_network: bool = True    # ağ paylaşımı tespit edilirse io_workers'ı artır
    watch_mode: str = "auto"          # auto | native | polling | off
    rescan_interval_sec: int = 0      # >0 ise periyodik yeniden tarama (ağ için)

    # -- eşikler -- #
    duplicate_hamming: int = 8        # bu eşik altı "birebir kopya" rozeti
    score_threshold: float = 0.0      # bu skorun altı sonuç gri gösterilir

    def config_path(self) -> str:
        return str(paths.data_dir() / "config.json")

    def save(self) -> None:
        path = self.config_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls) -> "AppConfig":
        path = str(paths.data_dir() / "config.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                known = {k for k in cls().__dataclass_fields__}  # type: ignore[attr-defined]
                return cls(**{k: v for k, v in data.items() if k in known})
            except Exception:
                pass
        return cls()

    def uses_embedding(self) -> bool:
        return self.backend in (BACKEND_EMBEDDING, BACKEND_HYBRID)

    def ext_set(self) -> set[str]:
        return {e.lower() if e.startswith(".") else "." + e.lower() for e in self.extensions}
