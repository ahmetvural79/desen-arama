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

from .core import formats, paths

#: Yapılandırma şeması sürümü. Artırıldığında :meth:`AppConfig._migrate`
#: eski dosyaları yeni varsayılanlara taşır.
CONFIG_VERSION = 1

# Arama arka uçları
BACKEND_HASH = "hash"        # algısal hash + BK-tree (hızlı, AI'sız — varsayılan)
BACKEND_EMBEDDING = "embedding"  # DINOv2 + FAISS
BACKEND_HYBRID = "hybrid"    # hash ön-eleme + embedding yeniden sıralama
BACKENDS = (BACKEND_HASH, BACKEND_EMBEDDING, BACKEND_HYBRID)


@dataclass
class AppConfig:
    # -- kütüphane -- #
    library_roots: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=lambda: list(formats.DEFAULT_EXTENSIONS))

    # -- şema sürümü (göç için) -- #
    # Varsayılan 0'dır: alanı içermeyen v1.0 dosyaları böylece "göç edilmemiş"
    # olarak tanınır. Sıfırdan oluşturulan yapılandırma :meth:`load` içinde
    # güncel sürüme damgalanır.
    config_version: int = 0

    # -- arama arka ucu -- #
    backend: str = BACKEND_HASH
    hash_algo: str = "phash"          # phash | dhash | ahash | whash
    hash_size: int = 8                # 8 => 64-bit; 16 => 256-bit (daha ince)
    # Bu Hamming eşiğinin üstü "benzemez" sayılır. **64-bit referansıyla**
    # verilir; 256-bit hash seçilirse otomatik ölçeklenir (bkz.
    # :meth:`effective_max_distance`). Eski varsayılan 12 idi ve yalnızca
    # birebir kopyaları geçiriyordu: aynı desenin kırpılmış veya farklı renkli
    # varyantları 20–30 bant aralığında kaldığı için hiç sonuç dönmüyordu.
    hash_max_distance: int = 22

    # -- AI (embedding) -- #
    model_key: str = "dinov2-small"   # dinov2-small | dinov2-base
    prefer_gpu: bool = False          # DirectML/CUDA varsa kullan
    tta: bool = True                  # döndürme dayanıklılığı (8x test-time augmentation)

    # -- yeniden sıralama -- #
    # 0 = yalnızca desen; artan değerler renk uyuşmazlığını cezalandırır
    # (çarpımsal, bkz. services.search._blend).
    #
    # Varsayılan 0'dır: halı arşivinde "aynı desen, farklı renk" (colorway)
    # birincil bir arama senaryosudur ve renk histogramı kesişimi kalibre
    # değildir — benzer paletteki alakasız görseller 0.9+ alırken aynı desenin
    # farklı renkli varyantı 0.0 alır. Renk ağırlığı vermek bu yüzden varsayılan
    # olarak doğru sonucu gömüyordu. Kullanıcı kaydırıcıyla anında artırabilir.
    color_alpha: float = 0.0
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
    duplicate_hamming: int = 8        # bu eşik altı "birebir kopya" rozeti (64-bit referans)
    # Kalibre edilmiş skor ölçeğinde (bkz. hasher.similarity) anlamlı bir taban.
    # 0.0 iken alakasız her sonuç listeleniyordu.
    score_threshold: float = 0.35

    def __post_init__(self) -> None:
        # Dataclass alanı **değildir**: diske yazılmaz, yalnızca bu oturumda
        # göçün ne değiştirdiğini arayüze taşır.
        self.migration_notes: list[str] = []
        # Göç kütüphane kapsamını genişlettiyse mevcut indeks eksiktir; arayüz
        # yalnızca bu durumda yeniden tarama önerir (eşik değişikliği için gerekmez).
        self.migration_needs_reindex: bool = False

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
                known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
                cfg = cls(**{k: v for k, v in data.items() if k in known})
                if cfg._migrate():
                    cfg.save()
                return cfg
            except Exception:
                pass
        fresh = cls()
        fresh.config_version = CONFIG_VERSION
        return fresh

    # -- göç ---------------------------------------------------------------- #
    def _migrate(self) -> bool:
        """Eski yapılandırmayı güncel şemaya taşır; değişiklik olduysa True.

        Göç sonrası hangi alanların değiştiği :attr:`migration_notes` içinde
        toplanır; arayüz bunu kullanıcıya gösterip yeniden tarama önerir.
        """
        self.migration_notes = []
        self.migration_needs_reindex = False
        if self.config_version >= CONFIG_VERSION:
            return False

        # v0 -> v1: uzantı varsayılanı jpg/jpeg/png ile sınırlıydı; BMP, TIFF ve
        # WebP arşivleri sessizce indekslenmiyordu. Kullanıcı listeyi hiç
        # özelleştirmediyse (yani tam olarak eski varsayılansa) genişlet.
        if formats.normalize_all(self.extensions) == set(formats.LEGACY_DEFAULT_EXTENSIONS):
            self.extensions = list(formats.DEFAULT_EXTENSIONS)
            added = [e for e in formats.DEFAULT_EXTENSIONS
                     if e not in formats.LEGACY_DEFAULT_EXTENSIONS]
            self.migration_notes.append(
                "Desteklenen görsel formatları genişletildi: " + ", ".join(added)
            )
            self.migration_needs_reindex = True

        # v0 -> v1: skor ölçeği kalibre edildi (alakasız görseller artık %50
        # yerine ~0 alıyor). v1.0'ın eşikleri bu yeni ölçekte yanlış davranır,
        # bu yüzden kullanıcı özelleştirmediyse (eski varsayılansa) taşı.
        if self.hash_max_distance == 12:      # v1.0 varsayılanı — kopyadan öteye geçirmiyordu
            self.hash_max_distance = 22
            self.migration_notes.append(
                "Hash arama eşiği 12'den 22'ye çıkarıldı — aynı desenin farklı "
                "renkli/kırpılmış varyantları artık sonuçlara girebiliyor."
            )
        if self.color_alpha == 0.2:           # v1.0 varsayılanı
            self.color_alpha = 0.0
            self.migration_notes.append(
                "Renk ağırlığı 0'a çekildi — aynı desenin farklı renkli "
                "varyantları artık gömülmüyor. Kaydırıcıdan artırabilirsiniz."
            )
        if self.score_threshold == 0.0:       # v1.0 varsayılanı
            self.score_threshold = 0.35
            self.migration_notes.append(
                "Benzerlik skorları kalibre edildi; alakasız sonuçları elemek "
                "için skor eşiği 0.35'e ayarlandı."
            )

        self.config_version = CONFIG_VERSION
        return True

    def uses_embedding(self) -> bool:
        return self.backend in (BACKEND_EMBEDDING, BACKEND_HYBRID)

    def ext_set(self) -> set[str]:
        return formats.normalize_all(self.extensions)

    def hash_bits(self) -> int:
        return self.hash_size * self.hash_size

    def effective_max_distance(self) -> int:
        """Seçili hash boyutuna ölçeklenmiş arama eşiği (0 = sınırsız)."""
        from .core import hasher

        if self.hash_max_distance <= 0:
            return 0
        return hasher.scale_distance(self.hash_max_distance, self.hash_bits())

    def effective_duplicate_hamming(self) -> int:
        """Seçili hash boyutuna ölçeklenmiş kopya eşiği."""
        from .core import hasher

        return hasher.scale_distance(self.duplicate_hamming, self.hash_bits())
