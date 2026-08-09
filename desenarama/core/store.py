"""SQLite metadata deposu.

Tüm metadata (yol, mtime, boyut, algısal hash'ler, renk histogramı, FAISS
satır no, thumbnail adı, durum) tek bir yerel ``.db`` dosyasında tutulur.
Vektörler FAISS indeksinde ayrı tutulur; buradaki ``vec_row`` eşlemeyi sağlar.

Depo yerel diske yazıldığından WAL modu güvenle kullanılır (eşzamanlı okuma +
tek yazar). Yazma erişimi bir kilitle serileştirilir; böylece indeksleyici
iş parçacıklarından güvenle çağrılabilir.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass

import logging

from . import paths

log = logging.getLogger("desenarama.store")

#: 1 → 2: ``vectors`` tablosu imaj başına tek vektör yerine **karo** (tile)
#: taşıyor. Vektörler türetilmiş veridir; şema atlarken yeniden üretilirler.
SCHEMA_VERSION = 2

#: Tam kareyi (karo bölmesi olmayan) temsil eden karo numarası.
WHOLE_IMAGE_TILE = 0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
    id         INTEGER PRIMARY KEY,
    path       TEXT NOT NULL,
    key        TEXT UNIQUE NOT NULL,
    mtime      REAL,
    size       INTEGER,
    width      INTEGER,
    height     INTEGER,
    phash      BLOB,
    dhash      BLOB,
    ahash      BLOB,
    whash      BLOB,
    color      BLOB,
    vec_row    INTEGER,
    thumb      TEXT,
    status     TEXT NOT NULL DEFAULT 'ok',
    error      TEXT,
    indexed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_vecrow ON images(vec_row);
CREATE TABLE IF NOT EXISTS vectors(
    image_id INTEGER NOT NULL,
    tile     INTEGER NOT NULL DEFAULT 0,   -- 0 = tam kare, 1..N = karo
    vec      BLOB NOT NULL,
    PRIMARY KEY(image_id, tile)
);
CREATE INDEX IF NOT EXISTS idx_vectors_image ON vectors(image_id);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class ImageRecord:
    id: int
    path: str
    key: str
    mtime: float
    size: int
    width: int
    height: int
    phash: bytes | None
    dhash: bytes | None
    ahash: bytes | None
    whash: bytes | None
    color: bytes | None
    vec_row: int | None
    thumb: str | None
    status: str
    error: str | None
    indexed_at: float


_HASH_COLS = ("phash", "dhash", "ahash", "whash")


class ImageStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(paths.data_dir() / "library.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate_schema()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.set_meta("schema_version", str(SCHEMA_VERSION))

    def _migrate_schema(self) -> None:
        """Eski şemaları güncel sürüme taşır.

        Vektörler **türetilmiş** veridir (kaynak görsellerden yeniden
        üretilebilir), bu yüzden şema atlarken tablo düşürülür ve embedding'ler
        yeniden hesaplanır. Metadata (yol, hash, thumbnail) korunur — asıl
        pahalı olan tarama tekrarlanmaz.
        """
        try:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:
            return  # meta tablosu yok => yeni veritabanı
        version = int(row["value"]) if row else 0
        if 0 < version < 2:
            log.info("Şema %d -> %d: vektör tablosu karo desteğiyle yeniden kuruluyor "
                     "(embedding'ler yeniden hesaplanacak).", version, SCHEMA_VERSION)
            self._conn.execute("DROP TABLE IF EXISTS vectors")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- meta / settings ---------------------------------------------------- #
    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        cur = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_setting(self, key: str, default=None):
        cur = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return json.loads(row["value"]) if row else default

    # -- imaj kayıtları ----------------------------------------------------- #
    def get_by_key(self, key: str) -> ImageRecord | None:
        cur = self._conn.execute("SELECT * FROM images WHERE key=?", (key,))
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def get_by_id(self, image_id: int) -> ImageRecord | None:
        cur = self._conn.execute("SELECT * FROM images WHERE id=?", (image_id,))
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def existing_keys(self) -> dict[str, tuple[float, int]]:
        """Tüm kayıtların {key: (mtime, size)} eşlemesi — artımlı fark için."""
        cur = self._conn.execute("SELECT key, mtime, size FROM images")
        return {r["key"]: (r["mtime"], r["size"]) for r in cur.fetchall()}

    def upsert(self, **fields) -> int:
        """Bir imaj kaydını ekler/günceller; ``id`` döndürür.

        ``key`` zorunludur. Verilmeyen alanlar dokunulmadan bırakılır (INSERT'te
        NULL/varsayılan olur).
        """
        key = fields["key"]
        cols = list(fields.keys())
        placeholders = ", ".join("?" for _ in cols)
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "key")
        sql = (
            f"INSERT INTO images({', '.join(cols)}) VALUES({placeholders}) "
            f"ON CONFLICT(key) DO UPDATE SET {update_clause}"
        )
        with self._lock:
            cur = self._conn.execute(sql, [fields[c] for c in cols])
            self._conn.commit()
            if cur.lastrowid:
                rec = self._conn.execute("SELECT id FROM images WHERE key=?", (key,)).fetchone()
                return rec["id"]
        rec = self._conn.execute("SELECT id FROM images WHERE key=?", (key,)).fetchone()
        return rec["id"]

    def mark_status(self, key: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE images SET status=?, error=? WHERE key=?", (status, error, key)
            )
            self._conn.commit()

    def delete_keys(self, keys: list[str]) -> None:
        with self._lock:
            for k in keys:
                row = self._conn.execute("SELECT id FROM images WHERE key=?", (k,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM vectors WHERE image_id=?", (row["id"],))
            self._conn.executemany("DELETE FROM images WHERE key=?", [(k,) for k in keys])
            self._conn.commit()

    # -- vektörler ---------------------------------------------------------- #
    def upsert_vector(self, image_id: int, vec: bytes,
                      tile: int = WHOLE_IMAGE_TILE) -> None:
        self.upsert_vectors([(image_id, tile, vec)])

    def upsert_vectors(self, items: list[tuple[int, int, bytes]]) -> None:
        """Toplu vektör yazımı — ``(image_id, tile, vec)`` (tek commit).

        Aynı imaj için önceki karolar **silinir**: karo ızgarası ayarlardan
        değiştiğinde eski karoların ortada kalması (orphan) engellenir.
        """
        if not items:
            return
        image_ids = {i for i, _, _ in items}
        with self._lock:
            self._conn.executemany(
                "DELETE FROM vectors WHERE image_id=?", [(i,) for i in image_ids]
            )
            self._conn.executemany(
                "INSERT INTO vectors(image_id, tile, vec) VALUES(?, ?, ?) "
                "ON CONFLICT(image_id, tile) DO UPDATE SET vec=excluded.vec",
                items,
            )
            self._conn.commit()

    def iter_vectors(self):
        """Durumu 'ok' olan imajların (image_id, tile, vec_blob) üçlülerini verir."""
        sql = (
            "SELECT v.image_id AS image_id, v.tile AS tile, v.vec AS vec FROM vectors v "
            "JOIN images i ON i.id = v.image_id WHERE i.status='ok' "
            "ORDER BY v.image_id, v.tile"
        )
        for row in self._conn.execute(sql):
            yield row["image_id"], row["tile"], row["vec"]

    def get_vectors_for(self, image_ids) -> dict[int, list[bytes]]:
        """Verilen imajların tüm karo vektörlerini **tek sorguda** getirir.

        Hibrit yeniden sıralama önceden aday başına ayrı bir sorgu açıyordu
        (200 aday = 200 sorgu) ve bunu servis katmanından ``store._conn``
        özel alanına erişerek yapıyordu.
        """
        ids = list(image_ids)
        if not ids:
            return {}
        out: dict[int, list[bytes]] = {}
        # SQLite değişken sınırına (varsayılan 999) karşı parçalayarak sor.
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            sql = (f"SELECT image_id, vec FROM vectors "
                   f"WHERE image_id IN ({placeholders}) ORDER BY image_id, tile")
            for row in self._conn.execute(sql, chunk):
                out.setdefault(row["image_id"], []).append(row["vec"])
        return out

    def clear_vectors(self) -> None:
        """Tüm embedding'leri siler (model/ön işleme değiştiğinde geçersizleşirler)."""
        with self._lock:
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()

    def count_vectors(self) -> int:
        """Toplam **karo** sayısı (FAISS satır sayısına karşılık gelir)."""
        cur = self._conn.execute("SELECT COUNT(*) c FROM vectors")
        return int(cur.fetchone()["c"])

    def count_vector_images(self) -> int:
        """Embedding'i olan **imaj** sayısı (karo sayısından bağımsız)."""
        cur = self._conn.execute("SELECT COUNT(DISTINCT image_id) c FROM vectors")
        return int(cur.fetchone()["c"])

    def keys_missing_vectors(self) -> list[tuple[str, str, float, int]]:
        """Durumu 'ok' olup vektörü olmayan imajların (key, path, mtime, size) listesi.

        Kullanıcı hash modunda indeksleyip sonra AI/hibrit moduna geçerse, mevcut
        imajların embedding'i bu yolla tamamlanır (baştan tarama gerektirmez).
        """
        sql = (
            "SELECT i.key AS key, i.path AS path, i.mtime AS mtime, i.size AS size "
            "FROM images i LEFT JOIN vectors v ON v.image_id = i.id "
            "WHERE i.status='ok' AND v.image_id IS NULL"
        )
        return [(r["key"], r["path"], r["mtime"], r["size"]) for r in self._conn.execute(sql)]

    def iter_records(self, only_ok: bool = True):
        sql = "SELECT * FROM images"
        if only_ok:
            sql += " WHERE status='ok'"
        sql += " ORDER BY id"
        for row in self._conn.execute(sql):
            yield self._row_to_record(row)

    def count(self, status: str | None = None) -> int:
        if status:
            cur = self._conn.execute("SELECT COUNT(*) c FROM images WHERE status=?", (status,))
        else:
            cur = self._conn.execute("SELECT COUNT(*) c FROM images")
        return int(cur.fetchone()["c"])

    def stats(self) -> dict:
        return {
            "total": self.count(),
            "ok": self.count("ok"),
            "missing": self.count("missing"),
            "error": self.count("error"),
        }

    # -- yardımcılar -------------------------------------------------------- #
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            id=row["id"], path=row["path"], key=row["key"], mtime=row["mtime"],
            size=row["size"], width=row["width"], height=row["height"],
            phash=row["phash"], dhash=row["dhash"], ahash=row["ahash"], whash=row["whash"],
            color=row["color"], vec_row=row["vec_row"], thumb=row["thumb"],
            status=row["status"], error=row["error"], indexed_at=row["indexed_at"],
        )


def now() -> float:
    return time.time()
