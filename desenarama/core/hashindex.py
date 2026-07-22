"""BK-tree — algısal hash'ler üzerinde hızlı Hamming-mesafesi araması.

czkawka ve imagededup, milyonlarca hash içinde eşik altındaki komşuları
bulmak için BK-tree kullanır. Metrik uzayında (Hamming) üçgen eşitsizliğini
kullanarak aramada ağacın büyük kısmını budar; böylece "eşik ≤ d olan tüm
komşular" sorgusu tüm koleksiyonu taramadan yanıtlanır.

Ayrıca küçük koleksiyonlar veya kesin sıralama için doğrusal (brute-force)
tarama da sağlanır; ikisi de aynı :func:`query` arayüzünü paylaşır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hasher import hamming


@dataclass
class _Node:
    key: int  # hash değeri
    value: int  # imaj id
    children: dict[int, "_Node"] = field(default_factory=dict)  # mesafe -> çocuk


class BKTree:
    """Hamming metriğiyle çalışan BK-tree.

    Aynı hash değerine sahip birden çok imaj olabileceğinden, her düğüm bir
    ``(hash, [id, ...])`` kovası gibi davranır: eşitlik (mesafe 0) çakışması
    çocuk yerine ``bucket`` listesine eklenir.
    """

    def __init__(self) -> None:
        self._root: _Node | None = None
        self._buckets: dict[int, list[int]] = {}  # hash -> aynı hash'li id'ler
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, hash_value: int, image_id: int) -> None:
        self._size += 1
        if hash_value in self._buckets:
            self._buckets[hash_value].append(image_id)
            return
        self._buckets[hash_value] = [image_id]
        node = _Node(key=hash_value, value=image_id)
        if self._root is None:
            self._root = node
            return
        cur = self._root
        while True:
            d = hamming(hash_value, cur.key)
            if d == 0:
                # Aynı hash zaten kovada; ağaç düğümü açmaya gerek yok.
                return
            child = cur.children.get(d)
            if child is None:
                cur.children[d] = node
                return
            cur = child

    def query(self, hash_value: int, max_distance: int) -> list[tuple[int, int]]:
        """Hamming mesafesi ``<= max_distance`` olan (image_id, mesafe) çiftleri.

        Sonuç mesafeye göre artan sırada döner (en benzer önce).
        """
        results: list[tuple[int, int]] = []
        if self._root is None:
            return results
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = hamming(hash_value, node.key)
            if d <= max_distance:
                for img_id in self._buckets.get(node.key, [node.value]):
                    results.append((img_id, d))
            # Üçgen eşitsizliği: yalnızca [d-max, d+max] aralığındaki çocuklara in.
            lo, hi = d - max_distance, d + max_distance
            for edge, child in node.children.items():
                if lo <= edge <= hi:
                    stack.append(child)
        results.sort(key=lambda t: t[1])
        return results


@dataclass
class HashSearchResult:
    image_id: int
    distance: int
    similarity: float  # 0..1 (1 = birebir)


class HashIndex:
    """Uygulama seviyesinde hash arama sarmalayıcısı.

    Küçük koleksiyonlarda (varsayılan ≤ 5000) BK-tree kurmadan doğrusal
    tarama yapar; büyük koleksiyonlarda BK-tree'yi kullanır. İki yol da
    aynı sonucu verir, yalnızca performans farkı vardır.
    """

    def __init__(self, hash_bits: int = 64, linear_threshold: int = 5000) -> None:
        self.hash_bits = hash_bits
        self.linear_threshold = linear_threshold
        self._items: list[tuple[int, int]] = []  # (image_id, hash)
        self._tree: BKTree | None = None

    def __len__(self) -> int:
        return len(self._items)

    def add(self, image_id: int, hash_value: int) -> None:
        self._items.append((image_id, hash_value))
        if self._tree is not None:
            self._tree.add(hash_value, image_id)

    def build(self) -> None:
        """Koleksiyon büyükse BK-tree kur (aksi halde doğrusal tarama yeter)."""
        if len(self._items) > self.linear_threshold:
            tree = BKTree()
            for image_id, h in self._items:
                tree.add(h, image_id)
            self._tree = tree

    def search(
        self, query_hash: int, k: int = 50, max_distance: int | None = None
    ) -> list[HashSearchResult]:
        """En benzer ``k`` imajı döndürür.

        ``max_distance`` verilirse yalnızca o Hamming eşiği altındakiler
        değerlendirilir (hızlı budama). Verilmezse tüm koleksiyon sıralanır.
        """
        bits = self.hash_bits
        if self._tree is not None and max_distance is not None:
            raw = self._tree.query(query_hash, max_distance)
            hits = [
                HashSearchResult(img_id, d, 1.0 - d / bits) for img_id, d in raw
            ]
            return hits[:k]

        # Doğrusal tarama
        scored: list[HashSearchResult] = []
        for image_id, h in self._items:
            d = int(query_hash ^ h).bit_count()
            if max_distance is not None and d > max_distance:
                continue
            scored.append(HashSearchResult(image_id, d, 1.0 - d / bits))
        scored.sort(key=lambda r: r.distance)
        return scored[:k]
