"""Halı-benzeri sentetik test seti üreteci.

Gerçek arşiv mantığını taklit eden desenler üretir: tekrarlayan motifler,
bordürler, madalyonlar. Her "aile" için varyantlar üretilir:

* orijinal
* farklı renk (colorway) — aynı desen, farklı palet
* döndürülmüş tarama (90°)
* yeniden sıkıştırılmış/küçültülmüş kopya (birebir kopya senaryosu)

Böylece Recall@k, TTA katkısı ve renk re-ranking gerçekçi biçimde ölçülebilir.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def _motif(rng, size, color):
    """Tek bir motif karosu üretir (geometrik, tekrarlanabilir)."""
    tile = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy = size / 2, size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    # eşmerkezli baklava/elmas motifi
    diamond = np.abs(xx - cx) + np.abs(yy - cy)
    ring = (np.sin(diamond * rng.uniform(0.3, 0.6)) > 0.2).astype(np.float32)
    for c in range(3):
        tile[:, :, c] = ring * color[c]
    # köşe noktaları
    dot = ((xx - cx) ** 2 + (yy - cy) ** 2) < (size * 0.12) ** 2
    tile[dot] = color
    return tile


def make_pattern(seed: int, palette, size=256, tile=64):
    """Bir desen ailesi üretir: seed deseni belirler, palette rengi belirler."""
    rng = np.random.default_rng(seed)
    reps = size // tile
    color = np.array(palette, dtype=np.float32)
    bg = color * 0.25
    img = np.tile(bg, (size, size, 1))
    motif = _motif(rng, tile, color)
    for i in range(reps):
        for j in range(reps):
            sub = motif if (i + j) % 2 == 0 else np.rot90(motif).copy()
            img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = sub
    # bordür
    b = max(4, size // 40)
    border_color = color * 0.8
    img[:b, :] = border_color
    img[-b:, :] = border_color
    img[:, :b] = border_color
    img[:, -b:] = border_color
    return np.clip(img, 0, 255).astype(np.uint8)


PALETTES = {
    "kirmizi": (200, 40, 40),
    "mavi": (40, 60, 200),
    "yesil": (40, 160, 60),
    "altin": (200, 170, 60),
}


def build(out_dir: str, n_families: int = 12) -> dict:
    """Test setini diske yazar. Dönüş: sorgu senaryoları için etiket haritası."""
    os.makedirs(out_dir, exist_ok=True)
    labels: dict[str, int] = {}  # dosya adı -> aile id
    manifest = {"families": {}, "queries": []}

    for fam in range(n_families):
        seed = 1000 + fam
        base_palette = list(PALETTES.values())[fam % len(PALETTES)]
        # orijinal
        orig = make_pattern(seed, base_palette)
        _save(out_dir, f"fam{fam}_orig.png", orig, labels, fam)
        # farklı renk (aynı desen, başka palet)
        alt_palette = list(PALETTES.values())[(fam + 1) % len(PALETTES)]
        recolor = make_pattern(seed, alt_palette)
        _save(out_dir, f"fam{fam}_recolor.png", recolor, labels, fam)
        # döndürülmüş tarama
        rot = np.rot90(orig).copy()
        _save(out_dir, f"fam{fam}_rot90.png", rot, labels, fam)
        # birebir kopya (küçültülüp yeniden kaydedilmiş)
        small = np.asarray(Image.fromarray(orig).resize((160, 160)).resize((256, 256)))
        _save(out_dir, f"fam{fam}_copy.jpg", small, labels, fam, jpeg=True)

        manifest["families"][fam] = [
            f"fam{fam}_orig.png", f"fam{fam}_recolor.png",
            f"fam{fam}_rot90.png", f"fam{fam}_copy.jpg",
        ]

    return {"labels": labels, "manifest": manifest, "dir": out_dir}


def _save(out_dir, name, arr, labels, fam, jpeg=False):
    im = Image.fromarray(arr)
    path = os.path.join(out_dir, name)
    if jpeg:
        im.save(path, quality=85)
    else:
        im.save(path)
    labels[name] = fam


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "test_dataset"
    info = build(out)
    print(f"{len(info['labels'])} imaj üretildi -> {out}")
