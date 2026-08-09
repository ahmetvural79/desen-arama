"""Recall@k ölçüm harness'ı — arka uç ve TTA karşılaştırması.

Raporun Faz-0/Faz-1 gereksinimini karşılar: temsilî bir set üzerinde model/
arka uç kararını **veriyle** vermek. Sentetik set (make_dataset) aile etiketli
olduğundan Recall@k doğrudan hesaplanabilir. Gerçek arşiv için ``--dataset``
ile kendi klasörünüzü ve etiketlerinizi verebilirsiniz.

Metrik: bir sorgu imajı F ailesine aitse, top-k sonuçta (kendisi hariç) aynı
aileden en az bir üye bulunma oranı = "family-hit@k"; ayrıca aile üyelerinin
ortalama geri-çağırma oranı = "recall@k".

Kullanım:
    DESENARAMA_DATA_DIR=/tmp/bench python tests/bench.py --dataset /tmp/ds --out bench.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desenarama import config as cfg_mod
from desenarama.services.engine import Engine
from desenarama.services.indexer import IndexerService
from desenarama.services.search import SearchService


def family_of(name: str) -> str:
    # "fam3_orig.png" -> "fam3"
    return os.path.basename(name).split("_")[0]


def scenario_of(name: str) -> str:
    """Sorgu senaryosu — "fam3_recolor.png" -> "recolor".

    Senaryo kırılımı şart: toplam Recall@10 tek bir sayıya indirgendiğinde
    "kırpılmış sorgular tamamen çalışmıyor" gibi bir arıza, diğer senaryoların
    yüksek skorları arasında görünmez hâle gelir.
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem.split("_", 1)[1] if "_" in stem else "?"


def evaluate(engine, files, ks=(1, 3, 5, 10)):
    svc = SearchService(engine)
    hit = {k: 0 for k in ks}
    recall = {k: 0.0 for k in ks}
    # aile üye sayıları
    fam_counts: dict[str, int] = {}
    for f in files:
        fam_counts[family_of(f)] = fam_counts.get(family_of(f), 0) + 1

    per_scenario: dict[str, dict] = {}

    t0 = time.time()
    for f in files:
        fam = family_of(f)
        scen = scenario_of(f)
        results = svc.search(query_path=f, k=max(ks) + 1)
        # kendini çıkar
        others = [r for r in results if os.path.abspath(r.path) != os.path.abspath(f)]
        names = [os.path.basename(r.path) for r in others]
        denom = max(fam_counts[fam] - 1, 1)
        bucket = per_scenario.setdefault(scen, {"n": 0, "hit10": 0, "recall10": 0.0})
        bucket["n"] += 1
        for k in ks:
            topk = names[:k]
            same = [n for n in topk if family_of(n) == fam]
            if same:
                hit[k] += 1
            recall[k] += len(same) / denom
            if k == 10:
                bucket["hit10"] += 1 if same else 0
                bucket["recall10"] += len(same) / denom
    dt = time.time() - t0
    for b in per_scenario.values():
        b["hit10"] /= b["n"]
        b["recall10"] /= b["n"]
    n = len(files)
    return {
        "hit": {k: hit[k] / n for k in ks},
        "recall": {k: recall[k] / n for k in ks},
        "query_ms": 1000 * dt / n,
        "scenarios": per_scenario,
    }


def run_config(dataset_dir, backend, tta, embedder=None, tile_grid=0):
    cfg = cfg_mod.AppConfig()
    cfg.library_roots = [dataset_dir]
    cfg.backend = backend
    cfg.tta = tta
    cfg.tile_grid = tile_grid
    cfg.save()
    engine = Engine(cfg)
    if embedder is not None:
        engine._embedder = embedder
    engine.open()
    # temiz indeks
    IndexerService(engine).reindex()
    # Desteklenen tüm formatlar ölçüme girer (kırpma senaryosu BMP'dir).
    from desenarama.core import formats

    files = sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith(formats.SUPPORTED_EXTENSIONS)
    )
    res = evaluate(engine, files)
    res["n"] = len(files)
    engine.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="bench.md")
    ap.add_argument("--skip-embedding", action="store_true")
    ap.add_argument("--tiles", type=int, default=0, metavar="N",
                    help="N×N karo indekslemeyi de ölç (ör. 3). İndeksleme "
                         "süresini ~10× artırır ama kırpılmış sorguları çözer.")
    args = ap.parse_args()

    rows = []
    ks = (1, 3, 5, 10)

    print("Hash (TTA kapalı) ...")
    rows.append(("hash", "kapalı", run_config(args.dataset, "hash", False)))
    print("Hash (TTA açık) ...")
    rows.append(("hash", "açık", run_config(args.dataset, "hash", True)))

    if not args.skip_embedding:
        # Model yoksa ölçüm yapma: v1.0'da bu durumda sessizce zayıf yedek
        # çıkarıcı kullanılıyor ve "embedding" satırı DINOv2 sanılıyordu.
        from desenarama.core import models

        spec = models.resolve(cfg_mod.AppConfig().model_key)
        if not models.is_available(spec):
            print(f"AI modeli yok ({models.local_path(spec)}) — embedding ölçümü "
                  f"atlanıyor. Modeli indirin veya --skip-embedding kullanın.")
        else:
            print("Embedding (TTA kapalı) ...")
            rows.append(("embedding", "kapalı", run_config(args.dataset, "embedding", False)))
            print("Embedding (TTA açık) ...")
            rows.append(("embedding", "açık", run_config(args.dataset, "embedding", True)))
            if args.tiles:
                print(f"Embedding + {args.tiles}×{args.tiles} karo ...")
                rows.append((f"embedding+karo{args.tiles}", "açık",
                             run_config(args.dataset, "embedding", True,
                                        tile_grid=args.tiles)))
                print(f"Hibrit + {args.tiles}×{args.tiles} karo ...")
                rows.append((f"hibrit+karo{args.tiles}", "açık",
                             run_config(args.dataset, "hybrid", True,
                                        tile_grid=args.tiles)))

    # Markdown tablo
    lines = ["# Benchmark — Recall@k", "",
             f"Veri seti: `{args.dataset}` ({rows[0][2]['n']} imaj)", ""]
    lines.append("| Arka uç | TTA | hit@1 | hit@5 | hit@10 | recall@5 | recall@10 | sorgu(ms) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for backend, tta, r in rows:
        lines.append(
            f"| {backend} | {tta} | {r['hit'][1]:.2f} | {r['hit'][5]:.2f} | {r['hit'][10]:.2f} "
            f"| {r['recall'][5]:.2f} | {r['recall'][10]:.2f} | {r['query_ms']:.1f} |"
        )

    # Senaryo kırılımı: toplam recall tek sayıya indiğinde "kırpılmış sorgular
    # hiç çalışmıyor" gibi arızalar görünmez olur.
    scenarios = sorted({s for _, _, r in rows for s in r["scenarios"]})
    if scenarios:
        lines += ["", "## Sorgu senaryosuna göre recall@10", "",
                  "| Arka uç | TTA | " + " | ".join(scenarios) + " |",
                  "|---" * (len(scenarios) + 2) + "|"]
        for backend, tta, r in rows:
            cells = [f"{r['scenarios'][s]['recall10']:.2f}" if s in r["scenarios"] else "—"
                     for s in scenarios]
            lines.append(f"| {backend} | {tta} | " + " | ".join(cells) + " |")

    out = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print("\n" + out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
