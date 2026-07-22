"""Komut satırı arayüzü — indeksleme, arama ve durum.

Örnekler:

    python -m desenarama index --root "D:/Desenler" --backend hash
    python -m desenarama index --root "\\\\sunucu\\desenler" --backend hybrid
    python -m desenarama search ornek.jpg --k 20 --alpha 0.3
    python -m desenarama status
    python -m desenarama gui
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __app_name__, __version__, config as cfg_mod
from .core import paths


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(paths.logs_dir() / "desenarama.log", encoding="utf-8"),
        ],
    )


def _load_engine(args):
    from .services.engine import Engine

    cfg = cfg_mod.AppConfig.load()
    # CLI ile geçersiz kılmalar
    if getattr(args, "backend", None):
        cfg.backend = args.backend
    if getattr(args, "model", None):
        cfg.model_key = args.model
    if getattr(args, "gpu", False):
        cfg.prefer_gpu = True
    if getattr(args, "root", None):
        cfg.library_roots = [r for r in args.root]
    if getattr(args, "no_tta", False):
        cfg.tta = False
    cfg.save()
    engine = Engine(cfg)
    engine.open()
    return engine


def cmd_index(args) -> int:
    engine = _load_engine(args)
    if not engine.config.library_roots:
        print("Hata: en az bir --root verilmeli.", file=sys.stderr)
        return 2
    from .services.indexer import IndexerService

    print(f"Kütüphane: {engine.config.library_roots}")
    print(f"Arka uç: {engine.config.backend} | hash: {engine.config.hash_algo} | TTA: {engine.config.tta}")

    last = {"done": -1}

    def cb(p):
        if p.phase == "process" and (p.done - last["done"] >= 50 or p.done == p.total):
            last["done"] = p.done
            pct = (100 * p.done / p.total) if p.total else 100
            print(f"\r  İşlenen: {p.done}/{p.total} (%{pct:.0f}) "
                  f"hata:{p.errors} hız:{p.rate:.1f}/sn", end="", flush=True)
        elif p.phase in ("scan", "finalize"):
            print(f"[{p.phase}] yeni:{p.new} değişen:{p.changed} kayıp:{p.missing}")

    svc = IndexerService(engine)
    prog = svc.reindex(progress_cb=cb)
    print()
    print(f"Bitti: {prog.done} işlendi, {prog.errors} hata, {prog.elapsed:.1f} sn "
          f"({prog.rate:.1f} imaj/sn)")
    print("Depo:", engine.store.stats())
    engine.close()
    return 0


def cmd_search(args) -> int:
    engine = _load_engine(args)
    from .services.search import SearchService

    svc = SearchService(engine)
    results = svc.search(query_path=args.query, k=args.k, color_alpha=args.alpha)
    if not results:
        print("Sonuç yok (kütüphane boş olabilir veya eşik altında).")
        engine.close()
        return 0
    print(f"En benzer {len(results)} sonuç:")
    for i, r in enumerate(results, 1):
        badge = " [KOPYA]" if r.is_duplicate else ""
        dim = " (eşik altı)" if r.below_threshold else ""
        ham = f" ham={r.hamming}" if r.hamming is not None else ""
        print(f"{i:3d}. skor={r.score:.3f} (desen={r.pattern_score:.3f} "
              f"renk={r.color_sim:.3f}{ham}){badge}{dim}  {r.path}")
    engine.close()
    return 0


def cmd_status(args) -> int:
    engine = _load_engine(args)
    st = engine.store.stats()
    cfg = engine.config
    print(f"{__app_name__} v{__version__}")
    print(f"Veri dizini : {paths.data_dir()}")
    print(f"Kütüphane   : {cfg.library_roots}")
    print(f"Arka uç     : {cfg.backend}")
    print(f"Hash        : {cfg.hash_algo} ({cfg.hash_size*cfg.hash_size}-bit)")
    print(f"Model       : {cfg.model_key} (GPU={cfg.prefer_gpu})")
    print(f"İmaj        : toplam={st['total']} ok={st['ok']} kayıp={st['missing']} hata={st['error']}")
    print(f"Vektör      : {engine.store.count_vectors()}")
    print(f"Hash indeksi: {len(engine.hash_index) if engine.hash_index else 0}")
    engine.close()
    return 0


def cmd_gui(args) -> int:
    from .gui.app import main as gui_main

    return gui_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="desenarama", description=f"{__app_name__} — halı deseni benzerlik arama")
    p.add_argument("--version", action="version", version=f"{__app_name__} {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="Kütüphaneyi (yeniden) indeksle")
    pi.add_argument("--root", action="append", help="Kütüphane klasörü (tekrarlanabilir)")
    pi.add_argument("--backend", choices=cfg_mod.BACKENDS)
    pi.add_argument("--model", choices=["dinov2-small", "dinov2-base"])
    pi.add_argument("--gpu", action="store_true")
    pi.add_argument("--no-tta", action="store_true")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="Sorgu imajıyla ara")
    ps.add_argument("query", help="Sorgu imajı yolu")
    ps.add_argument("--k", type=int, default=20)
    ps.add_argument("--alpha", type=float, default=None, help="Renk ağırlığı 0..1")
    ps.add_argument("--backend", choices=cfg_mod.BACKENDS)
    ps.add_argument("--no-tta", action="store_true")
    ps.set_defaults(func=cmd_search)

    pst = sub.add_parser("status", help="Durum ve yapılandırmayı göster")
    pst.set_defaults(func=cmd_status)

    pg = sub.add_parser("gui", help="Masaüstü arayüzü başlat")
    pg.set_defaults(func=cmd_gui)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
