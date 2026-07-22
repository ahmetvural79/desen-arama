"""Görsel embedding çıkarımı — AI (derin) benzerlik arka ucu.

Birincil model: Meta **DINOv2 ViT-S/14** (384-dim), **ONNX Runtime** ile CPU'da
(opsiyonel DirectML/CUDA GPU) çalıştırılır. Model dosyası yoksa uygulamanın
çalışmaya devam etmesi için deterministik bir **klasik fallback** embedder
sağlanır (renk + doku + gradyan öznitelikleri). Böylece:

* Hash tabanlı hızlı mod hiçbir model olmadan çalışır (embedder gerekmez).
* AI modu, model varsa DINOv2; yoksa fallback ile en azından işlevseldir.

Provider seçimi ``onnxruntime`` kurulu sağlayıcılara göre yapılır; DirectML
(``DmlExecutionProvider``) mevcutsa ve istenirse GPU hızlanması alınır — kod
değişmeden yalnızca provider değişir.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from . import colorhist

log = logging.getLogger("desenarama.embedder")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Ön işleme (her iki embedder için ortak)
# --------------------------------------------------------------------------- #
def preprocess(rgb: np.ndarray, size: int = 224, resize_short: int = 256) -> np.ndarray:
    """RGB uint8 → normalize edilmiş CHW float32 tensörü (1, 3, size, size).

    Kısa kenarı ``resize_short``'a ölçekler, ortadan ``size`` kırpar
    (DINOv2 standardı). Böylece en-boy oranı korunur.
    """
    im = Image.fromarray(rgb)
    w, h = im.size
    scale = resize_short / min(w, h)
    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    im = im.crop((left, top, left + size, top + size))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(chw[None, :, :, :], dtype=np.float32)


class Embedder(ABC):
    dim: int
    name: str

    @abstractmethod
    def embed_batch(self, rgb_list: list[np.ndarray]) -> np.ndarray:
        """RGB dizilerinden (N, dim) float32 embedding matrisi (normalize edilmemiş)."""

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        return self.embed_batch([rgb])[0]


# --------------------------------------------------------------------------- #
# ONNX DINOv2 embedder
# --------------------------------------------------------------------------- #
class OnnxEmbedder(Embedder):
    name = "onnx-dinov2"

    def __init__(
        self,
        model_path: str,
        prefer_gpu: bool = False,
        intra_threads: int | None = None,
        input_size: int = 224,
    ) -> None:
        import onnxruntime as ort

        providers = self._select_providers(ort, prefer_gpu)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Bazı ONNX çevrimleri (ör. DINOv2 ViT-S) girdi batch boyutunu 1'e SABİTLER.
        # Bu durumda çıkarımı sabit boyutlu alt-batch'ler hâlinde koştururuz.
        bd = inp.shape[0] if inp.shape else None
        self.static_batch = bd if isinstance(bd, int) and bd >= 1 else None
        # Girdi boyutu modelden okunur (dinamik değilse); yoksa parametreye düşer.
        try:
            self.input_size = inp.shape[2] if isinstance(inp.shape[2], int) else input_size
        except Exception:
            self.input_size = input_size
        self._output_strategy: str | None = None  # "2d" | "cls"
        self._output_name: str | None = None
        self.dim = self._probe_dim()
        log.info("ONNX embedder hazır: %s | providers=%s | dim=%d",
                 model_path, self.session.get_providers(), self.dim)

    @staticmethod
    def _select_providers(ort, prefer_gpu: bool) -> list[str]:
        available = set(ort.get_available_providers())
        chosen: list[str] = []
        if prefer_gpu:
            for gpu in ("DmlExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"):
                if gpu in available:
                    chosen.append(gpu)
                    break
        chosen.append("CPUExecutionProvider")
        return chosen

    def _probe_dim(self) -> int:
        """Kukla girişle çıkışı çalıştırıp embedding boyutu ve stratejiyi belirler."""
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: dummy})
        out_meta = self.session.get_outputs()
        # 2B (B, D) bir çıkış varsa onu tercih et; yoksa 3B (B, N, D) CLS tokenı.
        best_idx, best_2d = None, None
        for i, arr in enumerate(outputs):
            if arr.ndim == 2:
                best_2d = i
                break
            if arr.ndim == 3 and best_idx is None:
                best_idx = i
        if best_2d is not None:
            self._output_strategy = "2d"
            self._output_name = out_meta[best_2d].name
            return outputs[best_2d].shape[1]
        if best_idx is not None:
            self._output_strategy = "cls"
            self._output_name = out_meta[best_idx].name
            return outputs[best_idx].shape[2]
        raise RuntimeError("ONNX model çıkışı beklenmeyen biçimde (2B/3B değil)")

    def _run(self, batch: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: batch})
        names = [o.name for o in self.session.get_outputs()]
        arr = outputs[names.index(self._output_name)]
        if self._output_strategy == "cls":
            arr = arr[:, 0, :]  # CLS token
        return np.ascontiguousarray(arr, dtype=np.float32)

    def embed_batch(self, rgb_list: list[np.ndarray]) -> np.ndarray:
        if not rgb_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        tensors = [preprocess(r, size=self.input_size) for r in rgb_list]
        full = np.concatenate(tensors, axis=0)  # (N, 3, S, S)
        n = full.shape[0]
        if self.static_batch is None:
            return self._run(full)
        # Sabit batch boyutu: bs'lik dilimlerde koştur; son dilimi sıfırla doldur.
        bs = self.static_batch
        out = np.empty((n, self.dim), dtype=np.float32)
        for start in range(0, n, bs):
            chunk = full[start:start + bs]
            if chunk.shape[0] < bs:  # son kısmi dilim — pad
                pad = np.zeros((bs - chunk.shape[0], *chunk.shape[1:]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
                res = self._run(chunk)[: n - start]
            else:
                res = self._run(chunk)
            out[start:start + res.shape[0]] = res
        return out


# --------------------------------------------------------------------------- #
# Klasik fallback embedder (model yokken)
# --------------------------------------------------------------------------- #
class FallbackEmbedder(Embedder):
    """Model olmadan çalışan deterministik klasik öznitelik çıkarıcı.

    DINOv2 kalitesinde değildir; ancak renk + kaba yapı + gradyan yönü
    öznitelikleriyle "benzer desen" için makul bir taban çizgisi sunar ve
    uygulamanın/test hattının model indirmeden çalışmasını sağlar.
    """

    name = "fallback-classic"

    def __init__(self, grid: int = 16, orient_bins: int = 8) -> None:
        self.grid = grid
        self.orient_bins = orient_bins
        # yapı(grid*grid) + gradyan yönü(grid/2 blok * orient_bins yaklaşık) + renk(72)
        self._struct_dim = grid * grid
        self._orient_dim = 4 * 4 * orient_bins
        self.dim = self._struct_dim + self._orient_dim + colorhist.DIM

    def _features(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # 1) Kaba yapı: grid'e küçült, kontrast normalize et.
        small = cv2.resize(gray, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        small = (small - small.mean()) / (small.std() + 1e-6)
        struct = small.flatten()

        # 2) Gradyan yönü histogramı: 4x4 blokta yön dağılımı (dokuyu yakalar).
        g = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) + np.pi) * (self.orient_bins / (2 * np.pi))
        ang = np.clip(ang.astype(int), 0, self.orient_bins - 1)
        orient = np.zeros((4, 4, self.orient_bins), dtype=np.float32)
        bh, bw = 16, 16
        for by in range(4):
            for bx in range(4):
                a = ang[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                m = mag[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                for b in range(self.orient_bins):
                    orient[by, bx, b] = m[a == b].sum()
        orient = orient.flatten()
        n = np.linalg.norm(orient)
        if n > 0:
            orient /= n

        # 3) Renk histogramı
        color = colorhist.histogram(rgb)

        return np.concatenate([struct, orient, color]).astype(np.float32)

    def embed_batch(self, rgb_list: list[np.ndarray]) -> np.ndarray:
        if not rgb_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._features(r) for r in rgb_list], axis=0)


# --------------------------------------------------------------------------- #
# Fabrika
# --------------------------------------------------------------------------- #
def load_embedder(
    model_path: str | None,
    prefer_gpu: bool = False,
    intra_threads: int | None = None,
) -> Embedder:
    """Model dosyası varsa ONNX embedder, yoksa fallback döndürür."""
    if model_path:
        import os

        if os.path.exists(model_path):
            try:
                return OnnxEmbedder(model_path, prefer_gpu=prefer_gpu, intra_threads=intra_threads)
            except Exception as e:  # bozuk model, uyumsuz opset vb.
                log.warning("ONNX model yüklenemedi (%s); fallback'e geçiliyor: %s", model_path, e)
    log.warning("AI modeli yok — klasik fallback embedder kullanılıyor.")
    return FallbackEmbedder()
