from __future__ import annotations

import numpy as np

from .modelos import TipoRuido


def adicionar_ruido(
    img:        np.ndarray,
    tipo:       TipoRuido = TipoRuido.SALT_PEPPER,
    intensidade: float    = 0.05,
    seed:       int       = 42,
) -> np.ndarray:
    if tipo == TipoRuido.SALT_PEPPER:
        return _salt_pepper(img, intensidade, seed)
    return _gaussiano(img, intensidade, seed)


def _salt_pepper(img: np.ndarray, intensidade: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    ruidosa = img.copy()
    h, w = img.shape
    n = int(intensidade * h * w)
    ys, xs = rng.randint(0, h, n), rng.randint(0, w, n)
    ruidosa[ys, xs] = 255
    ys, xs = rng.randint(0, h, n), rng.randint(0, w, n)
    ruidosa[ys, xs] = 0
    return ruidosa


def _gaussiano(img: np.ndarray, intensidade: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    std = intensidade * 255
    ruido = rng.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
