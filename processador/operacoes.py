from __future__ import annotations

import cv2
import numpy as np

from .modelos import OperacaoMorfologica

MAPA_OPERACOES: dict = {
    OperacaoMorfologica.EROSAO:     lambda img, k: cv2.erode(img, k),
    OperacaoMorfologica.DILATACAO:  lambda img, k: cv2.dilate(img, k),
    OperacaoMorfologica.ABERTURA:   lambda img, k: cv2.morphologyEx(img, cv2.MORPH_OPEN,  k),
    OperacaoMorfologica.FECHAMENTO: lambda img, k: cv2.morphologyEx(img, cv2.MORPH_CLOSE, k),
}


def criar_kernel(tamanho: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (tamanho, tamanho))


def aplicar(operacao: OperacaoMorfologica, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return MAPA_OPERACOES[operacao](img, kernel)
