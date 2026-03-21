from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score, jaccard_score


def calcular_metricas(referencia: np.ndarray, resultado: np.ndarray) -> dict:
    """
    Compara `resultado` contra `referencia` (ambas uint8 binárias 0/255).
    Retorna: pixel_accuracy, jaccard (IoU), f1 (Dice), ssim.
    """
    ref = (referencia.flatten() / 255).astype(int)
    res = (resultado.flatten()  / 255).astype(int)

    return {
        "pixel_accuracy": float(np.mean(ref == res)),
        "jaccard":        float(jaccard_score(ref, res, zero_division=0)),
        "f1":             float(f1_score(ref, res, zero_division=0)),
        "ssim":           float(ssim(referencia, resultado, data_range=255)),
    }
