from __future__ import annotations

import cv2
import numpy as np


def check_sharpness(image: np.ndarray, threshold: float = 100.0) -> bool:
    """画像の鮮明度を簡易評価し、十分かどうかを返す。

    Laplacian の分散を用いた一般的な指標を利用する。

    Parameters
    ----------
    image:
        BGR またはグレースケール画像。
    threshold:
        合否判定の閾値。値が大きいほど厳しくなる。
    """

    if image is None or image.size == 0:
        return False

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= threshold


def is_quality_sufficient(image: np.ndarray, threshold: float = 100.0) -> tuple[bool, str]:
    """品質が十分かどうかの合否と理由を返す。

    Returns
    -------
    (ok, reason)
        ok が False の場合、reason には理由のメッセージが入る。
    """
    if image is None or image.size == 0:
        return False, "画像が空です"

    if not check_sharpness(image, threshold=threshold):
        return False, "画像が不鮮明です"

    return True, "OK"



