import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, ROOT)

from core.quality_gate import check_sharpness, is_quality_sufficient  # noqa: E402


def test_is_quality_sufficient_pass_and_fail():
    # 鮮明な画像（ランダムノイズで高周波成分を含む）
    sharp = (np.random.rand(128, 128) * 255).astype('uint8')
    ok, reason = is_quality_sufficient(sharp)
    assert ok is True
    assert reason == "OK"

    # ぼかした画像（ガウシアンぼかしでシャープネス低下）
    # ここでは一様画像を用いてシャープネス指標がほぼ0になることを期待
    blur = np.full((128, 128), 127, dtype='uint8')
    ok2, reason2 = is_quality_sufficient(blur)
    assert ok2 is False
    assert "不鮮明" in reason2

    # 空画像
    empty = np.zeros((0, 0), dtype='uint8')
    ok3, reason3 = is_quality_sufficient(empty)
    assert ok3 is False
    assert "空" in reason3



