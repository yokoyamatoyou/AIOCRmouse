
import pytest
import cv2
import numpy as np
import os

# srcディレクトリをパスに追加するために、プロジェクトルートからの相対パスを取得
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import preprocess

@pytest.fixture
def sample_image():
    """ テスト用の傾いたサンプル画像を生成 """
    # 真っ直ぐな画像を生成
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img.fill(255) # 白い背景
    cv2.putText(img, "Test Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 画像を10度傾ける
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def test_correct_skew(sample_image):
    """ 傾き補正が機能するかテスト """
    corrected = preprocess.correct_skew(sample_image)

    # 補正後の画像で再度Hough変換を実行し、角度を検証
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Hough変換のパラメータを調整して、短い直線も検出できるようにします
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10)

    # 直線が検出され、その角度がほぼ水平であることを確認
    assert lines is not None, "補正後の画像で直線が検出できませんでした"

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    assert abs(median_angle) < 1.5, f"補正後の角度が期待値から外れています: {median_angle:.2f}度"

def test_binarize(sample_image):
    """ 二値化が正しく行われるかテスト """
    binarized = preprocess.binarize(sample_image)
    assert len(binarized.shape) == 2 # グレースケールになっているか
    assert np.unique(binarized).size <= 2 # 0と255の2値になっているか

def test_crop_roi(sample_image):
    """ ROIの切り出しが正しく行われるかテスト """
    roi_box = [50, 75, 150, 50] # x, y, w, h
    cropped = preprocess.crop_roi(sample_image, roi_box)
    assert cropped.shape == (50, 150, 3) # (h, w, c)
