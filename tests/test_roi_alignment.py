import os
import sys
from pathlib import Path
import logging

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.db_manager import DBManager
from core.template_manager import TemplateManager
from core.ocr_agent import OcrAgent
from core.ocr_bridge import DummyOCR
from core import preprocess


def test_roi_alignment_with_shift(tmp_path, monkeypatch):
    os.chdir(tmp_path)

    # avoid skew correction for this test
    monkeypatch.setattr(preprocess, "correct_skew", lambda img: img)

    # create template image with distinctive features
    template_img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.putText(template_img, "TEST", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.circle(template_img, (150, 150), 20, (0, 0, 0), -1)
    template_path = tmp_path / "template.png"
    cv2.imwrite(str(template_path), template_img)

    # shift image
    M = np.float32([[1, 0, 20], [0, 1, 10]])
    shifted_img = cv2.warpAffine(template_img, M, (200, 200), borderValue=(255, 255, 255))

    # prepare environment
    db = DBManager(str(tmp_path / "ocr.db"))
    db.initialize()
    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    template_data = {
        "name": "test",
        "template_image_path": str(template_path),
        "rois": {
            "field": {"box": [40, 60, 80, 40], "confidence_threshold": 0.9}
        },
    }

    results, workspace = agent.process_document(
        shifted_img, "shifted.png", template_data, DummyOCR(), DummyOCR()
    )

    crop_path = Path(workspace) / "crops" / "P1_field.png"
    assert crop_path.exists()
    cropped = cv2.imread(str(crop_path))
    expected = shifted_img[70:110, 60:140]
    assert np.array_equal(cropped, expected)
    assert results["field"]["text"] == "ダミーテキスト(80x40)"
    db.close()


def test_roi_alignment_out_of_bounds_trimmed():
    """ROIs translated outside image bounds are clipped when cropped."""

    # create template with features
    template_img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.putText(template_img, "TEST", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.circle(template_img, (150, 150), 20, (0, 0, 0), -1)

    # shift image left/up so ROI goes out of bounds
    M = np.float32([[1, 0, -60], [0, 1, -30]])
    shifted_img = cv2.warpAffine(template_img, M, (200, 200), borderValue=(255, 255, 255))

    rois = {"field": {"box": [40, 60, 80, 40]}}
    aligned = preprocess.align_rois(template_img, shifted_img, rois)

    box = aligned["field"]["box"]
    assert box[0] < 0  # ensure ROI extends beyond image

    cropped = preprocess.crop_roi(shifted_img, box)
    expected = shifted_img[30:70, 0:60]
    assert cropped.shape == expected.shape
    assert np.array_equal(cropped, expected)


def test_align_rois_warns_on_failure(caplog):
    """Ensure warning is logged when feature matching fails."""

    template_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.putText(template_img, "A", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    blank_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    rois = {"field": {"box": [10, 10, 20, 20]}}

    with caplog.at_level(logging.WARNING):
        aligned = preprocess.align_rois(template_img, blank_img, rois)

    assert aligned == rois
    assert any("original ROI coordinates" in m for m in caplog.messages)


def test_roi_alignment_with_rotation_and_scale(monkeypatch):
    """align_rois handles rotation and scaling differences."""

    # avoid skew correction for this test
    monkeypatch.setattr(preprocess, "correct_skew", lambda img: img)

    template_img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.putText(template_img, "TEST", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.circle(template_img, (150, 150), 20, (0, 0, 0), -1)

    # rotate and scale image
    M = cv2.getRotationMatrix2D((100, 100), 15, 1.2)
    transformed_img = cv2.warpAffine(template_img, M, (200, 200), borderValue=(255, 255, 255))

    rois = {"field": {"box": [40, 60, 80, 40]}}

    aligned = preprocess.align_rois(template_img, transformed_img, rois)

    x, y, w, h = rois["field"]["box"]
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, M)
    xs = transformed[:, 0, 0]
    ys = transformed[:, 0, 1]
    exp_x, exp_y = xs.min(), ys.min()
    exp_w, exp_h = xs.max() - exp_x, ys.max() - exp_y

    box = aligned["field"]["box"]
    assert abs(box[0] - int(round(exp_x))) <= 1
    assert abs(box[1] - int(round(exp_y))) <= 1
    assert abs(box[2] - int(round(exp_w))) <= 1
    assert abs(box[3] - int(round(exp_h))) <= 1


def test_align_rois_template_matching_fallback(monkeypatch):
    """Fallback template matching aligns ROIs when ORB fails."""

    template_img = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(template_img, (20, 20), (60, 60), (0, 0, 0), -1)

    # scale and translate to create target image
    M = np.array([[1.1, 0, 5], [0, 1.1, 8]], dtype=np.float32)
    target_img = cv2.warpAffine(template_img, M, (100, 100), borderValue=(255, 255, 255))

    rois = {"field": {"box": [20, 20, 40, 40]}}

    class DummyORB:
        def detectAndCompute(self, *args, **kwargs):
            return [], None

    monkeypatch.setattr(cv2, "ORB_create", lambda *args, **kwargs: DummyORB())

    aligned = preprocess.align_rois(template_img, target_img, rois)
    x, y, w, h = aligned["field"]["box"]
    assert x == int(round(20 * 1.1 + 5))
    assert y == int(round(20 * 1.1 + 8))
    assert w == int(round(40 * 1.1))
    assert h == int(round(40 * 1.1))
