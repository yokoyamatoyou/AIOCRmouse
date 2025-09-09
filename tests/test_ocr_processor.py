
import asyncio
import json
import os
import shutil
import time
import logging

import cv2
import numpy as np
import pytest

# srcディレクトリをパスに追加
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.ocr_bridge import DummyOCR, BaseOCR
from core.ocr_processor import OCRProcessor

@pytest.fixture
def setup_workspace():
    """ テスト用のワークスペースとダミーの切り出し画像を作成 """
    workspace_dir = "test_workspace"
    crops_dir = os.path.join(workspace_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # ダミーの切り出し画像を2つ作成
    img1 = np.zeros((50, 100, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(crops_dir, "P1_field_a.png"), img1)

    img2 = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(crops_dir, "P2_field_b.png"), img2)

    yield workspace_dir # テスト関数にワークスペースのパスを渡す

    # テスト終了後にクリーンアップ
    shutil.rmtree(workspace_dir)

def test_process_all(setup_workspace):
    """OCRProcessorが正しくJSONファイルを生成するかテスト"""
    workspace_dir = setup_workspace

    ocr_engine = DummyOCR()
    processor = OCRProcessor(ocr_engine, workspace_dir)

    results = asyncio.run(processor.process_all())

    assert "field_a" in results
    assert "field_b" in results
    assert results["field_a"]["text"] == "ダミーテキスト(100x50)"
    assert results["field_b"]["confidence"] == 0.95

    json_path = os.path.join(workspace_dir, "extract.json")
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "field_a" in data
    assert data["field_b"]["source_image"] == "P2_field_b.png"


def test_missing_image_returns_needs_human(tmp_path, caplog):
    """欠損画像は空文字で低信頼としてフラグ付けされる"""
    workspace_dir = tmp_path / "ws_missing"
    crops_dir = workspace_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 空ファイルを作成してcv2.imreadがNoneを返すようにする
    (crops_dir / "P1_field_a.png").write_bytes(b"")

    processor = OCRProcessor(DummyOCR(), str(workspace_dir))

    with caplog.at_level(logging.ERROR):
        results = asyncio.run(processor.process_all())

    entry = results["field_a"]
    assert entry["text"] == ""
    assert entry["confidence"] == 0.0
    assert entry["text_mini"] == ""
    assert entry["confidence_level"] == "low"
    assert entry["needs_human"] is True
    assert "Failed to load image" in caplog.text


class LowPrimaryOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "0000", 0.85


class HighSecondaryOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "1111", 0.95


def test_double_check_confidence(tmp_path):
    """二重チェックで高信頼のテキストを採用する"""

    workspace_dir = tmp_path / "ws"
    crops_dir = workspace_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((20, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(crops_dir / "P1_field_a.png"), img)

    rois = {
        "field_a": {
            "validation_rule": "regex:\\d{4}",
            "confidence_threshold": 0.9,
        }
    }

    processor = OCRProcessor(
        LowPrimaryOCR(),
        str(workspace_dir),
        validator_engine=HighSecondaryOCR(),
        rois=rois,
    )
    results = asyncio.run(processor.process_all())

    entry = results["field_a"]
    assert entry["text"] == "1111"
    assert entry["text_mini"] == "0000"
    assert entry["text_nano"] == "1111"
    assert entry["confidence"] == 0.95
    # 複合スコア閾値によっては"low"になることもある
    assert entry["confidence_level"] in {"medium", "low"}


class LowMatchOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "0000", 0.5


class SameOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "0000", 0.99


def test_validator_low_confidence(tmp_path):
    """バリデータ一致でも閾値未満ならneeds_human"""

    workspace_dir = tmp_path / "ws_low"
    crops_dir = workspace_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((20, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(crops_dir / "P1_field_a.png"), img)

    rois = {"field_a": {"confidence_threshold": 0.9}}

    processor = OCRProcessor(
        LowMatchOCR(),
        str(workspace_dir),
        validator_engine=SameOCR(),
        rois=rois,
    )
    results = asyncio.run(processor.process_all())

    entry = results["field_a"]
    assert entry["text"] == "0000"
    assert entry["text_mini"] == "0000"
    assert entry["text_nano"] == "0000"
    assert entry["confidence"] == (0.5 + 0.99) / 2
    assert entry["confidence_level"] == "low"
    assert entry["needs_human"] is True


class SleepOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        await asyncio.sleep(0.1)
        h, w = image.shape[:2]
        return f"ダミーテキスト({w}x{h})", 0.95


def test_process_all_parallel_execution(setup_workspace):
    """OCR処理が並列に実行されることを確認"""
    workspace_dir = setup_workspace
    ocr_engine = SleepOCR()
    processor = OCRProcessor(ocr_engine, workspace_dir)

    start = time.perf_counter()
    results = asyncio.run(processor.process_all())
    elapsed = time.perf_counter() - start

    assert elapsed < 0.18
    assert results["field_a"]["text"] == "ダミーテキスト(100x50)"
    assert results["field_b"]["text"] == "ダミーテキスト(120x60)"


def test_process_all_semaphore_limit(setup_workspace):
    """セマフォにより同時実行数が制限されることを確認"""
    workspace_dir = setup_workspace
    ocr_engine = SleepOCR()
    processor = OCRProcessor(ocr_engine, workspace_dir)

    start = time.perf_counter()
    results = asyncio.run(processor.process_all(max_concurrency=1))
    elapsed = time.perf_counter() - start

    assert elapsed > 0.18
    assert results["field_a"]["text"] == "ダミーテキスト(100x50)"
    assert results["field_b"]["text"] == "ダミーテキスト(120x60)"


def test_process_all_config_limit(monkeypatch, setup_workspace):
    """設定値で同時実行数が制限されることを確認"""
    from core import config

    workspace_dir = setup_workspace
    ocr_engine = SleepOCR()

    # デフォルト設定で制限を1にする
    monkeypatch.setattr(config.settings, "OCR_MAX_CONCURRENCY", 1)

    processor = OCRProcessor(ocr_engine, workspace_dir)

    start = time.perf_counter()
    results = asyncio.run(processor.process_all())
    elapsed = time.perf_counter() - start

    assert elapsed > 0.18
    assert results["field_a"]["text"] == "ダミーテキスト(100x50)"
    assert results["field_b"]["text"] == "ダミーテキスト(120x60)"


class LowConfOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "0000", 0.5


class HighConfOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "0000", 0.9


def test_custom_confidence_threshold(tmp_path):
    workspace_dir = tmp_path / "ws2"
    crops_dir = workspace_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(crops_dir / "P1_field_a.png"), img)

    rois = {"field_a": {"confidence_threshold": 0.4}}

    processor = OCRProcessor(HighConfOCR(), str(workspace_dir), rois=rois)
    results = asyncio.run(processor.process_all())

    assert "needs_human" not in results["field_a"]


def test_apply_corrections_substring():
    """Literal substrings should be replaced wherever they appear."""
    processor = OCRProcessor(
        DummyOCR(),
        "ws",
        corrections=[{"wrong": "cat", "correct": "dog"}],
    )

    assert processor._apply_corrections("cat") == "dog"
    assert processor._apply_corrections("concatenate") == "condogenate"


def test_apply_corrections_regex():
    """Regex patterns are supported when the ``regex`` flag is set."""
    processor = OCRProcessor(
        DummyOCR(),
        "ws",
        corrections=[{"wrong": r"\d{4}", "correct": "YYYY", "regex": True}],
    )

    assert processor._apply_corrections("2024-05-01") == "YYYY-05-01"


def test_apply_corrections_japanese_substring():
    """Japanese characters are handled in substring replacements."""
    processor = OCRProcessor(
        DummyOCR(),
        "ws",
        corrections=[{"wrong": "誤", "correct": "正"}],
    )

    assert processor._apply_corrections("誤解") == "正解"


def test_invalid_rule_error_in_result(setup_workspace):
    workspace_dir = setup_workspace
    rois = {"field_a": {"validation_rule": "foo:bar"}}
    processor = OCRProcessor(DummyOCR(), workspace_dir, rois=rois)
    results = asyncio.run(processor.process_all())
    entry = results["field_a"]
    assert entry["needs_human"] is True
    assert "error" in entry
    assert "Unknown validation rule" in entry["error"]
