import os
from pathlib import Path
import json

import cv2
import numpy as np
from datetime import datetime
from unittest.mock import patch

from core.db_manager import DBManager
from core.template_manager import TemplateManager
from core.ocr_agent import OcrAgent
from core.ocr_bridge import DummyOCR, BaseOCR
from core import preprocess


def test_ocr_agent_process_document(tmp_path):
    # change working directory to temporary path to avoid polluting repo
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    # create dummy image
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    template_data = {
        "name": "test",
        "enable_quality_gate": False,
        "rois": {"field": {"box": [0, 0, 10, 10], "confidence_threshold": 0.9}},
    }

    results, workspace = agent.process_document(
        image, "test.png", template_data, DummyOCR(), DummyOCR()
    )

    assert "field" in results
    assert results["field"]["result_id"] == 1
    assert Path(workspace).exists()
    db_results = db.fetch_results(1)
    assert db_results[0]["roi_name"] == "field"
    assert db_results[0]["text_mini"] == "ダミーテキスト(10x10)"
    assert db_results[0]["text_nano"] == "ダミーテキスト(10x10)"
    assert db_results[0]["confidence_score"] == 0.95
    assert db_results[0]["status"] == "high"
    assert db_results[0]["template_name"] == "test"
    with open(Path(workspace) / "extract.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["field"]["result_id"] == 1
    db.close()


def test_process_document_binarizes_when_requested(tmp_path, monkeypatch):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.full((20, 20, 3), 127, dtype=np.uint8)
    template_data = {
        "name": "test",
        "enable_quality_gate": False,
        "binarize": True,
        "rois": {"field": {"box": [0, 0, 20, 20], "confidence_threshold": 0.9}},
    }

    original = preprocess.binarize

    called = {"flag": False}

    def spy(img):
        called["flag"] = True
        return original(img)

    monkeypatch.setattr("core.ocr_agent.preprocess.binarize", spy)

    _, workspace = agent.process_document(image, "test.png", template_data, DummyOCR())

    assert called["flag"]
    crop_path = Path(workspace) / "crops" / "P1_field.png"
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(crop)).issubset({0, 255})
    db.close()


def test_process_document_creates_unique_directories(tmp_path):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {
        "name": "test",
        "rois": {"field": {"box": [0, 0, 10, 10], "confidence_threshold": 0.9}},
    }

    fixed_time = datetime(2025, 1, 1, 0, 0, 0)
    with patch("core.ocr_agent.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        _, workspace1 = agent.process_document(image, "a.png", template_data, DummyOCR())
        _, workspace2 = agent.process_document(image, "b.png", template_data, DummyOCR())

    assert workspace1 != workspace2
    assert Path(workspace1).exists()
    assert Path(workspace2).exists()
    db.close()


class FaultyOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "m1sread", 0.95


def test_ocr_agent_corrections(tmp_path):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {
        "name": "test",
        "enable_quality_gate": False,
        "rois": {
            "field": {
                "box": [0, 0, 10, 10],
                "confidence_threshold": 0.9,
            }
        },
        "corrections": [{"wrong": "m1sread", "correct": "misread"}],
    }

    results, _ = agent.process_document(image, "test.png", template_data, FaultyOCR())

    assert results["field"]["text"] == "misread"
    db.close()


def test_ocr_agent_global_corrections(tmp_path):
    os.chdir(tmp_path)

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    with open(workspace_dir / "corrections.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"wrong": "m1sread", "correct": "misread"}) + "\n")

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {
        "name": "test",
        "enable_quality_gate": False,
        "rois": {
            "field": {
                "box": [0, 0, 10, 10],
                "confidence_threshold": 0.9,
            }
        },
    }

    results, _ = agent.process_document(image, "test.png", template_data, FaultyOCR())

    assert results["field"]["text"] == "misread"
    db.close()


def test_quality_gate_skips_processing(tmp_path):
    # 品質ゲートONかつ低品質画像の場合、OCRをスキップしDBにQualityCheckFailedが記録される
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    # 極端にぼやけた画像（実質一定値）
    image = np.full((50, 50, 3), 127, dtype=np.uint8)

    template_data = {
        "name": "test",
        "enable_quality_gate": True,
        "quality_threshold": 10_000.0,  # 非常に高い閾値で必ず落ちるように
        "rois": {
            "a": {"box": [0, 0, 10, 10], "confidence_threshold": 0.9},
            "b": {"box": [10, 10, 10, 10], "confidence_threshold": 0.9},
        },
    }

    results, workspace = agent.process_document(image, "test.png", template_data, DummyOCR())

    # 各ROIが空結果で生成され、needs_humanとエラー理由が入る
    assert set(results.keys()) == {"a", "b"}
    for k, v in results.items():
        assert v["text"] == ""
        assert v["confidence"] == 0.0
        assert v["needs_human"] is True
        assert v["confidence_level"] == "low"
        assert "不鮮明" in v["error"]

    # DBにQualityCheckFailedで保存される
    # この呼び出しで作成されたjob_idは1のはず
    rows = list(db.fetch_results(1))
    assert len(rows) == 2
    assert {r["roi_name"] for r in rows} == {"a", "b"}
    assert all(r["status"] == "QualityCheckFailed" for r in rows)
    db.close()


def test_ocr_agent_multiple_images_single_job(tmp_path):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {
        "name": "test",
        "rois": {"field": {"box": [0, 0, 10, 10], "confidence_threshold": 0.9}},
    }

    job_id = db.create_job("test", "2025-01-01T00:00:00")
    for name in ["a.png", "b.png"]:
        agent.process_document(
            image,
            name,
            template_data,
            DummyOCR(),
            DummyOCR(),
            job_id=job_id,
        )

    db_results = db.fetch_results(job_id)
    assert len(db_results) == 2
    assert {r["image_name"] for r in db_results} == {"a.png", "b.png"}
    assert {r["result_id"] for r in db_results} == {1, 2}
    db.close()


def test_process_document_sanitizes_roi_keys(tmp_path):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {
        "name": "test",
        "enable_quality_gate": False,
        "rois": {
            "../unsafe": {"box": [0, 0, 10, 10], "confidence_threshold": 0.9}
        },
    }

    results, workspace = agent.process_document(
        image, "test.png", template_data, DummyOCR()
    )

    crops_dir = Path(workspace) / "crops"
    expected = crops_dir / "P1_.._unsafe.png"
    assert expected.exists()
    assert expected.resolve().parent == crops_dir.resolve()
    assert "../unsafe" in results
    db.close()


def test_process_document_resolves_key_collisions(tmp_path):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {
        "name": "test",
        "enable_quality_gate": False,
        "rois": {
            "field a": {"box": [0, 0, 10, 10], "confidence_threshold": 0.9},
            "field@a": {"box": [0, 0, 10, 10], "confidence_threshold": 0.9},
        },
    }

    results, workspace = agent.process_document(
        image, "test.png", template_data, DummyOCR()
    )

    crops_dir = Path(workspace) / "crops"
    assert (crops_dir / "P1_field_a.png").exists()
    assert (crops_dir / "P2_field_a_1.png").exists()
    assert set(results.keys()) == {"field a", "field@a"}
    db.close()
