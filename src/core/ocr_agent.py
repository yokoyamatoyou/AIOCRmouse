from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
import json
from pathlib import Path
from typing import Dict, Tuple
import asyncio
import re

import cv2
import numpy as np

from . import preprocess
from .ocr_bridge import BaseOCR
from .ocr_processor import OCRProcessor

from .db_manager import DBManager
from .template_manager import TemplateManager
from .quality_gate import is_quality_sufficient


@dataclass
class OcrAgent:
    """Core class orchestrating the OCR workflow.

    The agent ties together template handling, preprocessing, OCR execution
    and database persistence into a single entry point.
    """

    db: DBManager
    templates: TemplateManager

    def process_document(
        self,
        image: np.ndarray,
        image_name: str,
        template_data: Dict[str, any],
        ocr_engine: BaseOCR,
        validator_engine: BaseOCR | None = None,
        job_id: int | None = None,
    ) -> Tuple[Dict[str, dict], str]:
        """Process a single document and persist results.

        Parameters
        ----------
        image:
            Source image as a ``numpy.ndarray``.
        image_name:
            Original filename of the uploaded image.
        template_data:
            Loaded template definition containing ROI information.
        ocr_engine:
            OCR engine implementation used for primary text extraction.
        validator_engine:
            Optional secondary OCR engine used for double-checking results.
        job_id:
            Existing database job identifier. If ``None``, a new job is created
            per document. When provided, all results are associated with the
            supplied job, enabling multiple images under a single job.

        Returns
        -------
        results: dict
            OCR results keyed by ROI name.
        workspace_dir: str
            Path to the workspace directory used for intermediate files.
        """

        now = datetime.now()
        doc_id = f"DOC_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}"
        workspace_dir = Path("workspace") / doc_id
        crops_dir = workspace_dir / "crops"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Save template for traceability
        with (workspace_dir / "template.json").open("w", encoding="utf-8") as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)

        rois = template_data.get("rois", {})

        # 品質ゲート（既定ON、テンプレートで閾値調整可能）
        threshold = float(template_data.get("quality_threshold", 100.0))
        enable_qg = template_data.get("enable_quality_gate", True)
        ok, reason = (True, "OK") if not enable_qg else is_quality_sufficient(image, threshold=threshold)

        if not ok:
            now = datetime.now()
            if job_id is None:
                job_id = self.db.create_job(template_data.get("name", ""), now.isoformat())

            doc_id = f"DOC_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}"
            workspace_dir = Path("workspace") / doc_id
            crops_dir = workspace_dir / "crops"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)

            # Save template for traceability
            with (workspace_dir / "template.json").open("w", encoding="utf-8") as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)

            # 品質チェック失敗は各ROIに対して空結果を保存
            rois = template_data.get("rois", {})
            results: Dict[str, dict] = {}
            for roi_name in rois.keys():
                result_id = self.db.add_result(
                    job_id,
                    image_name,
                    roi_name,
                    template_data.get("name", ""),
                    text_mini="",
                    text_nano="",
                    final_text="",
                    confidence_score=0.0,
                    status="QualityCheckFailed",
                )
                results[roi_name] = {
                    "text": "",
                    "confidence": 0.0,
                    "source_image": None,
                    "text_mini": "",
                    "text_nano": "",
                    "confidence_level": "low",
                    "needs_human": True,
                    "quality_failed": True,
                    "error": reason,
                    "result_id": result_id,
                }

            extract_path = workspace_dir / "extract.json"
            with extract_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            return results, str(workspace_dir)

        # Preprocess image and align ROIs
        corrected_image = preprocess.correct_skew(image)

        template_path = template_data.get("template_image_path")
        if template_path and Path(template_path).exists():
            template_img = cv2.imread(str(template_path))
            aligned_rois = preprocess.align_rois(template_img, corrected_image, rois)
        else:
            aligned_rois = rois

        if template_data.get("binarize"):
            corrected_image = preprocess.binarize(corrected_image)

        sanitized_rois: Dict[str, dict] = {}
        key_mapping: Dict[str, str] = {}
        for i, (key, roi_info) in enumerate(aligned_rois.items()):
            box = roi_info["box"]
            cropped = preprocess.crop_roi(corrected_image, box)
            safe_key = re.sub(r"[^\w.-]", "_", key)
            resolved_key = safe_key
            suffix = 1
            while resolved_key in sanitized_rois:
                resolved_key = f"{safe_key}_{suffix}"
                suffix += 1
            filename = f"P{i+1}_{resolved_key}.png"
            cv2.imwrite(str(crops_dir / filename), cropped)
            sanitized_rois[resolved_key] = roi_info
            key_mapping[resolved_key] = key

        # Execute OCR
        corrections = list(template_data.get("corrections", []))

        global_corrections_path = Path("workspace") / "corrections.jsonl"
        if global_corrections_path.exists():
            try:
                with global_corrections_path.open("r", encoding="utf-8") as cf:
                    for line in cf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            isinstance(item, dict)
                            and item.get("wrong")
                            and item.get("correct")
                        ):
                            corrections.append(
                                {"wrong": item["wrong"], "correct": item["correct"]}
                            )
            except OSError:
                pass

        processor = OCRProcessor(
            ocr_engine,
            str(workspace_dir),
            validator_engine=validator_engine,
            rois=sanitized_rois,
            key_mapping=key_mapping,
            corrections=corrections,
        )
        # Use existing event loop in Streamlit environment
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in a running event loop, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, processor.process_all())
                    results = future.result()
            else:
                results = asyncio.run(processor.process_all())
        except RuntimeError:
            # Fallback to asyncio.run if no event loop exists
            results = asyncio.run(processor.process_all())

        # Persist to database
        if job_id is None:
            job_id = self.db.create_job(template_data.get("name", ""), now.isoformat())

        for roi_name, info in results.items():
            result_id = self.db.add_result(
                job_id,
                image_name,
                roi_name,
                template_data.get("name", ""),
                text_mini=info.get("text_mini"),
                text_nano=info.get("text_nano"),
                final_text=info.get("text"),
                confidence_score=info.get("confidence"),
                composite_score=info.get("composite_score"),
                score_ocr=info.get("score_ocr"),
                score_rule=info.get("score_rule"),
                score_agreement=info.get("score_agreement"),
                status=info.get("confidence_level"),
            )
            info["result_id"] = result_id

        # Overwrite extract.json with result IDs included
        extract_path = workspace_dir / "extract.json"
        with extract_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results, str(workspace_dir)
