
import os
import cv2
import json
import re
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple, List

from .ocr_bridge import BaseOCR
from . import postprocess
from .config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OCRProcessor:
    """OCR処理全体を管理するクラス"""

    def __init__(
        self,
        primary_engine: BaseOCR,
        workspace_dir: str,
        validator_engine: Optional[BaseOCR] = None,
        rois: Optional[Dict[str, Any]] = None,
        key_mapping: Optional[Dict[str, str]] = None,
        corrections: Optional[List[Dict[str, str]]] = None,
    ):
        self.primary_engine = primary_engine
        self.validator_engine = validator_engine
        self.workspace_dir = workspace_dir
        self.crops_dir = os.path.join(self.workspace_dir, "crops")
        self.rois = rois or {}
        self.key_mapping = key_mapping or {}
        self.corrections = corrections or []

    def _apply_corrections(self, text: str) -> str:
        """Apply stored corrections to ``text``.

        Each correction is a mapping containing ``wrong`` and ``correct``
        strings and an optional ``regex`` flag. If ``regex`` is true, the
        ``wrong`` value is treated as a regular expression pattern. Otherwise,
        the value is interpreted as a literal substring to replace. All
        occurrences are substituted.
        """

        for item in self.corrections:
            wrong = item.get("wrong")
            correct = item.get("correct")
            if not wrong or correct is None:
                continue
            pattern = wrong if item.get("regex") else re.escape(wrong)
            try:
                text = re.sub(pattern, correct, text)
            except re.error as exc:
                logger.warning("Invalid correction pattern %r: %s", wrong, exc)
        return text

    async def _process_file(self, filename: str) -> Tuple[str, Dict[str, Any]]:
        key = "_".join(filename.split("_")[1:]).replace(".png", "")
        orig_key = self.key_mapping.get(key, key)
        image_path = os.path.join(self.crops_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            logger.error("Failed to load image: %s", image_path)
            entry = {
                "text": "",
                "confidence": 0.0,
                "source_image": filename,
                "text_mini": "",
                "confidence_level": "low",
                "needs_human": True,
            }
            return orig_key, entry

        roi_def = self.rois.get(key, {})
        field_type = (roi_def.get("field_type") or "fixed").lower()

        if field_type == "qualitative":
            return await self._process_qualitative_field(image, filename, key, orig_key, roi_def)
        else:
            return await self._process_fixed_field(image, filename, key, orig_key, roi_def)

    async def _process_fixed_field(
        self,
        image: Any,
        filename: str,
        key: str,
        orig_key: str,
        roi_def: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        primary_text, primary_conf = await self.primary_engine.run(image)
        norm_primary = self._apply_corrections(
            postprocess.normalize_text(primary_text)
        )

        rule = roi_def.get("validation_rule")
        threshold = roi_def.get(
            "confidence_threshold", postprocess.DEFAULT_CONFIDENCE_THRESHOLD
        )

        norm_secondary = None
        needs_human = False
        error_message: Optional[str] = None
        score_agreement = 1.0

        if self.validator_engine is not None:
            secondary_text, secondary_conf = await self.validator_engine.run(image)
            norm_secondary = self._apply_corrections(
                postprocess.normalize_text(secondary_text)
            )
            score_agreement = postprocess.text_similarity(norm_primary, norm_secondary)
            if norm_primary == norm_secondary:
                final_text = norm_primary
                confidence = (primary_conf + secondary_conf) / 2
                try:
                    valid = postprocess.check_validation(final_text, rule)
                except postprocess.ValidationRuleError as exc:
                    logger.error("Validation rule error for %s: %s", key, exc)
                    valid = False
                    error_message = str(exc)
                needs_human = confidence < threshold or not valid
                confidence_level = "low" if needs_human else "high"
            else:
                if secondary_conf > primary_conf:
                    final_text = norm_secondary
                    confidence = secondary_conf
                else:
                    final_text = norm_primary
                    confidence = primary_conf
                try:
                    valid = postprocess.check_validation(final_text, rule)
                except postprocess.ValidationRuleError as exc:
                    logger.error("Validation rule error for %s: %s", key, exc)
                    valid = False
                    error_message = str(exc)
                needs_human = confidence < threshold or not valid
                confidence_level = "low" if needs_human else "medium"
        else:
            try:
                final_text, needs_human = postprocess.postprocess_result(
                    primary_text, primary_conf, rule, threshold
                )
            except postprocess.ValidationRuleError as exc:
                logger.error("Validation rule error for %s: %s", key, exc)
                final_text = postprocess.normalize_text(primary_text)
                needs_human = True
                error_message = str(exc)
            final_text = self._apply_corrections(final_text)
            confidence = primary_conf
            confidence_level = "low" if needs_human else "high"

        # スコア算出
        score_ocr = max(min(confidence, 1.0), 0.0)
        try:
            valid_rule = postprocess.check_validation(final_text, rule)
        except postprocess.ValidationRuleError:
            valid_rule = False
        score_rule = 1.0 if valid_rule else 0.0
        if norm_secondary is None:
            score_agreement = 1.0  # 単一モデルの場合は一致度を1とする

        # セッション上書きがあれば優先
        try:
            import streamlit as st  # noqa: WPS433
            w_ocr = float(st.session_state.get("SCORE_WEIGHT_OCR", settings.SCORE_WEIGHT_OCR))
            w_rule = float(st.session_state.get("SCORE_WEIGHT_RULE", settings.SCORE_WEIGHT_RULE))
            w_agree = float(st.session_state.get("SCORE_WEIGHT_AGREEMENT", settings.SCORE_WEIGHT_AGREEMENT))
            decision_threshold = float(st.session_state.get("COMPOSITE_DECISION_THRESHOLD", settings.COMPOSITE_DECISION_THRESHOLD))
        except Exception:
            w_ocr = settings.SCORE_WEIGHT_OCR
            w_rule = settings.SCORE_WEIGHT_RULE
            w_agree = settings.SCORE_WEIGHT_AGREEMENT
            decision_threshold = settings.COMPOSITE_DECISION_THRESHOLD

        composite = postprocess.calculate_composite_score(
            score_ocr, score_rule, score_agreement, w_ocr, w_rule, w_agree
        )
        if composite < decision_threshold:
            needs_human = True
            confidence_level = "low"

        entry = {
            "text": final_text,
            "confidence": confidence,
            "source_image": filename,
            "text_mini": norm_primary,
            "confidence_level": confidence_level,
            "composite_score": composite,
            "score_ocr": score_ocr,
            "score_rule": score_rule,
            "score_agreement": score_agreement,
        }
        if norm_secondary is not None:
            entry["text_nano"] = norm_secondary
        if needs_human:
            entry["needs_human"] = True
        if error_message:
            entry["error"] = error_message

        return orig_key, entry

    async def _process_qualitative_field(
        self,
        image: Any,
        filename: str,
        key: str,
        orig_key: str,
        roi_def: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        # 現段階では固定フィールドと同一の処理。将来的にロジックを分岐させる。
        return await self._process_fixed_field(image, filename, key, orig_key, roi_def)

    async def process_all(self, max_concurrency: Optional[int] = None) -> dict:
        """cropsディレクトリ内の画像を並行処理し、結果をJSONにまとめる

        Args:
            max_concurrency: 同時に実行するタスク数の上限。 ``None`` の場合は
                設定値 ``settings.OCR_MAX_CONCURRENCY`` が使用される。どちらも
                ``None`` の場合は制限なし。
        """

        if max_concurrency is None:
            max_concurrency = settings.OCR_MAX_CONCURRENCY

        crop_files = sorted(f for f in os.listdir(self.crops_dir) if f.endswith(".png"))

        semaphore: Optional[asyncio.Semaphore] = None
        if isinstance(max_concurrency, int) and max_concurrency > 0:
            semaphore = asyncio.Semaphore(max_concurrency)

        async def sem_task(filename: str):
            if semaphore is not None:
                async with semaphore:
                    return await self._process_file(filename)
            return await self._process_file(filename)

        tasks = [sem_task(filename) for filename in crop_files]
        processed = await asyncio.gather(*tasks)
        results = {key: entry for key, entry in processed}

        output_path = os.path.join(self.workspace_dir, "extract.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results
