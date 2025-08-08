import logging
import re
from typing import Optional, Tuple
from difflib import SequenceMatcher


class ValidationRuleError(Exception):
    """Raised when a validation rule is malformed or unsupported."""

DEFAULT_CONFIDENCE_THRESHOLD = 0.9

FULLWIDTH_MAP = str.maketrans(
    "０１２３４５６７８９－",
    "0123456789-"
)


logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """簡易的なテキスト正規化"""
    if text is None:
        return ""
    text = text.strip()
    text = text.translate(FULLWIDTH_MAP)
    text = text.replace(" ", "").replace("\u3000", "")
    return text


def check_validation(text: str, rule: Optional[str]) -> bool:
    """Check text against validation rule.

    Raises:
        ValidationRuleError: If the rule cannot be parsed or is unsupported.
    """
    if not rule:
        return True

    if rule.startswith("regex:"):
        pattern = rule[len("regex:") :]
        try:
            return re.fullmatch(pattern, text) is not None
        except re.error as e:
            raise ValidationRuleError(f"Invalid regex rule '{rule}': {e}") from e

    if rule.startswith("range:"):
        values = rule[len("range:") :]
        try:
            min_val_str, max_val_str = values.split(",", 1)
            min_val = float(min_val_str)
            max_val = float(max_val_str)
            numeric = float(text)
            return min_val <= numeric <= max_val
        except (ValueError, TypeError) as e:
            raise ValidationRuleError(f"Invalid range rule '{rule}': {e}") from e

    if rule.startswith("enum:"):
        options = [v.strip() for v in rule[len("enum:") :].split(",") if v.strip()]
        if not options:
            raise ValidationRuleError(f"Invalid enum rule '{rule}': no options provided")
        return text in options

    raise ValidationRuleError(f"Unknown validation rule: {rule}")


def postprocess_result(
    text: str,
    confidence: float,
    rule: Optional[str],
    threshold: float,
) -> Tuple[str, bool]:
    norm_text = normalize_text(text)
    try:
        valid = check_validation(norm_text, rule)
    except ValidationRuleError as exc:
        logger.error("Validation rule error: %s", exc)
        raise
    needs_human = confidence < threshold or not valid
    return norm_text, needs_human


def text_similarity(a: str, b: str) -> float:
    """Return a similarity ratio between two texts in [0, 1]."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def calculate_composite_score(
    score_ocr: float,
    score_rule: float,
    score_agreement: float,
    w_ocr: float,
    w_rule: float,
    w_agreement: float,
) -> float:
    """Weighted composite score in [0, 1]. Weights are normalized internally."""
    # Clamp individual scores
    score_ocr = min(max(score_ocr, 0.0), 1.0)
    score_rule = min(max(score_rule, 0.0), 1.0)
    score_agreement = min(max(score_agreement, 0.0), 1.0)

    total = max(w_ocr + w_rule + w_agreement, 1e-6)
    w_ocr_n = w_ocr / total
    w_rule_n = w_rule / total
    w_agree_n = w_agreement / total
    composite = score_ocr * w_ocr_n + score_rule * w_rule_n + score_agreement * w_agree_n
    return float(min(max(composite, 0.0), 1.0))
