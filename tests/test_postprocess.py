import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import postprocess


def test_regex_rule_failure():
    text = "12345O7"  # O instead of 0
    normalized, needs_human = postprocess.postprocess_result(
        text,
        0.99,
        "regex:^\\d{7}$",
        postprocess.DEFAULT_CONFIDENCE_THRESHOLD,
    )
    assert normalized == "12345O7".replace(" ", "")
    assert needs_human


def test_range_rule_failure():
    text = "5"
    normalized, needs_human = postprocess.postprocess_result(
        text,
        0.99,
        "range:10,20",
        postprocess.DEFAULT_CONFIDENCE_THRESHOLD,
    )
    assert normalized == "5"
    assert needs_human


def test_enum_rule_failure():
    text = "B"
    normalized, needs_human = postprocess.postprocess_result(
        text,
        0.99,
        "enum:A,C",
        postprocess.DEFAULT_CONFIDENCE_THRESHOLD,
    )
    assert normalized == "B"
    assert needs_human


def test_low_confidence_flag():
    text = "1234567"
    normalized, needs_human = postprocess.postprocess_result(
        text,
        0.5,
        "regex:^\\d{7}$",
        postprocess.DEFAULT_CONFIDENCE_THRESHOLD,
    )
    assert normalized == "1234567"
    assert needs_human


def test_invalid_regex_rule_raises_error():
    with pytest.raises(postprocess.ValidationRuleError):
        postprocess.postprocess_result(
            "text",
            0.99,
            "regex:[0-9{2}",
            postprocess.DEFAULT_CONFIDENCE_THRESHOLD,
        )


def test_invalid_range_rule_raises_error():
    with pytest.raises(postprocess.ValidationRuleError):
        postprocess.postprocess_result(
            "5",
            0.99,
            "range:1,a",
            postprocess.DEFAULT_CONFIDENCE_THRESHOLD,
        )


def test_unknown_rule_type_raises_error():
    with pytest.raises(postprocess.ValidationRuleError):
        postprocess.postprocess_result(
            "text",
            0.99,
            "foo:bar",
            postprocess.DEFAULT_CONFIDENCE_THRESHOLD,
        )
