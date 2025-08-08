
import asyncio
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# srcディレクトリをパスに追加
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.ocr_bridge import DummyOCR, GPT4oMiniVisionOCR, get_available_engines
from core.config import settings

# OpenAI APIキーが設定されているかチェック
API_KEY_SET = settings.OPENAI_API_KEY not in [None, "", "YOUR_API_KEY_HERE", "ここにあなたのOpenAI APIキーを入力してください"]

@pytest.fixture
def sample_text_image():
    """テスト用の日本語テキストを含む画像を生成"""
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img.fill(255)
    cv2.putText(img, "Test OCR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

def test_dummy_ocr(sample_text_image):
    """DummyOCRが期待通りに動作するかテスト"""
    ocr = DummyOCR()
    text, confidence = asyncio.run(ocr.run(sample_text_image))
    assert text == "ダミーテキスト(300x100)"
    assert confidence == 0.95

@pytest.mark.skipif(not API_KEY_SET, reason="OPENAI_API_KEYが設定されていません")
def test_gpt4o_mini_vision_ocr_integration(sample_text_image):
    """GPT4oMiniVisionOCRが実際にAPIと通信して結果を取得できるかテスト"""
    ocr = GPT4oMiniVisionOCR()
    text, confidence = asyncio.run(ocr.run(sample_text_image))

    if confidence == 0.0:
        pytest.skip("OpenAI API call failed")

    assert isinstance(text, str)
    assert text.strip() != ""
    assert isinstance(confidence, float)
    assert confidence > 0.0

class MockResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def json(self):
        return {"choices": [{"message": {"content": "モックされたOCR結果"}}]}

    async def text(self):
        return ""


@patch("aiohttp.ClientSession.post", return_value=MockResponse())
def test_gpt4o_mini_vision_ocr_mocked(mock_post, sample_text_image):
    """GPT4oMiniVisionOCRのAPI呼び出しをモックしてテスト"""
    ocr = GPT4oMiniVisionOCR()
    text, confidence = asyncio.run(ocr.run(sample_text_image))

    assert text == "モックされたOCR結果"
    assert confidence == 0.99
    mock_post.assert_called_once()


def test_engines_contains_gpt5_models():
    engines = get_available_engines()
    assert "GPT-5-mini-2025-08-07" in engines
    assert "GPT-5-nano-2025-08-07" in engines
