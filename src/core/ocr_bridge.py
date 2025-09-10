"""OCR engine interfaces and implementations with asynchronous support."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from typing import Tuple, Dict, Type

import aiohttp
import cv2
import numpy as np

from .config import settings


class BaseOCR(ABC):
    """すべてのOCRエンジンのための抽象基底クラス"""

    @abstractmethod
    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        """画像を受け取り、(テキスト, 信頼度) のタプルを返す"""

    async def _request_openai(self, image: np.ndarray, model: str, prompt_type: str = "standard") -> Tuple[str, float]:
        """共通のOpenAI Vision APIリクエストロジック"""
        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        # プロンプトタイプに応じたテキストを選択
        prompts = {
            "standard": "この画像に書かれている日本語のテキストを、改行やスペースは無視して、全ての文字を繋げて書き出してください。",
            "handwriting": "この画像に書かれている手書きの日本語テキストを正確に読み取ってください。文字の形が不規則でも、できるだけ正確に認識し、改行やスペースは無視して全ての文字を繋げて書き出してください。",
            "printed": "この画像に書かれている印刷された日本語テキストを正確に読み取ってください。印刷文字の特徴を活かして、改行やスペースは無視して全ての文字を繋げて書き出してください。",
            "mixed": "この画像に書かれている日本語テキストを読み取ってください。手書きと印刷が混在している可能性がありますが、それぞれの特徴を考慮して正確に認識し、改行やスペースは無視して全ての文字を繋げて書き出してください。",
            "form": "この画像は帳票やフォームです。各項目の値を正確に読み取ってください。手書きの場合は文字の形が不規則でも正確に認識し、印刷文字の場合はその特徴を活かして読み取ってください。改行やスペースは無視して、各項目の値を正確に書き出してください。"
        }
        
        text_prompt = prompts.get(prompt_type, prompts["standard"])

        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 300,
            # Low temperature ensures deterministic, reproducible OCR
            "temperature": 0.1,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"OpenAI API request failed ({resp.status}): {error_text}"
                        )
                    data = await resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            confidence = 0.99
            return text, confidence
        except Exception as e:  # pragma: no cover - network errors
            print(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
            return "エラー", 0.0


class DummyOCR(BaseOCR):
    """ダミーのOCRエンジン。常に固定のテキストと信頼度を返す。"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        h, w = image.shape[:2]
        dummy_text = f"ダミーテキスト({w}x{h})"
        return dummy_text, 0.95


class GPT4oMiniVisionOCR(BaseOCR):
    """GPT-4o mini を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, settings.OPENAI_MODEL_STANDARD)


class GPT4oNanoVisionOCR(BaseOCR):
    """GPT-4o nano を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, settings.OPENAI_MODEL_FAST)


class GPT5MiniVisionOCR(BaseOCR):
    """GPT-5 mini を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, settings.OPENAI_MODEL_HIGH_QUALITY)


class GPT5NanoVisionOCR(BaseOCR):
    """GPT-5 nano を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, settings.OPENAI_MODEL_ULTRA_FAST)


class HandwritingOptimizedOCR(BaseOCR):
    """手書き文字特化のOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, settings.OPENAI_MODEL_HANDWRITING, "handwriting")


class FormOptimizedOCR(BaseOCR):
    """帳票・フォーム特化のOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, settings.OPENAI_MODEL_FORM, "form")


class MixedContentOCR(BaseOCR):
    """手書き・印刷混在コンテンツ用のOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, settings.OPENAI_MODEL_MIXED, "mixed")


# エンジンの一覧を名前→クラスで公開（UI側が動的に取得するため）
ENGINES: Dict[str, Type[BaseOCR]] = {
    # 既定: 一次OCRは nano、検証・ダブルチェックに mini を想定
    "GPT-5-nano-2025-08-07": GPT5NanoVisionOCR,
    "GPT-5-mini-2025-08-07": GPT5MiniVisionOCR,
    # 特化型OCRエンジン
    "Handwriting-Optimized": HandwritingOptimizedOCR,
    "Form-Optimized": FormOptimizedOCR,
    "Mixed-Content": MixedContentOCR,
    # 旧モデルも残しておく（後方互換 / デモ用）
    "GPT-4.1-nano": GPT4oNanoVisionOCR,
    "GPT-4.1-mini": GPT4oMiniVisionOCR,
    "DummyOCR": DummyOCR,
}


def get_available_engines() -> Dict[str, Type[BaseOCR]]:
    return dict(ENGINES)

