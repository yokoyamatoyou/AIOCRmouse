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

    async def _request_openai(self, image: np.ndarray, model: str) -> Tuple[str, float]:
        """共通のOpenAI Vision APIリクエストロジック"""
        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

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
                            "text": "この画像に書かれている日本語のテキストを、改行やスペースは無視して、全ての文字を繋げて書き出してください。",
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
        return await self._request_openai(image, "gpt-4.1-mini")


class GPT4oNanoVisionOCR(BaseOCR):
    """GPT-4o nano を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, "gpt-4.1-nano")


class GPT5MiniVisionOCR(BaseOCR):
    """GPT-5 mini を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, "gpt-5-mini-2025-08-07")


class GPT5NanoVisionOCR(BaseOCR):
    """GPT-5 nano を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        return await self._request_openai(image, "gpt-5-nano-2025-08-07")


# エンジンの一覧を名前→クラスで公開（UI側が動的に取得するため）
ENGINES: Dict[str, Type[BaseOCR]] = {
    # 既定: 一次OCRは nano、検証・ダブルチェックに mini を想定
    "GPT-5-nano-2025-08-07": GPT5NanoVisionOCR,
    "GPT-5-mini-2025-08-07": GPT5MiniVisionOCR,
    # 旧モデルも残しておく（後方互換 / デモ用）
    "GPT-4.1-nano": GPT4oNanoVisionOCR,
    "GPT-4.1-mini": GPT4oMiniVisionOCR,
    "DummyOCR": DummyOCR,
}


def get_available_engines() -> Dict[str, Type[BaseOCR]]:
    return dict(ENGINES)

