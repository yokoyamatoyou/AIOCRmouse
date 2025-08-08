import os
from unittest.mock import AsyncMock, patch

import numpy as np
import cv2
import pytest

# Ensure src directory is in path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.ocr_bridge import GPT4oMiniVisionOCR


@pytest.fixture
def sample_text_image():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    cv2.putText(img, "A", (1, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return img


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
async def test_ocr_temperature_parameter(mock_post, sample_text_image):
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})
    mock_post.return_value.__aenter__.return_value = response

    ocr = GPT4oMiniVisionOCR()
    await ocr.run(sample_text_image)

    assert mock_post.call_args.kwargs["json"]["temperature"] == 0.1
