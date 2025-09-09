
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    OPENAI_API_KEY: str = "YOUR_API_KEY_HERE"
    # デフォルトのOCR同時実行数。Noneの場合は制限なし。
    OCR_MAX_CONCURRENCY: int | None = None
    # 複合スコアの重み（0-1で調整し、内部で正規化）
    SCORE_WEIGHT_OCR: float = 0.5
    SCORE_WEIGHT_RULE: float = 0.3
    SCORE_WEIGHT_AGREEMENT: float = 0.2
    # 複合スコアの確定閾値（下回ると needs_human=True）
    COMPOSITE_DECISION_THRESHOLD: float = 0.8

settings = Settings()
