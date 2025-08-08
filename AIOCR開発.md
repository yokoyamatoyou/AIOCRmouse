\# AIコーディング指示書: LLMパラメータ最適化

\*\*発行日:\*\* 2025年8月7日  
\*\*担当AI:\*\* \[実行AIのIDを記載\]  
\*\*プロジェクト:\*\* AIOCR 精度向上イニシアチブ

\---

\#\# 1\. 現状分析 (Current State)

\- \*\*リポジトリ:\*\* \`yokoyamatoyou/aiocr\`  
\- \*\*最終合意事項:\*\* OCR精度と結果の再現性を最大化するため、OpenAI APIへのリクエスト時に\`temperature\`パラメータを\`0.1\`に固定する。  
\- \*\*関連ファイル群:\*\*  
    \- \`src/core/ocr\_bridge.py\`: APIリクエストのコアロジックを格納。  
    \- \`tests/\`: pytestによるテストスイートが存在。  
\- \*\*進捗:\*\* \`temperature\`変更に関する議論が完了し、実装方針が確定した段階。これから実際のコーディング、テスト、ドキュメント化に着手する。

\---

\#\# 2\. タスク概要 (Mission Briefing)

本タスクの最終目標は、\`temperature\`パラメータの変更を、\*\*テストによってその正当性を証明し、変更履歴をドキュメント化する\*\*という一連のプロセスを通じて、安全かつ確実にリポジトリに統合することである。

以下のフェーズを順番に、厳密に実行せよ。

\---

\#\# 3\. 開発フェーズ (Execution Phases)

\#\#\# \*\*フェーズ 1: コアロジックの修正 (Code Modification)\*\*

\*\*目標:\*\* \`temperature\`パラメータをAPIリクエストに組み込む。

\*\*手順:\*\*  
1\.  \`src/core/ocr\_bridge.py\`ファイルを開く。  
2\.  以下のコードブロックの内容で、既存のファイルを\*\*完全に上書き\*\*する。このコードには、\`payload\`に\`"temperature": 0.1\`を追加し、その理由を説明するコメントが含まれている。

    \`\`\`python  
    """OCR engine interfaces and implementations with asynchronous support."""

    from \_\_future\_\_ import annotations

    from abc import ABC, abstractmethod  
    import base64  
    from typing import Tuple

    import aiohttp  
    import cv2  
    import numpy as np

    from .config import settings

    class BaseOCR(ABC):  
        """すべてのOCRエンジンのための抽象基底クラス"""

        @abstractmethod  
        async def run(self, image: np.ndarray) \-\> Tuple\[str, float\]:  
            """画像を受け取り、(テキスト, 信頼度) のタプルを返す"""

        async def \_request\_openai(self, image: np.ndarray, model: str) \-\> Tuple\[str, float\]:  
            """共通のOpenAI Vision APIリクエストロジック"""  
            \_, buffer \= cv2.imencode(".png", image)  
            base64\_image \= base64.b64encode(buffer).decode("utf-8")

            headers \= {  
                "Authorization": f"Bearer {settings.OPENAI\_API\_KEY}",  
                "Content-Type": "application/json",  
            }  
            payload \= {  
                "model": model,  
                "messages": \[  
                    {  
                        "role": "user",  
                        "content": \[  
                            {  
                                "type": "text",  
                                "text": "この画像に書かれている日本語のテキストを、改行やスペースは無視して、全ての文字を繋げて書き出してください。",  
                            },  
                            {  
                                "type": "image\_url",  
                                "image\_url": {"url": f"data:image/png;base64,{base64\_image}"},  
                            },  
                        \],  
                    }  
                \],  
                "max\_tokens": 300,  
                \# \--- ここから変更 \---  
                \# temperatureを低く設定することで、出力のランダム性を抑え、  
                \# より決定的で事実に基づいた（=正確な）OCR結果を期待できます。  
                \# 変更日: 2025-08-07  
                "temperature": 0.1,  
                \# \--- ここまで変更 \---  
            }

            try:  
                async with aiohttp.ClientSession() as session:  
                    async with session.post(  
                        "\[https://api.openai.com/v1/chat/completions\](https://api.openai.com/v1/chat/completions)",  
                        headers=headers,  
                        json=payload,  
                    ) as resp:  
                        if resp.status \!= 200:  
                            error\_text \= await resp.text()  
                            raise RuntimeError(  
                                f"OpenAI API request failed ({resp.status}): {error\_text}"  
                            )  
                        data \= await resp.json()  
                text \= data\["choices"\]\[0\]\["message"\]\["content"\].strip()  
                \# temperatureを固定値にしたため、信頼度はAPIの応答に関わらず固定値を返す  
                confidence \= 0.99  
                return text, confidence  
            except Exception as e:  \# pragma: no cover \- network errors  
                print(f"OpenAI API呼び出し中にエラーが発生しました: {e}")  
                return "エラー", 0.0

    class DummyOCR(BaseOCR):  
        """ダミーのOCRエンジン。常に固定のテキストと信頼度を返す。"""

        async def run(self, image: np.ndarray) \-\> Tuple\[str, float\]:  
            h, w \= image.shape\[:2\]  
            dummy\_text \= f"ダミーテキスト({w}x{h})"  
            return dummy\_text, 0.95

    class GPT4oMiniVisionOCR(BaseOCR):  
        """GPT-4o mini を利用したOCRエンジン"""

        async def run(self, image: np.ndarray) \-\> Tuple\[str, float\]:  
            return await self.\_request\_openai(image, "gpt-4.1-mini")

    class GPT4oNanoVisionOCR(BaseOCR):  
        """GPT-4o nano を利用したOCRエンジン"""

        async def run(self, image: np.ndarray) \-\> Tuple\[str, float\]:  
            return await self.\_request\_openai(image, "gpt-4.1-nano")  
    \`\`\`

3\.  変更を保存してファイルを閉じる。

\*\*完了確認:\*\* \`src/core/ocr\_bridge.py\`の内容が上記と一致していることを確認する。

\#\#\# \*\*フェーズ 2: 変更を検証するテストの追加 (Test Implementation)\*\*

\*\*目標:\*\* フェーズ1の変更が正しく機能していることを証明し、将来の意図しない変更（デグレード）を防止する自動テストを実装する。

\*\*手順:\*\*  
1\.  \`tests/\`ディレクトリ内に、\`test\_ocr\_parameters.py\`という名前で新しいファイルを作成する。  
2\.  作成したファイルに、以下のテストコードを記述する。このテストは、API呼び出しをシミュレートし、リクエスト内容に\`"temperature": 0.1\`が含まれているかを検証するものである。

    \`\`\`python  
    import asyncio  
    from unittest.mock import patch, MagicMock

    import numpy as np  
    import pytest

    from core.ocr\_bridge import GPT4oMiniVisionOCR

    class AsyncMock(MagicMock):  
        """ aiohttp.ClientSession.post をモックするための非同期対応モック """  
        async def \_\_aenter\_\_(self):  
            \# 非同期コンテキストマネージャの\_\_aenter\_\_をモック  
            \# 内部でさらにモックされたレスポンスを返す  
            mock\_response \= MagicMock()  
            mock\_response.status \= 200  
            async def json():  
                return {"choices": \[{"message": {"content": "mocked response"}}\]}  
            mock\_response.json \= json  
            return mock\_response

        async def \_\_aexit\_\_(self, exc\_type, exc, tb):  
            pass

    @pytest.mark.asyncio  
    async def test\_openai\_request\_includes\_temperature():  
        """  
        OpenAI APIへのリクエストペイロードに temperature=0.1 が含まれていることを確認するテスト。  
        """  
        \# aiohttp.ClientSession.postを非同期対応のモックに置き換える  
        with patch("aiohttp.ClientSession.post", new\_callable=lambda: AsyncMock()) as mock\_post:  
            \# テスト用のダミー画像を作成  
            dummy\_image \= np.zeros((100, 100, 3), dtype=np.uint8)

            \# OCRエンジンをインスタンス化  
            ocr\_engine \= GPT4oMiniVisionOCR()

            \# OCR処理を実行  
            await ocr\_engine.run(dummy\_image)

            \# postメソッドが少なくとも1回呼び出されたことを確認  
            mock\_post.assert\_called\_once()

            \# 呼び出し時の引数を取得  
            \# mock\_post.call\_args は (args, kwargs) のタプルを返す  
            \_, kwargs \= mock\_post.call\_args

            \# kwargsの中に'json'キーが存在することを確認  
            assert "json" in kwargs

            \# ペイロード（jsonデータ）を取得  
            payload \= kwargs\["json"\]

            \# ペイロードに'temperature'キーが存在し、その値が0.1であることを確認  
            assert "temperature" in payload  
            assert payload\["temperature"\] \== 0.1  
    \`\`\`

3\.  ファイルを保存して閉じる。

\*\*完了確認:\*\* \`tests/test\_ocr\_parameters.py\`が作成され、内容が上記と一致していることを確認する。

\#\#\# \*\*フェーズ 3: 統合テストの実行 (Integration Testing)\*\*

\*\*目標:\*\* 新しく追加されたテストを含め、プロジェクト全体のテストスイートがすべて正常に完了することを確認する。

\*\*手順:\*\*  
1\.  ターミナルまたはコマンドプロンプトで、プロジェクトのルートディレクトリに移動する。  
2\.  仮想環境が有効化されていることを確認する。  
3\.  \`pytest\`コマンドを実行する。  
4\.  出力結果を注意深く確認し、すべてのテストが\`PASSED\`と表示され、\`FAILED\`や\`ERROR\`が存在しないことを確認する。

\*\*完了確認:\*\* \`pytest\`の実行結果が100%成功となる。

\#\#\# \*\*フェーズ 4: 変更履歴のドキュメント化 (Documentation)\*\*

\*\*目標:\*\* 「なぜこの変更が行われたのか」という文脈を、将来の参照のために公式な記録として残す。

\*\*手順:\*\*  
1\.  プロジェクトのルートディレクトリに、\`AGENT\_MODIFICATION\_LOG.md\`という名前で新しいファイルを作成する。  
2\.  作成したファイルに、以下の内容を記述する。

    \`\`\`markdown  
    \# AIOCR 変更履歴ログ

    このドキュメントは、AIアシスタントや将来の開発者がプロジェクトの重要な変更意図を理解し、追跡するための公式な記録です。

    \---

    \#\#\# \*\*変更日: 2025年8月7日\*\*

    \*\*変更目的:\*\*  
    OCR処理における精度と結果の再現性を向上させるため。

    \*\*変更内容:\*\*  
    すべてのOpenAI Vision APIへのリクエストにおいて、LLMの\`temperature\`パラメータを\`0.1\`に固定しました。

    \*\*理論的根拠:\*\*  
    \`temperature\`は、LLMの応答の「ランダム性」や「創造性」を制御するパラメータです。OCRのタスクは、画像に書かれたテキストを創造的に解釈することではなく、事実として正確に抽出することです。

    \`temperature\`を\`0.1\`のような低い値に設定することで、LLMは最も確率の高い、決定的で事実に忠実な応答を返すようになります。これにより、同じ画像に対して常に同じ、最も信頼性の高い結果が得られるようになり、システムの全体的な精度と安定性が向上します。

    \*\*関連ファイル:\*\*  
    \* \`src/core/ocr\_bridge.py\`: 実際のロジック変更が加えられたファイル。  
    \* \`tests/test\_ocr\_parameters.py\`: この変更が意図通りに適用されていることを恒久的に保証するための新しいテストファイル。  
    \`\`\`

3\.  ファイルを保存して閉じる。

\*\*完了確認:\*\* \`AGENT\_MODIFICATION\_LOG.md\`が作成され、内容が上記と一致していることを確認する。

\---

\*\*全フェーズ完了報告:\*\*  
すべてのフェーズが正常に完了したことを確認し、リポジトリが安定した状態であることを報告せよ。  
