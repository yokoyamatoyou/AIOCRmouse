import asyncio
import os
import tempfile
import zipfile
import shutil
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from core.ocr_bridge import get_available_engines
from app.cache_utils import get_template_manager, get_db_manager, list_templates
from core.ocr_agent import OcrAgent
from core import preprocess


class LocalUploadedFile:
    """Simple wrapper to mimic Streamlit's UploadedFile for local files."""

    def __init__(self, path: str):
        self._path = path
        self.name = os.path.basename(path)

    def read(self) -> bytes:
        with open(self._path, "rb") as f:
            return f.read()


# テンプレート名と検出キーワードはテンプレートファイル内で管理
# TemplateManager を通じて読み込む


def main() -> None:
    st.set_page_config(page_title="AIOCR")
    st.title("AIOCR処理実行")

    # --- サイドバー ---
    st.sidebar.title("設定")
    st.sidebar.page_link("pages/1_Main_OCR.py", label="Main OCR")
    st.sidebar.page_link("pages/2_Review.py", label="Review & 修正")
    st.sidebar.page_link("pages/3_Dashboard.py", label="ダッシュボード")
    st.sidebar.page_link("pages/4_Admin.py", label="管理設定")

    # OCRエンジン選択
    engines = get_available_engines()
    engine_names = list(engines.keys())
    # 既定は一次OCRに指定された nano モデル
    default_index = engine_names.index("GPT-5-nano-2025-08-07") if "GPT-5-nano-2025-08-07" in engines else 0
    ocr_engine_choice = st.sidebar.selectbox(
        "OCRエンジンを選択",
        tuple(engine_names),
        index=default_index,
    )
    
    # 前処理モード選択
    preprocessing_modes = {
        "標準": "standard",
        "手書き文字特化": "handwriting", 
        "印刷文字特化": "printed",
        "混在コンテンツ": "mixed"
    }
    preprocessing_mode_choice = st.sidebar.selectbox(
        "前処理モード",
        list(preprocessing_modes.keys()),
        index=0,
    )
    
    # 前処理の有効/無効
    enable_preprocessing = st.sidebar.checkbox(
        "画像前処理を有効にする",
        value=True,
        help="画像の品質を自動評価し、必要に応じて前処理を適用します"
    )

    # --- メイン画面 ---

    # 1. アップロード形式の選択
    upload_mode = st.radio(
        "アップロード形式",
        ("画像ファイル", "ZIP/フォルダ"),
        horizontal=True,
    )

    uploaded_images = []
    temp_dir: str | None = None
    cleanup_dir: str | None = None
    if upload_mode == "画像ファイル":
        uploaded_images = st.file_uploader(
            "画像ファイルをアップロードしてください",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        ) or []
    else:
        uploaded_zip = st.file_uploader(
            "ZIPアーカイブをアップロードしてください",
            type=["zip"],
            accept_multiple_files=False,
        )
        folder_path = st.text_input("またはローカルフォルダパスを入力")
        st.caption("ZIPやフォルダは一時ディレクトリに展開され、含まれる画像が順次処理されます")
        if uploaded_zip is not None:
            temp_dir = tempfile.mkdtemp()
            zip_bytes = BytesIO(uploaded_zip.read())
            with zipfile.ZipFile(zip_bytes) as zf:
                zf.extractall(temp_dir)
            cleanup_dir = temp_dir
        elif folder_path:
            if os.path.isdir(folder_path):
                temp_dir = tempfile.mkdtemp()
                shutil.copytree(folder_path, temp_dir, dirs_exist_ok=True)
                cleanup_dir = temp_dir
            else:
                st.error("指定されたフォルダが見つかりません")
        if temp_dir:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        uploaded_images.append(LocalUploadedFile(os.path.join(root, file)))

    # テンプレート選択肢を準備
    template_manager = get_template_manager()
    template_names = list_templates()
    index = 0
    selected = st.session_state.get("selected_template")
    if selected in template_names:
        index = template_names.index(selected) + 1
    template_option = st.selectbox(
        "帳票テンプレートを選択",
        ["自動検出"] + template_names,
        index=index,
    )

    if uploaded_images and template_names:
        if st.button("OCR処理実行"):
            db = get_db_manager()
            agent = OcrAgent(db=db, templates=template_manager)

            # OCRエンジンを選択
            st.write(f"{ocr_engine_choice} でOCR処理を実行しています...")
            ocr_engine = engines[ocr_engine_choice]()
            # バリデータは mini を既定にする（ユーザの要望: 一次 nano / 二次 mini）
            validator_cls = engines.get("GPT-5-mini-2025-08-07") or list(engines.values())[0]
            nano_engine = validator_cls()
            
            # 前処理モードを設定
            preprocessing_mode = preprocess.PreprocessingMode(preprocessing_modes[preprocessing_mode_choice])

            # Create a single job to group all documents in this run
            job_id = db.create_job(template_option, datetime.now().isoformat())

            combined_results: dict[str, dict] = {}
            workspace_dirs: dict[str, str] = {}
            quality_failed_images: list[str] = []

            # 画像をメモリに読み込む
            loaded_images: list[tuple[str, np.ndarray]] = []
            for uploaded_image in uploaded_images:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                loaded_images.append((uploaded_image.name, image))

            try:
                with st.spinner('AI-OCR処理を実行中です (GPT-4.1-nanoでダブルチェック)...'):
                    template_map: dict[str, dict] = {}

                    if template_option == "自動検出":
                        st.write("全画像のテンプレートを自動検出しています...")

                        async def detect_all() -> list[tuple[str, str]]:
                            progress_bar = st.progress(0)
                            total = len(loaded_images)
                            completed = 0

                            async def detect(name: str, img: np.ndarray) -> tuple[str, str]:
                                nonlocal completed
                                text, _ = await nano_engine.run(img)
                                completed += 1
                                progress_bar.progress(completed / total)
                                return name, text

                            tasks = [detect(name, img) for name, img in loaded_images]
                            return await asyncio.gather(*tasks)

                        detection_results = asyncio.run(detect_all())
                        for name, text in detection_results:
                            detected = template_manager.detect_template(text)
                            if detected:
                                _, template_data = detected
                                template_map[name] = template_data
                            else:
                                st.warning(
                                    f"{name} のテンプレートを特定できません。テンプレートを選択するかスキップしてください。"
                                )
                                choice = st.selectbox(
                                    f"{name} 用テンプレート",
                                    ["スキップ"] + template_names,
                                    key=f"manual_template_{name}",
                                )
                                if choice != "スキップ":
                                    template_map[name] = template_manager.load(choice)
                                else:
                                    st.info(f"{name} はスキップされます。")
                    else:
                        static_template = template_manager.load(template_option)
                        template_map = {name: static_template for name, _ in loaded_images}

                    progress = st.progress(0)
                    total = len(loaded_images)
                    for idx, (name, image) in enumerate(loaded_images, start=1):
                        template_data = template_map.get(name)
                        if template_data is None:
                            st.info(f"{name} はテンプレート未選択のためスキップされました。")
                            progress.progress(idx / total)
                            continue
                        try:
                            ocr_results, workspace_dir = agent.process_document(
                                image,
                                name,
                                template_data,
                                ocr_engine,
                                validator_engine=nano_engine,
                                job_id=job_id,
                            )
                        except Exception as e:  # noqa: BLE001
                            st.error(f"{name} の処理中にエラーが発生しました: {e}")
                            progress.progress(idx / total)
                            continue
                        combined_results[name] = ocr_results
                        workspace_dirs[name] = workspace_dir
                        # 品質ゲートで除外された場合は計上
                        if all(v.get("quality_failed") for v in ocr_results.values()):
                            quality_failed_images.append(name)
                        progress.progress(idx / total)
            finally:
                if cleanup_dir:
                    shutil.rmtree(cleanup_dir)

            # 処理完了メッセージと結果を表示
            st.success("処理が完了しました！")
            st.subheader("作業ディレクトリ")
            for name, path in workspace_dirs.items():
                st.write(f"{name}: {os.path.abspath(path)}")

            st.subheader("OCR抽出結果 (extract.json)")
            st.json(combined_results)

            # 品質ゲートの集計
            processed_count = len(combined_results)
            quality_failed_count = len(quality_failed_images)
            analyzed_count = processed_count - quality_failed_count
            st.subheader("品質ゲート集計")
            st.write(f"解析対象画像数: {processed_count}")
            st.write(f"低品質で除外: {quality_failed_count}")
            st.write(f"実際に解析: {analyzed_count}")
            if quality_failed_images:
                st.write("低品質と判定されたファイル:")
                for n in quality_failed_images:
                    st.write(f"- {n}")

            st.subheader("信頼度とダブルチェック結果")
            for img_name, ocr_result in combined_results.items():
                st.markdown(f"### {img_name}")
                for field, info in ocr_result.items():
                    conf_score = info.get("confidence", 0.0)
                    level = info.get("confidence_level", "")
                    needs_human = info.get("needs_human", False)
                    icon = "✅" if not needs_human else "⚠️"
                    
                    # 前処理情報の表示
                    preprocessing_info = info.get("preprocessing_info", {})
                    applied_ops = preprocessing_info.get("applied_operations", [])
                    quality = preprocessing_info.get("quality", "unknown")
                    
                    st.write(f"{icon} {field}: 信頼度 {conf_score:.2f} ({level})")
                    if applied_ops:
                        st.caption(f"前処理: {', '.join(applied_ops)} | 品質: {quality}")
                    else:
                        st.caption(f"前処理: なし | 品質: {quality}")


if __name__ == "__main__":
    main()
