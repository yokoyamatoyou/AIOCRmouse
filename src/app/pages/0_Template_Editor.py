"""UI for creating and editing OCR templates."""

from __future__ import annotations

from typing import Dict, List

import inspect
from pathlib import Path

from PIL import Image
import streamlit as st
try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore
    HAS_CANVAS = True
except Exception:
    st_canvas = None  # type: ignore
    HAS_CANVAS = False

from app.cache_utils import get_template_manager, list_templates
from core import postprocess


NEW_TEMPLATE = "新規作成"
DEFAULT_CONFIDENCE_THRESHOLD = postprocess.DEFAULT_CONFIDENCE_THRESHOLD


def _load_initial_drawing(rois: Dict[str, Dict[str, List[int]]]) -> Dict[str, List[Dict[str, int]]]:
    """Convert ROI dict to drawable-canvas format."""
    objects: List[Dict[str, int]] = []
    for roi in rois.values():
        x, y, w, h = roi.get("box", [0, 0, 0, 0])
        objects.append(
            {
                "type": "rect",
                "left": x,
                "top": y,
                "width": w,
                "height": h,
                "fill": "rgba(255,0,0,0.3)",
                "stroke": "red",
            }
        )
    return {"version": "5.0.0", "objects": objects}


def main() -> None:
    st.title("Template Editor")

    manager = get_template_manager()
    templates = list_templates()

    selection = st.selectbox("テンプレートを選択", [NEW_TEMPLATE] + templates)
    template_name = st.text_input("テンプレート名", value="" if selection == NEW_TEMPLATE else selection)

    existing_rois: Dict[str, Dict[str, List[int]]] = {}
    existing_keywords: List[str] = []
    if selection != NEW_TEMPLATE:
        try:
            existing = manager.load(selection)
            existing_rois = existing.get("rois", {})
            existing_keywords = existing.get("keywords", [])
        except FileNotFoundError:
            st.warning("テンプレートが見つかりません。")

    uploaded = st.file_uploader("基準画像をアップロード", type=["png", "jpg", "jpeg"])
    if not HAS_CANVAS:
        st.warning(
            "描画キャンバス拡張が現在のStreamlitバージョンと互換しないため、手動入力モードでROIを設定します。"
        )

    keywords_text = st.text_input(
        "キーワード (カンマ区切り)",
        value=", ".join(existing_keywords),
    )

    st.markdown("---")
    st.subheader("品質ゲート設定（任意）")
    enable_qg = st.checkbox("品質ゲートを有効化する", value=True)
    qg_threshold = st.number_input(
        "鮮明度閾値 (Laplacian分散)",
        value=100.0,
        min_value=0.0,
        step=10.0,
        help="数値が大きいほど厳格に弾きます",
    )

    if uploaded is None:
        st.info("画像をアップロードしてください。")
        return

    image = Image.open(uploaded)

    roi_boxes: List[List[int]] = []
    if HAS_CANVAS and st_canvas is not None:
        initial = _load_initial_drawing(existing_rois) if existing_rois else None
        # Check whether this st_canvas supports initial_drawing
        canvas_kwargs = {}
        if initial:
            try:
                if "initial_drawing" in inspect.signature(st_canvas).parameters:
                    canvas_kwargs["initial_drawing"] = initial
            except (ValueError, TypeError):
                pass

        canvas_result = st_canvas(
            fill_color="rgba(255,0,0,0.3)",
            stroke_width=2,
            stroke_color="red",
            background_image=image,
            height=image.height,
            width=image.width,
            drawing_mode="rect",
            key="canvas",
            **canvas_kwargs,
        )

        if canvas_result and canvas_result.json_data:
            for obj in canvas_result.json_data.get("objects", []):
                if obj.get("type") == "rect":
                    roi_boxes.append(
                        [
                            int(obj.get("left", 0)),
                            int(obj.get("top", 0)),
                            int(obj.get("width", 0)),
                            int(obj.get("height", 0)),
                        ]
                    )
    else:
        st.subheader("ROI 手動入力モード")
        default_count = max(1, len(existing_rois))
        roi_count = int(
            st.number_input("ROIの数", min_value=1, max_value=100, value=default_count)
        )
        existing_list = list(existing_rois.values())
        for i in range(roi_count):
            st.markdown(f"ROI {i+1}")
            if i < len(existing_list):
                bx, by, bw, bh = existing_list[i].get("box", [0, 0, 0, 0])
            else:
                bx, by, bw, bh = 0, 0, 0, 0
            col1, col2, col3, col4 = st.columns(4)
            x = int(col1.number_input(f"x_{i+1}", value=int(bx), key=f"x_{i}", step=1))
            y = int(col2.number_input(f"y_{i+1}", value=int(by), key=f"y_{i}", step=1))
            w = int(col3.number_input(f"w_{i+1}", value=int(bw), key=f"w_{i}", step=1))
            h = int(col4.number_input(f"h_{i+1}", value=int(bh), key=f"h_{i}", step=1))
            roi_boxes.append([x, y, w, h])

    roi_definitions: Dict[str, Dict[str, object]] = {}
    for i, box in enumerate(roi_boxes):
        default_name = list(existing_rois.keys())[i] if i < len(existing_rois) else f"roi_{i + 1}"
        default_rule = (
            list(existing_rois.values())[i].get("validation_rule", "")
            if i < len(existing_rois)
            else ""
        )
        default_threshold = (
            list(existing_rois.values())[i].get(
                "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
            )
            if i < len(existing_rois)
            else DEFAULT_CONFIDENCE_THRESHOLD
        )

        default_field_type = (
            list(existing_rois.values())[i].get("field_type", "fixed")
            if i < len(existing_rois)
            else "fixed"
        )

        default_rule_type = ""
        default_rule_param = ""
        default_min = 0.0
        default_max = 0.0
        if default_rule.startswith("regex:"):
            default_rule_type = "regex"
            default_rule_param = default_rule[len("regex:") :]
        elif default_rule.startswith("range:"):
            default_rule_type = "range"
            try:
                min_str, max_str = default_rule[len("range:") :].split(",", 1)
                default_min = float(min_str)
                default_max = float(max_str)
            except ValueError:
                pass
        elif default_rule.startswith("enum:"):
            default_rule_type = "enum"
            default_rule_param = default_rule[len("enum:") :]

        name = st.text_input(
            f"ROI {i + 1} 名称", value=default_name, key=f"roi_name_{i}"
        )

        field_type = st.selectbox(
            f"ROI {i + 1} フィールドタイプ",
            ["fixed", "qualitative"],
            index=["fixed", "qualitative"].index(str(default_field_type).lower()),
            key=f"roi_field_type_{i}",
            help="固定: 日付・金額など厳格検証。定性: 備考等でLLM重視。",
        )

        rule_type_options = ["", "regex", "range", "enum"]
        rule_type = st.selectbox(
            f"ROI {i + 1} ルールタイプ",
            rule_type_options,
            index=rule_type_options.index(default_rule_type),
            key=f"roi_rule_type_{i}",
        )

        rule = ""
        if rule_type == "regex":
            pattern = st.text_input(
                f"ROI {i + 1} 正規表現",
                value=default_rule_param,
                key=f"roi_rule_regex_{i}",
            )
            if pattern:
                rule = f"regex:{pattern}"
        elif rule_type == "range":
            min_val = st.number_input(
                f"ROI {i + 1} 最小値",
                value=default_min,
                key=f"roi_rule_range_min_{i}",
            )
            max_val = st.number_input(
                f"ROI {i + 1} 最大値",
                value=default_max,
                key=f"roi_rule_range_max_{i}",
            )
            rule = f"range:{min_val},{max_val}"
        elif rule_type == "enum":
            enum_values = st.text_input(
                f"ROI {i + 1} 選択肢 (カンマ区切り)",
                value=default_rule_param,
                key=f"roi_rule_enum_{i}",
            )
            if enum_values:
                rule = f"enum:{enum_values}"

        threshold = st.number_input(
            f"ROI {i + 1} 信頼度閾値",
            value=float(default_threshold),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=f"roi_threshold_{i}",
        )

        roi_definitions[name] = {
            "box": box,
            "validation_rule": rule,
            "confidence_threshold": threshold,
            "field_type": field_type,
        }

    if st.button("保存"):
        if not template_name:
            st.error("テンプレート名を入力してください。")
        elif not roi_definitions:
            st.error("ROIを少なくとも1つ描画してください。")
        else:
            keywords = [
                kw.strip() for kw in keywords_text.split(",") if kw.strip()
            ]

            # save uploaded reference image
            suffix = Path(uploaded.name).suffix or ".png"
            image_path = manager.template_dir / f"{template_name}{suffix}"
            image.save(image_path)

            data = {
                "name": template_name,
                "keywords": keywords,
                "rois": roi_definitions,
                "template_image_path": str(image_path),
                # corrections are stored as a list for forward compatibility
                "corrections": [],
                "enable_quality_gate": enable_qg,
                "quality_threshold": qg_threshold,
            }
            manager.save(template_name, data)
            list_templates.clear()
            st.success("テンプレートを保存しました。")
            st.session_state["selected_template"] = template_name
            st.switch_page("pages/1_Main_OCR.py")


if __name__ == "__main__":
    main()
