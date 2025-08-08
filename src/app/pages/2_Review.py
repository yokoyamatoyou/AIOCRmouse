import os
import json
import streamlit as st

from app.cache_utils import get_db_manager, get_template_manager

st.title("レビュー")

WORKSPACE_DIR = "workspace"


def save_correction(item: dict, new_text: str, add_dict: bool) -> None:
    """Persist corrected text to JSON, DB and optional dictionaries."""
    item["data"][item["key"]]["text"] = new_text
    item["data"][item["key"]].pop("needs_human", None)
    with open(item["extract_path"], "w", encoding="utf-8") as f:
        json.dump(item["data"], f, ensure_ascii=False, indent=4)

    db = get_db_manager()
    # Mark the entry as confirmed in the database as the reviewer approved
    # the corrected text.
    db.update_result(item["result_id"], new_text, status="confirmed")
    st.info("DBを更新しました")

    corrections_path = os.path.join(WORKSPACE_DIR, "corrections.jsonl")
    if add_dict:
        entry = {"wrong": item["text"], "correct": new_text}
        with open(corrections_path, "a", encoding="utf-8") as cf:
            cf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        template_json = os.path.join(WORKSPACE_DIR, item["doc"], "template.json")
        try:
            with open(template_json, "r", encoding="utf-8") as tf:
                tdata = json.load(tf)
            template_name = tdata.get("name")
            tm = get_template_manager()
            tm.append_correction(template_name, item["text"], new_text)
            st.info("テンプレートを更新しました")
        except Exception:
            st.warning("テンプレートの更新に失敗しました")

# Extract documents that contain extract.json with needs_human

def load_review_items():
    items = []
    if not os.path.exists(WORKSPACE_DIR):
        return items
    for doc in sorted(os.listdir(WORKSPACE_DIR)):
        doc_path = os.path.join(WORKSPACE_DIR, doc)
        extract_path = os.path.join(doc_path, "extract.json")
        crops_dir = os.path.join(doc_path, "crops")
        if not os.path.isfile(extract_path):
            continue
        try:
            with open(extract_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for key, info in data.items():
            if info.get("needs_human"):
                item = {
                    "doc": doc,
                    "key": key,
                    "text": info.get("text", ""),
                    "source": info.get("source_image"),
                    "extract_path": extract_path,
                    "crops_dir": crops_dir,
                    "data": data,
                    "result_id": info.get("result_id"),
                }
                items.append(item)
    return items

items = load_review_items()

if not items:
    st.info("レビューが必要な項目はありません")
else:
    for idx, item in enumerate(items):
        st.subheader(f"{item['doc']} - {item['key']}")
        img_path = os.path.join(item["crops_dir"], item["source"]) if item["source"] else None
        if img_path and os.path.isfile(img_path):
            st.image(img_path)
        st.text(f"AI結果: {item['text']}")
        new_text = st.text_input("修正後のテキスト", value=item["text"], key=f"text_{idx}")
        add_dict = st.checkbox("辞書に登録", key=f"dict_{idx}")
        if st.button("修正を保存", key=f"save_{idx}"):
            save_correction(item, new_text, add_dict)
            st.success("保存しました")
