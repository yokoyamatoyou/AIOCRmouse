import streamlit as st

from app.cache_utils import get_db_manager
from core.config import settings


def main() -> None:
    st.set_page_config(page_title="管理設定")
    st.title("管理設定")

    st.caption("モデルやスコア関連の閾値を運用中に調整できます（セッション内設定）。")

    with st.expander("スコア重みと閾値"):
        w_ocr = st.slider("OCR自信度の重み", 0.0, 1.0, float(settings.SCORE_WEIGHT_OCR), 0.01)
        w_rule = st.slider("ルール適合の重み", 0.0, 1.0, float(settings.SCORE_WEIGHT_RULE), 0.01)
        w_agree = st.slider("モデル一致度の重み", 0.0, 1.0, float(settings.SCORE_WEIGHT_AGREEMENT), 0.01)
        threshold = st.slider("複合スコアの確定閾値", 0.0, 1.0, float(settings.COMPOSITE_DECISION_THRESHOLD), 0.01)

        if st.button("変更を適用"):
            # セッション内に保存（即時反映）。プロセス全体の恒久化は.env編集が必要。
            st.session_state["SCORE_WEIGHT_OCR"] = w_ocr
            st.session_state["SCORE_WEIGHT_RULE"] = w_rule
            st.session_state["SCORE_WEIGHT_AGREEMENT"] = w_agree
            st.session_state["COMPOSITE_DECISION_THRESHOLD"] = threshold
            st.success("変更を適用しました（セッション内）")

    with st.expander("データベースメンテナンス"):
        db = get_db_manager()
        st.write("データベースパス: ", db.db_path)
        if st.button("接続を再初期化"):
            db.close()
            st.info("次回の操作で再接続されます。")


if __name__ == "__main__":
    main()



