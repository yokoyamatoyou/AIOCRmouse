import streamlit as st
import pandas as pd
from core.dashboard_utils import compute_metrics


@st.cache_data
def load_metrics():
    """Compute and cache dashboard metrics."""
    return compute_metrics("workspace")

st.title("パフォーマンス・ダッシュボード")

if 'metrics' not in st.session_state or st.button("更新"):
    st.session_state['metrics'] = load_metrics()

total_docs, total_fields, auto_rate, daily_df, template_df = st.session_state['metrics']

col1, col2, col3 = st.columns(3)
col1.metric("総処理ドキュメント数", total_docs)
col2.metric("総処理フィールド数", total_fields)
col3.metric("自動確定率", f"{auto_rate*100:.1f}%")

st.progress(auto_rate)

if not daily_df.empty:
    daily_df = daily_df.set_index('date')
    st.line_chart(daily_df)
else:
    st.info("データがありません")

st.subheader("テンプレート別ドキュメント数")
if not template_df.empty:
    st.bar_chart(template_df.set_index('template'))
else:
    st.info("テンプレートデータがありません")
