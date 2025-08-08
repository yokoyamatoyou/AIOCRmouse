"""Streamlit entry point for the AIOCR application.

This module sets up simple navigation and redirects to the main OCR page.
"""

import streamlit as st


def main() -> None:
    """Redirect to the main OCR page while exposing navigation links."""
    st.sidebar.page_link("pages/0_Template_Editor.py", label="Template Editor")
    st.sidebar.page_link("pages/1_Main_OCR.py", label="Main OCR")
    st.sidebar.page_link("pages/2_Review.py", label="Review & 修正")
    st.sidebar.page_link("pages/3_Dashboard.py", label="ダッシュボード")
    st.sidebar.page_link("pages/4_Admin.py", label="管理設定")

    st.switch_page("pages/1_Main_OCR.py")


if __name__ == "__main__":
    main()

