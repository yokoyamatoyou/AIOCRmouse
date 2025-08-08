from __future__ import annotations

"""Caching helpers for Streamlit apps.

This module provides cached constructors for commonly used resources such
as :class:`TemplateManager` and :class:`DBManager`.  Using Streamlit's caching
primitives avoids repeated disk access for templates and keeps a single
SQLite connection alive throughout a session.
"""

import streamlit as st

from core.template_manager import TemplateManager
from core.db_manager import DBManager


@st.cache_resource
def get_template_manager() -> TemplateManager:
    """Return a cached :class:`TemplateManager` instance."""
    return TemplateManager()


@st.cache_resource
def get_db_manager() -> DBManager:
    """Return an initialised and cached :class:`DBManager` instance."""
    db = DBManager()
    db.initialize()
    return db


@st.cache_data
def list_templates() -> list[str]:
    """Return available template names with data caching."""
    return get_template_manager().list_templates()
