import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def template_editor():
    """Dynamically load the template editor page module.

    The page filename begins with a number so it cannot be imported using the
    normal ``import`` statement.  This fixture loads it via ``importlib`` and
    provides minimal stubs for Streamlit dependencies so the module can be
    imported in a test environment.
    """

    def identity_decorator(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

    # Provide very small stand-ins for ``streamlit`` and
    # ``streamlit_drawable_canvas`` so importing the page succeeds.
    st_module = types.ModuleType("streamlit")
    st_module.cache_resource = identity_decorator
    st_module.cache_data = identity_decorator
    sys.modules.setdefault("streamlit", st_module)
    canvas_module = types.ModuleType("streamlit_drawable_canvas")
    canvas_module.st_canvas = lambda *a, **k: None
    sys.modules.setdefault("streamlit_drawable_canvas", canvas_module)

    # Ensure application modules can be imported regardless of current working directory
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((root / "src").resolve()))

    path = root / "src" / "app" / "pages" / "0_Template_Editor.py"
    spec = importlib.util.spec_from_file_location("template_editor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


def test_load_initial_drawing(template_editor):
    rois = {
        "field1": {"box": [1, 2, 3, 4]},
        "field2": {"box": [5, 6, 7, 8]},
    }
    drawing = template_editor._load_initial_drawing(rois)
    assert drawing["version"] == "5.0.0"
    assert len(drawing["objects"]) == 2
    first = drawing["objects"][0]
    assert first["left"] == 1
    assert first["top"] == 2
    assert first["width"] == 3
    assert first["height"] == 4
