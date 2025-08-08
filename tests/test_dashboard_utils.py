import os
import json
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core.dashboard_utils import compute_metrics


def test_compute_metrics(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    doc1 = workspace / "DOC_20250101_000000"
    doc1.mkdir()
    with open(doc1 / "extract.json", "w", encoding="utf-8") as f:
        json.dump({"a": {"text": "x"}, "b": {"text": "y", "needs_human": True}}, f, ensure_ascii=False)
    with open(doc1 / "template.json", "w", encoding="utf-8") as f:
        json.dump({"name": "invoice"}, f, ensure_ascii=False)

    doc2 = workspace / "DOC_20250102_000000"
    doc2.mkdir()
    with open(doc2 / "extract.json", "w", encoding="utf-8") as f:
        json.dump({"c": {"text": "z"}}, f, ensure_ascii=False)
    with open(doc2 / "template.json", "w", encoding="utf-8") as f:
        json.dump({"name": "receipt"}, f, ensure_ascii=False)

    total_docs, total_fields, auto_rate, daily_df, template_df = compute_metrics(str(workspace))

    assert total_docs == 2
    assert total_fields == 3
    assert abs(auto_rate - 2/3) < 1e-6
    assert isinstance(daily_df, pd.DataFrame)
    assert list(daily_df["date"]) == ["20250101", "20250102"]
    assert list(daily_df["count"]) == [1, 1]
    assert list(template_df.sort_values("template")["template"]) == ["invoice", "receipt"]
    assert list(template_df.sort_values("template")["count"]) == [1, 1]
