import json
import os
import re
from typing import Tuple

import pandas as pd


def compute_metrics(
    workspace_dir: str,
) -> Tuple[int, int, float, pd.DataFrame, pd.DataFrame]:
    """Compute dashboard metrics from all extract.json files."""
    total_docs = 0
    total_fields = 0
    auto_confirmed = 0
    daily_counts: dict[str, int] = {}
    template_counts: dict[str, int] = {}

    if not os.path.exists(workspace_dir):
        return (
            0,
            0,
            0.0,
            pd.DataFrame(columns=["date", "count"]),
            pd.DataFrame(columns=["template", "count"]),
        )

    for doc in os.listdir(workspace_dir):
        extract_path = os.path.join(workspace_dir, doc, "extract.json")
        if not os.path.isfile(extract_path):
            continue
        total_docs += 1
        with open(extract_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_fields += len(data)
        for info in data.values():
            if not info.get("needs_human"):
                auto_confirmed += 1

        template_path = os.path.join(workspace_dir, doc, "template.json")
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as tf:
                tdata = json.load(tf)
            tname = tdata.get("name", "unknown")
        else:
            tname = "unknown"
        template_counts[tname] = template_counts.get(tname, 0) + 1

        m = re.match(r"DOC_(\d{8})", doc)
        if m:
            day = m.group(1)
            daily_counts[day] = daily_counts.get(day, 0) + 1

    auto_rate = auto_confirmed / total_fields if total_fields else 0.0
    daily_df = pd.DataFrame(
        {
            "date": sorted(daily_counts),
            "count": [daily_counts[d] for d in sorted(daily_counts)],
        }
    )
    template_df = pd.DataFrame(
        {
            "template": sorted(template_counts),
            "count": [template_counts[t] for t in sorted(template_counts)],
        }
    )
    return total_docs, total_fields, auto_rate, daily_df, template_df
