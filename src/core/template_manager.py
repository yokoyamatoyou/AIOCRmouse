from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import re
from difflib import SequenceMatcher


class TemplateManager:
    """Manage template files stored in JSON format.

    Each template describes regions of interest (ROIs) and optional
    prompt rules or correction dictionaries for post processing.
    """

    def __init__(self, template_dir: str = "templates") -> None:
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

    def list_templates(self) -> List[str]:
        """Return a list of available template names."""
        return [p.stem for p in self.template_dir.glob("*.json")]

    def _normalise(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return template data with a consistent structure."""
        keywords = data.get("keywords", [])
        data["keywords"] = keywords if isinstance(keywords, list) else []

        corrections = data.get("corrections", [])
        if isinstance(corrections, dict):
            corrections = [{"wrong": k, "correct": v} for k, v in corrections.items()]
        elif not isinstance(corrections, list):
            corrections = []
        data["corrections"] = corrections
        return data

    def load(self, name: str) -> Dict[str, Any]:
        """Load a template by name and normalise its structure.

        Parameters
        ----------
        name: str
            Template file name without extension.
        """
        path = self.template_dir / f"{name}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return self._normalise(data)

    def save(self, name: str, data: Dict[str, Any]) -> None:
        """Save template data to a JSON file.

        Any additional fields are preserved to allow forward compatible
        extensions such as ``template_image_path``.  The ``keywords`` field is
        normalised to always be present as a list to simplify downstream
        consumption.
        """
        path = self.template_dir / f"{name}.json"
        data_to_save = self._normalise(dict(data))
        with path.open("w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    def get_keywords(self, name: str) -> List[str]:
        """Return the list of detection keywords for a template.

        If the template does not define any keywords an empty list is
        returned."""
        data = self.load(name)
        keywords = data.get("keywords", [])
        return keywords if isinstance(keywords, list) else []


    def detect_template(self, text: str) -> tuple[str, Dict[str, Any]] | None:
        """Select the best template for ``text`` based on keyword matches.

        Parameters
        ----------
        text:
            OCR'd text used to determine which template is most appropriate.

        Returns
        -------
        tuple or ``None``
            A tuple of ``(template_name, template_data)`` for the best match.
            ``None`` is returned when no template contains any of the
            configured keywords.
        """
        text_lower = text.lower()
        words = re.findall(r"[\w一-龯ぁ-んァ-ンー]+", text_lower)
        best_score = 0.0
        best_name: str | None = None
        best_data: Dict[str, Any] | None = None
        for name in self.list_templates():
            data = self.load(name)
            keywords = data.get("keywords", [])
            score = 0.0
            if isinstance(keywords, list):
                for kw in keywords:
                    try:
                        if re.search(kw, text, flags=re.IGNORECASE):
                            score += 1
                            continue
                    except re.error:
                        pass
                    kw_lower = kw.lower()
                    if kw_lower in text_lower:
                        score += 1
                        continue
                    max_ratio = max(
                        (
                            SequenceMatcher(None, kw_lower, word).ratio()
                            for word in words
                        ),
                        default=0.0,
                    )
                    if max_ratio >= 0.8:
                        score += max_ratio
            if score > best_score:
                best_score = score
                best_name = name
                best_data = data
        if best_name is None or best_score == 0:
            return None
        return best_name, best_data

    def append_correction(self, name: str, wrong: str, correct: str) -> None:
        """Append a correction pair to template's correction list.

        The previous implementation stored corrections in a mapping which
        overwrote existing entries when the same ``wrong`` text appeared
        multiple times.  To preserve the full history of human feedback,
        corrections are now recorded as a list of ``{"wrong": ..., "correct": ...}``
        dictionaries. Duplicate pairs are ignored.
        """
        data = self.load(name)
        corrections = data.setdefault("corrections", [])
        if not any(
            c.get("wrong") == wrong and c.get("correct") == correct
            for c in corrections
        ):
            corrections.append({"wrong": wrong, "correct": correct})
            self.save(name, data)
