from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cases(path: str | Path = "evaluation_cases.json") -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_selected(
    selected: list[dict[str, Any]],
    case: dict[str, Any],
) -> dict[str, Any]:
    expected_source = str(case.get("expected_source") or "").lower()
    expected_type = str(case.get("expected_content_type") or "").lower()

    source_hit = any(
        expected_source
        and expected_source in str(
            item.get("source") or item.get("metadata", {}).get("source", "")
        ).lower()
        for item in selected
    )

    type_hit = any(
        expected_type
        and expected_type == str(
            item.get("content_type")
            or item.get("metadata", {}).get("content_type", "")
        ).lower()
        for item in selected
    )

    return {
        "source_hit": source_hit,
        "content_type_hit": type_hit,
        "hit": source_hit or type_hit,
    }
