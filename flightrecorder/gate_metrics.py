from __future__ import annotations

from typing import Any


def validation_metrics(validation_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(validation_summary, dict):
        return {
            "available": False,
            "passed": False,
            "strict": False,
            "target_count": 0,
            "error_count": 0,
            "warning_count": 0,
        }
    return {
        "available": True,
        "passed": bool(validation_summary.get("passed")),
        "strict": bool(validation_summary.get("strict")),
        "target_count": _int_value(validation_summary.get("target_count")),
        "error_count": _int_value(validation_summary.get("error_count")),
        "warning_count": _int_value(validation_summary.get("warning_count")),
    }


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0
