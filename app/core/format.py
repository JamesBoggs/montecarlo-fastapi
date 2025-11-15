# app/core/format.py
from typing import Any, Dict


def metric_payload(
    id: str,
    label: str,
    series: Any,
    status: str = "online",
) -> Dict[str, Any]:
    """
    Normalize metric output so the dashboard still sees the same keys
    it used to (metric + value), but we also add id/label/status for
    nicer UI use.

    - If the metric fn returns a dict like {"metric": "var", "value": 0.12},
      we just spread that into the top-level and add id/label/status.
    - If it returns something else (list, number), we tuck it under "series".
    """
    if isinstance(series, dict):
        base: Dict[str, Any] = dict(series)
    else:
        base = {"series": series}

    # Ensure there's a metric name; default to id if not present
    base.setdefault("metric", id)

    # Add the standardized bits for the UI
    base["id"] = id
    base["label"] = label
    base["status"] = status

    return base
