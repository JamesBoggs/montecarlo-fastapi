# app/core/format.py
from typing import Any, Dict, List


def meta_payload(
    name: str,
    description: str,
    version: str,
    engine: str,
    device: str,
    metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Standard shape for /meta responses so the frontend always sees
    the same keys across services.
    """
    return {
        "name": name,
        "description": description,
        "version": version,
        "engine": engine,
        "device": device,
        "metrics": metrics,
    }


def metric_payload(
    id: str,
    label: str,
    series: Any,
    status: str = "online",
) -> Dict[str, Any]:
    """
    Normalize metric output so the dashboard still sees 'metric' and 'value'
    like before, but we also add id/label/status for nicer UI use.

    - If the metric fn returns a dict like {"metric": "var", "value": 0.12},
      we spread that into the top-level and add id/label/status.
    - Otherwise we tuck it under "series".
    """
    if isinstance(series, dict):
        base: Dict[str, Any] = dict(series)
    else:
        base = {"series": series}

    # Ensure there's a metric name; default to id if not present
    base.setdefault("metric", id)

    # Add standardized bits for the UI
    base["id"] = id
    base["label"] = label
    base["status"] = status

    return base
