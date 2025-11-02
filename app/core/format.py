from typing import Any, Dict, List, Iterable


def to_series(data: Any) -> List[Dict[str, float]]:
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            data = data.tolist()
    except Exception:
        pass

    if data is None:
        return []

    if isinstance(data, list) and (not data or isinstance(data[0], (int, float))):
        return [{"x": i, "y": float(y)} for i, y in enumerate(data)]

    if isinstance(data, list) and data and isinstance(data[0], dict):
        kx = next((k for k in data[0] if k.lower() in ("x","t","time","index","idx")), "x")
        ky = next((k for k in data[0] if k.lower() in ("y","v","value","val","price")), "y")
        out = []
        for i, d in enumerate(data):
            x = d.get(kx, i)
            y = d.get(ky, 0.0)
            out.append({"x": (float(x) if isinstance(x,(int,float)) else i), "y": float(y)})
        return out
    return []


def metric_payload(id: str, label: str, series: Any, status: str = "online") -> Dict[str, Any]:
    return {"id": id, "label": label, "status": status, "chartData": to_series(series)}


def meta_payload(name: str, description: str, metrics: Iterable[Dict[str, str]], version: str = "1.0.0") -> Dict[str, Any]:
    return {"name": name, "description": description, "version": version, "metrics": list(metrics)}
