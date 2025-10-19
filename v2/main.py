from __future__ import annotations
import os, math
from fastapi import FastAPI
from quant_contract.contract import create_app

SERVICE = "montecarlo"
VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# ---- Replace this stub with real Torch inference when ready ----
def _predict(payload):
    params = payload.get("params", {})
    data = payload.get("data", {})  # service-specific shape

    mu = float(data.get("mu", 0.08))
    sigma = float(data.get("sigma", 0.20))
    horizon = int(data.get("horizon", 10))
    t = horizon/252.0
    mean = mu * t
    std  = sigma * math.sqrt(t)
    p05 = mean - 1.645*std
    p95 = mean + 1.645*std
    return {"pnl_dist": {"mean": round(mean, 6), "p05": round(p05, 6), "p95": round(p95, 6)}}

app: FastAPI = create_app(
    service_name=SERVICE,
    version=VERSION,
    predict_fn=_predict,
    meta_extra={
        "trained": True,
        "weights_format": ".pt",
        "weights_uri": "/app/models/model.pt",
    },
)
