from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, math, json, os
from typing import List

# === 1. Create the FastAPI app ===
app = FastAPI(title="Monte Carlo Simulator", version="1.0.0")

# === 2. Enable CORS (must come AFTER app is defined) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # or restrict to ["https://www.jamesboggs.online"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 3. Load Model ===
MODEL_PATH = "models/montecarlo.pt"
META_PATH = "models/montecarlo_meta.json"

try:
    weights = torch.load(MODEL_PATH, map_location="cpu")
    mu = float(weights["mu"])
    sigma = float(weights["sigma"])
except Exception as e:
    raise RuntimeError(f"Failed to load model weights: {e}")

# === 4. Optional Metadata ===
meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)

# === 5. Pydantic Schemas ===
class SimRequest(BaseModel):
    S0: float = 100.0
    T: float = 1.0
    steps: int = 100
    sims: int = 1000

class SimResponse(BaseModel):
    paths: List[List[float]]

# === 6. Health Check ===
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": "MonteCarlo",
        "version": "1.0.0",
        "meta": meta or None
    }

# === 7. Simulation Endpoint ===
@app.post("/simulate", response_model=SimResponse)
def simulate(req: SimRequest):
    try:
        dt = req.T / req.steps
        Z = torch.randn(req.sims, req.steps)
        S = torch.zeros(req.sims, req.steps + 1)
        S[:, 0] = req.S0

        drift = (mu - 0.5 * sigma ** 2) * dt
        vol = sigma * math.sqrt(dt)

        for t in range(1, req.steps + 1):
            S[:, t] = S[:, t - 1] * torch.exp(drift + vol * Z[:, t - 1])

        return {"paths": S.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
