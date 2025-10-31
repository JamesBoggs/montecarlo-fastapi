from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, os, json, math, random

# === FastAPI Setup ===
app = FastAPI(title="Monte Carlo GBM Simulator", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # frontend dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model Paths ===
MODEL_PATH = "models/montecarlo.pt"
META_PATH = "models/montecarlo_meta.json"

# === Load Model Weights ===
try:
    weights = torch.load(MODEL_PATH, map_location="cpu")
    mu = float(weights.get("mu", 0.05))
    sigma = float(weights.get("sigma", 0.2))
except Exception as e:
    mu, sigma = 0.05, 0.2
    print(f"[WARN] Failed to load model weights: {e}")

# === Optional Metadata ===
meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)

# === Health Check ===
@app.get("/ping")
async def ping():
    """Simple health check for dashboard"""
    return {"status": "online"}

# === Meta Info ===
@app.get("/meta")
async def meta_info():
    """Return model details, parameters, and endpoints"""
    return {
        "status": "online",
        "model": "Monte Carlo GBM Simulator",
        "version": "2.2.0",
        "params": {"mu": mu, "sigma": sigma},
        "framework": "FastAPI + PyTorch",
        "device": "CPU",
        "categories": [
            {"Category": "Value-at-Risk (VaR)", "What It Measures": "Downside risk at 95–99% confidence", "Endpoint": "/montecarlo/var"},
            {"Category": "Conditional VaR (CVaR)", "What It Measures": "Average loss beyond VaR cutoff", "Endpoint": "/montecarlo/cvar"},
            {"Category": "Sharpe / Sortino Ratio", "What It Measures": "Risk-adjusted return metrics", "Endpoint": "/montecarlo/sharpe"},
            {"Category": "Stress Testing", "What It Measures": "Effect of volatility or drift shocks", "Endpoint": "/montecarlo/stress"},
            {"Category": "Option Pricing", "What It Measures": "Simulate call/put payoffs under GBM", "Endpoint": "/montecarlo/options"},
            {"Category": "Portfolio Path Simulation", "What It Measures": "Multi-asset correlation + rebalancing", "Endpoint": "/montecarlo/portfolio"},
            {"Category": "Forecast Error Bands", "What It Measures": "Wrap forecasts in stochastic variance", "Endpoint": "/montecarlo/uncertainty"},
            {"Category": "Drawdown Analysis", "What It Measures": "Worst cumulative loss trajectory", "Endpoint": "/montecarlo/drawdown"},
            {"Category": "Correlation Breakdown", "What It Measures": "Monte Carlo correlation shifts over time", "Endpoint": "/montecarlo/corrshift"},
            {"Category": "Tail Risk / Skewness", "What It Measures": "Higher-moment statistics of returns", "Endpoint": "/montecarlo/tailrisk"},
        ],
    }

# === Example Simulation Endpoint ===
class SimRequest(BaseModel):
    S0: float = 100.0
    T: float = 1.0
    steps: int = 252
    paths: int = 1000

@app.post("/montecarlo/var")
async def montecarlo_var(req: SimRequest):
    """Simple VaR Monte Carlo simulation"""
    try:
        dt = req.T / req.steps
        shocks = torch.randn(req.paths, req.steps) * sigma * math.sqrt(dt)
        drift = (mu - 0.5 * sigma ** 2) * dt
        paths = req.S0 * torch.exp(torch.cumsum(drift + shocks, dim=1))
        final_prices = paths[:, -1]
        var_95 = float(torch.quantile(final_prices, 0.05))
        var_99 = float(torch.quantile(final_prices, 0.01))
        return {
            "VaR_95": round(req.S0 - var_95, 3),
            "VaR_99": round(req.S0 - var_99, 3),
            "params": {"mu": mu, "sigma": sigma},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Root Welcome ===
@app.get("/")
async def root():
    return {"message": "Monte Carlo API online."}
