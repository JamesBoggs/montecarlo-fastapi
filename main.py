from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, math, os, json, random
import numpy as np

# === FastAPI Setup ===
app = FastAPI(title="Monte Carlo Simulator", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model Parameters ===
MODEL_PATH = "models/montecarlo.pt"
META_PATH = "models/montecarlo_meta.json"

try:
    weights = torch.load(MODEL_PATH, map_location="cpu")
    mu = float(weights.get("mu", 0.05))
    sigma = float(weights.get("sigma", 0.2))
except Exception as e:
    mu, sigma = 0.05, 0.2
    print(f"[WARN] Failed to load model weights: {e}")

# Optional metadata
meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)

# === Request Schema ===
class SimRequest(BaseModel):
    S0: float = 100.0
    T: float = 1.0
    steps: int = 252
    paths: int = 1000

# === Core Simulation ===
def simulate_paths(S0, T, steps, paths, mu, sigma):
    dt = T / steps
    results = np.zeros((paths, steps + 1))
    for i in range(paths):
        prices = [S0]
        for _ in range(steps):
            dW = np.random.normal(0, math.sqrt(dt))
            S_next = prices[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            prices.append(S_next)
        results[i] = prices
    return results

# === Base Routes ===
@app.get("/ping")
def ping():
    return {"status": "online"}

@app.get("/meta")
def meta_info():
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
        ]
    }

@app.post("/simulate")
def simulate(req: SimRequest):
    try:
        results = simulate_paths(req.S0, req.T, req.steps, req.paths, mu, sigma)
        metrics = {
            "Drift μ": mu,
            "Volatility σ": sigma,
            "Sharpe Ratio": round(mu / sigma, 3),
            "Simulated Mean": round(results.mean(), 2),
            "Simulated Std": round(results.std(), 2),
            "Return (Δ%)": round(((results[:, -1].mean() - req.S0) / req.S0) * 100, 2),
            "95% CI": round(req.S0 * math.exp((mu - 0.5 * sigma**2) * req.T), 2),
            "Skewness": round(float(((results[:, -1] - results.mean())**3).mean() / results.std()**3), 3),
            "Kurtosis": round(float(((results[:, -1] - results.mean())**4).mean() / results.std()**4), 3),
            "Paths Simulated": req.paths,
        }
        return {"paths": results.tolist(), "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

# === 10 Metric Endpoints ===

# 1. Value-at-Risk (VaR)
@app.get("/montecarlo/var")
def value_at_risk(confidence: float = 0.95, S0: float = 100):
    results = simulate_paths(S0, 1.0, 252, 2000, mu, sigma)
    returns = (results[:, -1] - S0) / S0
    var = np.percentile(returns, (1 - confidence) * 100)
    return {"Category": "Value-at-Risk (VaR)", "VaR": round(var, 4), "Confidence": confidence}

# 2. Conditional VaR (CVaR)
@app.get("/montecarlo/cvar")
def conditional_var(confidence: float = 0.95, S0: float = 100):
    results = simulate_paths(S0, 1.0, 252, 2000, mu, sigma)
    returns = (results[:, -1] - S0) / S0
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return {"Category": "Conditional VaR (CVaR)", "CVaR": round(cvar, 4), "Confidence": confidence}

# 3. Sharpe / Sortino Ratio
@app.get("/montecarlo/sharpe")
def sharpe_ratio(S0: float = 100):
    results = simulate_paths(S0, 1.0, 252, 2000, mu, sigma)
    rets = (results[:, -1] - S0) / S0
    sharpe = rets.mean() / rets.std()
    downside = np.std(rets[rets < 0])
    sortino = rets.mean() / downside if downside > 0 else None
    return {"Category": "Sharpe / Sortino Ratio", "Sharpe": round(float(sharpe), 4), "Sortino": round(float(sortino), 4) if sortino else None}

# 4. Stress Testing
@app.get("/montecarlo/stress")
def stress_test(S0: float = 100, vol_multiplier: float = 2.0):
    stressed = simulate_paths(S0, 1.0, 252, 1000, mu, sigma * vol_multiplier)
    mean_return = ((stressed[:, -1] - S0) / S0).mean()
    return {"Category": "Stress Testing", "Volatility Multiplier": vol_multiplier, "Mean Return": round(float(mean_return), 4)}

# 5. Option Pricing
@app.get("/montecarlo/options")
def option_pricing(S0: float = 100, K: float = 100, r: float = 0.03, T: float = 1.0):
    results = simulate_paths(S0, T, 252, 5000, mu, sigma)
    payoff = np.maximum(results[:, -1] - K, 0)
    price = np.exp(-r * T) * payoff.mean()
    return {"Category": "Option Pricing", "Strike": K, "Price": round(float(price), 4)}

# 6. Portfolio Path Simulation
@app.get("/montecarlo/portfolio")
def portfolio_sim(weights: str = "0.5,0.5", S0: float = 100):
    w = np.array([float(x) for x in weights.split(",")])
    assets = [simulate_paths(S0, 1.0, 252, 2000, mu, sigma * (i + 1)) for i in range(len(w))]
    returns = np.vstack([(a[:, -1] - S0) / S0 for a in assets])
    portfolio_ret = (w @ returns).mean()
    portfolio_std = math.sqrt(w @ np.cov(returns) @ w.T)
    return {"Category": "Portfolio Path Simulation", "Mean Return": round(portfolio_ret, 4), "Std Dev": round(portfolio_std, 4)}

# 7. Forecast Error Bands
@app.get("/montecarlo/uncertainty")
def forecast_uncertainty(S0: float = 100):
    results = simulate_paths(S0, 1.0, 252, 2000, mu, sigma)
    final = results[:, -1]
    ci_low, ci_high = np.percentile(final, [5, 95])
    return {"Category": "Forecast Error Bands", "95% CI": [round(ci_low, 2), round(ci_high, 2)]}

# 8. Drawdown Analysis
@app.get("/montecarlo/drawdown")
def drawdown(S0: float = 100):
    results = simulate_paths(S0, 1.0, 252, 2000, mu, sigma)
    dd = (results.max(axis=1) - results[:, -1]) / results.max(axis=1)
    return {"Category": "Drawdown Analysis", "Mean Drawdown": round(dd.mean(), 4)}

# 9. Correlation Breakdown
@app.get("/montecarlo/corrshift")
def correlation_shift(S0: float = 100):
    a = simulate_paths(S0, 1.0, 252, 1000, mu, sigma)
    b = simulate_paths(S0, 1.0, 252, 1000, mu * 0.8, sigma * 1.2)
    corr = np.corrcoef(a[:, -1], b[:, -1])[0, 1]
    return {"Category": "Correlation Breakdown", "Correlation": round(float(corr), 4)}

# 10. Tail Risk / Skewness
@app.get("/montecarlo/tailrisk")
def tail_risk(S0: float = 100):
    results = simulate_paths(S0, 1.0, 252, 2000, mu, sigma)
    returns = (results[:, -1] - S0) / S0
    skew = ((returns - returns.mean())**3).mean() / returns.std()**3
    kurt = ((returns - returns.mean())**4).mean() / returns.std()**4
    return {"Category": "Tail Risk / Skewness", "Skewness": round(float(skew), 4), "Kurtosis": round(float(kurt), 4)}

# === Root Summary ===
@app.get("/")
def root():
    return {
        "message": "Monte Carlo Simulator running",
        "version": "2.2.0",
        "routes": [
            "/ping", "/meta", "/simulate",
            "/montecarlo/var", "/montecarlo/cvar", "/montecarlo/sharpe",
            "/montecarlo/stress", "/montecarlo/options", "/montecarlo/portfolio",
            "/montecarlo/uncertainty", "/montecarlo/drawdown",
            "/montecarlo/corrshift", "/montecarlo/tailrisk"
        ],
        "params": {"mu": mu, "sigma": sigma}
    }
