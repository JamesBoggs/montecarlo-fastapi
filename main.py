from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import random

app = FastAPI()

# Allow CORS (for Vercel to call this safely)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/montecarlo")
async def montecarlo():
    trials = 5000
    growth_rate = torch.normal(mean=0.05, std=0.02, size=(trials,))
    churn_rate = torch.normal(mean=0.03, std=0.015, size=(trials,))
    base_arr = 100.0

    arr_outcomes = base_arr * (1 + growth_rate - churn_rate)
    mean_arr = torch.mean(arr_outcomes).item()
    std_arr = torch.std(arr_outcomes).item()

    return {
        "model": "montecarlo-v1.0.0",
        "status": "online",
        "latency": "88ms",
        "lastUpdated": "2025-10-11",
        "data": {
            "trials": trials,
            "meanARR": round(mean_arr, 2),
            "stdDev": round(std_arr, 2),
            "confidenceInterval": "±{:.1f}%".format((1.96 * std_arr / mean_arr) * 100)
        }
    }
    @app.post("/elasticity")
async def elasticity():
    import torch

    # Simulated price–demand curve
    prices = torch.linspace(10, 100, steps=20)
    demand = 500 * (prices ** -0.8) + torch.normal(0, 10, size=(20,))

    # Log transform for elasticity regression
    log_price = torch.log(prices).unsqueeze(1)
    log_demand = torch.log(demand).unsqueeze(1)

    # Linear regression: log(Demand) = β * log(Price) + intercept
    A = torch.cat([log_price, torch.ones_like(log_price)], dim=1)
    beta, _ = torch.lstsq(log_demand, A)
    beta = beta[:2]

    # Elasticity is slope
    elasticity = beta[0].item()
    intercept = beta[1].item()

    # R²
    predicted = A @ beta
    residuals = log_demand - predicted
    ss_res = torch.sum(residuals ** 2)
    ss_tot = torch.sum((log_demand - log_demand.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Fake optimal price based on intercept + elasticity
    optimal_price = torch.exp((torch.tensor(7.0) - intercept) / elasticity).item()

    return {
        "model": "elasticity-v1.2.3",
        "status": "online",
        "latency": "71ms",
        "lastUpdated": "2025-10-11",
        "data": {
            "elasticity": round(elasticity, 3),
            "rSquared": round(r2.item(), 4),
            "optimalPrice": round(optimal_price, 2)
        }
    }
@app.post("/price-engine")
def price_engine():
    import random
    tiers = [
        {"name": "Starter", "price": round(random.uniform(15, 25), 2), "targetMargin": f"{random.randint(55, 60)}%"},
        {"name": "Growth", "price": round(random.uniform(40, 60), 2), "targetMargin": f"{random.randint(60, 65)}%"},
        {"name": "Scale", "price": round(random.uniform(110, 140), 2), "targetMargin": f"{random.randint(65, 70)}%"},
    ]
    return {
        "model": "price-engine-v0.9.0",
        "status": "online",
        "lastUpdated": "2025-10-11",
        "latency": "63ms",
        "data": {
            "tiers": tiers,
            "recommended": "Use tiered pricing with anchoring between Growth and Scale"
        }
    }
}
