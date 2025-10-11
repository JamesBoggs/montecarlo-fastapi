from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import datetime

app = FastAPI()

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "model": "montecarlo-v1.0.0"}

@app.get("/montecarlo")
def montecarlo_simulation(
    trials: int = 500,
    initial_revenue: float = 1_000_000,
    mean_growth: float = 0.02,
    std_dev: float = 0.05,
    months: int = 12,
):
    np.random.seed(42)
    simulations = []

    for _ in range(trials):
        revenue = initial_revenue
        path = [revenue]
        for _ in range(months):
            shock = np.random.normal(loc=mean_growth, scale=std_dev)
            revenue *= (1 + shock)
            path.append(revenue)
        simulations.append(path)

    percentiles = np.percentile(simulations, [5, 50, 95], axis=0)

    return {
        "model": "montecarlo-v1.0.0",
        "last_updated": datetime.datetime.utcnow().isoformat(),
        "revenue_projection": {
            "p5": list(np.round(percentiles[0], 2)),
            "p50": list(np.round(percentiles[1], 2)),
            "p95": list(np.round(percentiles[2], 2)),
        },
        "params": {
            "trials": trials,
            "months": months,
            "initial_revenue": initial_revenue,
            "mean_growth": mean_growth,
            "std_dev": std_dev,
        }
    }
