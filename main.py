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
