from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept requests from anywhere for now
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/montecarlo")
def montecarlo():
    return {
        "model": "montecarlo-v1.0.0",
        "status": "online",
        "lastUpdated": str(datetime.date.today()),
        "latency": "72ms",
        "data": {
            "simulations": 1000,
            "meanARR": 122000,
            "p5": 97000,
            "p95": 148000
        }
    }

@app.get("/elasticity")
def elasticity():
    return {
        "model": "elasticity-v1.2.1",
        "status": "online",
        "lastUpdated": str(datetime.date.today()),
        "latency": "69ms",
        "data": {
            "elasticity": -0.94,
            "optimalPrice": 38.2,
            "revenuePeak": 124000
        }
    }

@app.get("/forecast")
def forecast():
    days_30 = np.linspace(120000, 126000, 30) + np.random.normal(0, 800, 30)
    days_90 = np.linspace(122000, 132000, 90) + np.random.normal(0, 1000, 90)

    return {
        "model": "forecast-v0.8.1",
        "status": "online",
        "lastUpdated": str(datetime.date.today()),
        "latency": "72ms",
        "data": {
            "forecast_30d": days_30.round().tolist(),
            "forecast_90d": days_90.round().tolist(),
            "trend": "upward",
            "expectedARR": int(np.mean(days_90))
        }
    }
