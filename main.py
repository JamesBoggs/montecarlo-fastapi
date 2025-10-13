# main.py for montecarlo-fastapi
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/montecarlo")
async def run_monte_carlo():
    # Simulated stats (replace with real Monte Carlo later)
    return {
        "model": "montecarlo-v1.0.0",
        "status": "online",
        "lastUpdated": str(datetime.utcnow().date()),
        "data": {
            "mean": 112.45,
            "std_dev": 7.23,
            "confidence_interval": [99.34, 125.56]
        }
    }
