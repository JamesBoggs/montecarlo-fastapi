from fastapi import FastAPI
from datetime import datetime
import random

app = FastAPI()

@app.get("/montecarlo")
async def montecarlo():
    revenue = [round(random.uniform(10000, 15000), 2) for _ in range(12)]
    return {
        "model": "montecarlo-v1.0.1",
        "status": "online",
        "lastUpdated": str(datetime.utcnow().date()),
        "data": {
            "simulatedRevenue": revenue,
            "mean": round(sum(revenue) / len(revenue), 2)
        }
    }
