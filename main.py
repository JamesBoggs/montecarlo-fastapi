from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import random

app = FastAPI()

# Allow frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class ElasticityRequest(BaseModel):
    price: float
    base_demand: float

@app.get("/")
def read_root():
    return {"message": "FastAPI pricing models backend"}

# --------------------
# MONTECARLO ENDPOINT
# --------------------
@app.post("/montecarlo")
async def montecarlo():
    data = [{"month": i, "revenue": random.randint(10000, 50000)} for i in range(1, 13)]
    return {
        "model": "montecarlo-v1.0.0",
        "status": "online",
        "lastUpdated": "2025-10-11",
        "results": data
    }

# --------------------
# ELASTICITY ENDPOINT
# --------------------
@app.post("/elasticity")
async def elasticity(req: ElasticityRequest):
    try:
        # Dummy PyTorch calculation (linear demand drop)
        price_tensor = torch.tensor(req.price)
        base_demand_tensor = torch.tensor(req.base_demand)

        elasticity_coeff = torch.tensor(-0.3)  # Sensitivity to price
        demand = base_demand_tensor + elasticity_coeff * (price_tensor - 100)

        return {
            "model": "elasticity-v1.0.0",
            "status": "online",
            "lastUpdated": "2025-10-11",
            "input": {"price": req.price, "base_demand": req.base_demand},
            "output": {"demand": float(torch.clamp(demand, min=0).item())}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# STUBS FOR FUTURE MODELS
# --------------------
@app.post("/forecast")
async def forecast():
    raise HTTPException(status_code=501, detail="Coming soon")

@app.post("/price-engine")
async def price_engine():
    raise HTTPException(status_code=501, detail="Coming soon")

@app.post("/rl-pricing")
async def rl_pricing():
    raise HTTPException(status_code=501, detail="Coming soon")

@app.post("/sentiment")
async def sentiment():
    raise HTTPException(status_code=501, detail="Coming soon")

@app.post("/volatility")
async def volatility():
    raise HTTPException(status_code=501, detail="Coming soon")
