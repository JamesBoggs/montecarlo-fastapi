from fastapi.middleware.cors import CORSMiddleware
from ops_instrumentation import attach_ops
# main.py for montecarlo-fastapi
from fastapi import FastAPI
from meta import router as meta_router
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(meta_router)
attach_ops(app)

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

@app.get("/health")
def health():
    return {"ok": True}
