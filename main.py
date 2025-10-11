from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime
import numpy as np
import random

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "FastAPI backend running"}


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
        "latency": "71ms",
        "data": {
            "forecast_30d": days_30.round().tolist(),
            "forecast_90d": days_90.round().tolist(),
            "trend": "upward",
            "expectedARR": int(np.mean(days_90))
        }
    }


@app.get("/rl-pricing")
def rl_pricing():
    return {
        "model": "rl-pricing-v0.4.7",
        "status": "beta",
        "lastUpdated": str(datetime.date.today()),
        "latency": "84ms",
        "data": {
            "currentTier": "$29",
            "nextBestPrice": "$33",
            "expectedRevenueGain": "6.8%",
            "confidence": "moderate"
        }
    }


@app.get("/sentiment")
def sentiment():
    # Placeholder for real-time Twitter/Reddit API integration
    try:
        # EXAMPLE: Simulated score from external API
        sentiment_score = round(random.uniform(-1, 1), 2)
        price_delta = round(sentiment_score * 5, 2)

        return {
            "model": "sentiment-v0.3.1",
            "status": "online",
            "lastUpdated": str(datetime.date.today()),
            "latency": "91ms",
            "data": {
                "sentimentScore": sentiment_score,
                "expectedPriceChange": f"{price_delta}%",
                "source": "Twitter + Reddit"
            }
        }
    except Exception as e:
        return {"error": "Sentiment API call failed", "detail": str(e)}


@app.get("/volatility")
def volatility():
    days = 30
    vol_forecast = [round(0.15 + 0.02 * np.sin(i / 4) + np.random.normal(0, 0.01), 3) for i in range(days)]

    return {
        "model": "volatility-v1.1.0",
        "status": "beta",
        "lastUpdated": str(datetime.date.today()),
        "latency": "77ms",
        "data": {
            "volForecast": vol_forecast,
            "avgVol": round(np.mean(vol_forecast), 3),
            "modelUsed": "GARCH-LSTM Hybrid"
        }
    }


@app.get("/price-engine")
def price_engine():
    return {
        "model": "price-engine-v1.5.9",
        "status": "online",
        "lastUpdated": str(datetime.date.today()),
        "latency": "63ms",
        "data": {
            "tiers": [
                {"name": "Starter", "price": 19, "features": 5},
                {"name": "Pro", "price": 49, "features": 12},
                {"name": "Enterprise", "price": 99, "features": 30}
            ],
            "recommended": "Pro"
        }
    }
