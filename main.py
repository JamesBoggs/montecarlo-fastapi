# main.py — Forecast FastAPI (FRED CPI-based GRU Forecast)
# ========================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
import pandas as pd
from fredapi import Fred
import os, datetime, math

# === Setup ===
app = FastAPI(title="Forecast GRU", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model + FRED Setup ===
MODEL_PATH = "models/forecast.pt"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
fred = Fred(api_key= 0f4e6f6b0a0158deaba656681dcbb4c6)

# Load the GRU model
try:
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
except Exception as e:
    raise RuntimeError(f"[ERROR] Could not load forecast.pt: {e}")

# --- Utility ---
def fetch_fred_series(series_id: str, start="2010-01-01") -> pd.Series:
    """Fetch a FRED time series and return as a pandas Series."""
    data = fred.get_series(series_id)
    df = data[data.index >= start]
    return df

def preprocess(series: pd.Series, seq_len: int = 12):
    """Convert series into rolling sequences for GRU input."""
    arr = (series - series.mean()) / (series.std() + 1e-8)
    X = []
    for i in range(len(arr) - seq_len):
        X.append(arr.iloc[i:i+seq_len].values)
    X = np.array(X, dtype=np.float32)
    return torch.tensor(X[-1:]).unsqueeze(-1)  # last window only

def predict_forecast(series_id: str, seq_len: int = 12, steps_ahead: int = 3):
    """Run forecast for given FRED series."""
    series = fetch_fred_series(series_id)
    x = preprocess(series, seq_len)
    with torch.no_grad():
        y_pred = model(x).squeeze().numpy()
    return float(y_pred[-1]) if isinstance(y_pred, np.ndarray) else float(y_pred)

def calc_mom(series_id: str):
    """Compute month-over-month inflation."""
    series = fetch_fred_series(series_id)
    pct = series.pct_change().dropna() * 100
    return pct.iloc[-1]

def calc_yoy(series_id: str):
    """Compute year-over-year inflation."""
    series = fetch_fred_series(series_id)
    yoy = ((series.iloc[-1] / series.iloc[-13]) - 1) * 100
    return yoy

# === Routes ===

@app.get("/")
def root():
    return {"status": "ok", "message": "Forecast GRU API is running."}

@app.get("/meta")
def meta():
    return {
        "name": "Forecast GRU",
        "description": "Forecasts headline CPI, core CPI, and inflation rates using FRED data + PyTorch GRU.",
        "isOnline": True,
        "version": "2.0.0"
    }

@app.get("/forecast")
def run_forecast(
    mode: str = Query("headline", enum=["headline", "core", "mom", "yoy"]),
):
    """
    mode options:
      - headline: CPIAUCSL
      - core: CPILFESL
      - mom: month-over-month % change
      - yoy: year-over-year % change
    """
    try:
        if mode == "headline":
            value = predict_forecast("CPIAUCSL")
            label = "Headline CPI Forecast"
        elif mode == "core":
            value = predict_forecast("CPILFESL")
            label = "Core CPI Forecast"
        elif mode == "mom":
            value = calc_mom("CPIAUCSL")
            label = "MoM Inflation Rate"
        elif mode == "yoy":
            value = calc_yoy("CPIAUCSL")
            label = "YoY Inflation Rate"
        else:
            raise ValueError("Invalid mode.")
        
        return {
            "mode": mode,
            "label": label,
            "value": round(value, 3),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Example run ===
# https://forecast-fastapi.onrender.com/forecast?mode=headline
# https://forecast-fastapi.onrender.com/forecast?mode=core
# https://forecast-fastapi.onrender.com/forecast?mode=mom
# https://forecast-fastapi.onrender.com/forecast?mode=yoy
