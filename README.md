# montecarlo-fastapi

FastAPI service for simulating future price paths using Geometric Brownian Motion (GBM) with learned parameters (mu, sigma) from AAPL log returns.

## 🔁 Endpoint

### POST `/simulate`
Simulates GBM price paths.

**Input:**
```json
{
  "S0": 100.0,
  "T": 1.0,
  "steps": 100,
  "sims": 1000
}
