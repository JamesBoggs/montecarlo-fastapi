from fastapi import FastAPI
from datetime import datetime
import torch

app = FastAPI()

@app.get("/montecarlo")
async def run_montecarlo(start_price: float = 100.0, mu: float = 0.07, sigma: float = 0.15, T: float = 1.0, steps: int = 252, simulations: int = 10000):
    dt = T / steps
    torch.manual_seed(42)

    # Generate random returns
    rand_normals = torch.randn(simulations, steps)
    price_paths = torch.zeros(simulations, steps + 1)
    price_paths[:, 0] = start_price

    for t in range(1, steps + 1):
        price_paths[:, t] = price_paths[:, t-1] * torch.exp((mu - 0.5 * sigma**2) * dt + sigma * torch.sqrt(torch.tensor(dt)) * rand_normals[:, t-1])

    final_prices = price_paths[:, -1]
    expected_return = final_prices.mean().item()
    std_dev = final_prices.std().item()

    return {
        "model": "montecarlo-v1.0.1",
        "status": "online",
        "lastUpdated": str(datetime.utcnow().date()),
        "data": {
            "start_price": start_price,
            "expected_return": expected_return,
            "std_dev": std_dev,
            "simulated_path": price_paths[0].tolist()[:50]  # shorten for payload
        }
    }
