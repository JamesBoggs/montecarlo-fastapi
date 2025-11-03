# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, meta, metrics

app = FastAPI(title="Monte Carlo API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire routers (these import your registry and calcs)
app.include_router(health.router)
app.include_router(meta.router)
app.include_router(metrics.router)

# Debug: list every mounted route (handy on Render)
@app.get("/__routes")
def __routes():
    return [{"path": r.path, "methods": list(getattr(r, "methods", []))} for r in app.router.routes]
