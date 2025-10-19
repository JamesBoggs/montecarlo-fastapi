from __future__ import annotations
import os, time, hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

class Health(BaseModel):
    ok: bool = True

class Meta(BaseModel):
    model: str
    model_version: str
    git_sha: str
    last_updated: str
    trained: bool
    weights_format: str
    weights_uri: str
    runtime: Dict[str, Any]

class PredictRequest(BaseModel):
    model: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    data: Any

class ErrorPayload(BaseModel):
    type: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)

class ErrorResponse(BaseModel):
    ok: bool = False
    trace_id: str
    error: ErrorPayload

class PredictResponse(BaseModel):
    ok: bool = True
    trace_id: str
    model: str
    model_version: str
    timing_ms: float
    result: Any
    warnings: List[str] = Field(default_factory=list)

REQS = Counter('http_requests_total','HTTP request count',['route','status'])
LAT  = Histogram('http_request_latency_seconds','HTTP request latency (s)',['route'])

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _trace(req: Request) -> str:
    src = f"{time.time()}:{req.client.host if req.client else 'x'}:{req.url.path}:{os.getpid()}"
    return hashlib.sha1(src.encode()).hexdigest()[:16]

class _Limiter:
    def __init__(self, limit=120):
        self.limit = limit
        self.bucket: Dict[str, tuple[int,int]] = {}
    def allow(self, ip: str) -> bool:
        now = int(time.time()//60)
        win,cnt = self.bucket.get(ip,(now,0))
        if win != now:
            self.bucket[ip] = (now,1); return True
        if cnt+1 > self.limit:
            return False
        self.bucket[ip] = (now,cnt+1); return True

def create_app(
    service_name: str,
    version: str,
    predict_fn,
    meta_extra: Dict[str, Any],
    rate_limit_per_min: int = int(os.getenv('RATE_LIMIT_PER_MIN','120'))
) -> FastAPI:
    app = FastAPI(title=service_name, version=version)
    limiter = _Limiter(rate_limit_per_min)

    @app.get("/health", response_model=Health)
    def health():
        return Health()

    @app.get("/meta", response_model=Meta)
    def meta():
        return Meta(
            model=service_name.replace('-', ' ').title(),
            model_version=version,
            git_sha=os.getenv("GIT_SHA","dev"),
            last_updated=_now(),
            trained=bool(meta_extra.get("trained", False)),
            weights_format=meta_extra.get("weights_format",".pt"),
            weights_uri=meta_extra.get("weights_uri","/app/models/model.pt"),
            runtime=meta_extra.get("runtime",{
                "python": os.getenv("PYTHON_VERSION","3.x"),
                "os": os.uname().sysname if hasattr(os, "uname") else "Unknown",
                "docker_image": os.getenv("DOCKER_IMAGE","local"),
            })
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: Request, body: PredictRequest):
        ip = req.client.host if req.client else "x"
        if not limiter.allow(ip):
            trace = _trace(req)
            payload = ErrorResponse(
                trace_id=trace,
                error=ErrorPayload(type="RateLimit", message="Too many requests")
            )
            REQS.labels('/predict','429').inc()
            return JSONResponse(status_code=429, content=payload.model_dump())
        t0 = time.perf_counter()
        trace = _trace(req)
        try:
            result = predict_fn({"params": body.params, "data": body.data})
            dt = (time.perf_counter()-t0)*1000.0
            resp = PredictResponse(
                trace_id=trace,
                model=service_name.replace('-', ' ').title(),
                model_version=version,
                timing_ms=round(dt,3),
                result=result,
            )
            REQS.labels('/predict','200').inc(); LAT.labels('/predict').observe(dt/1000.0)
            h = JSONResponse(resp.model_dump())
        except Exception as e:
            dt = (time.perf_counter()-t0)*1000.0
            err = ErrorResponse(
                trace_id=trace,
                error=ErrorPayload(type=type(e).__name__, message=str(e))
            )
            REQS.labels('/predict','500').inc(); LAT.labels('/predict').observe(dt/1000.0)
            h = JSONResponse(status_code=500, content=err.model_dump())
        h.headers["X-Trace-Id"] = trace
        h.headers["X-Model"] = service_name
        h.headers["X-Model-Version"] = version
        return h

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
