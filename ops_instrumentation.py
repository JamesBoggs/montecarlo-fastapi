from __future__ import annotations
import os, time, hashlib
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Tuple
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
REQS = Counter('http_requests_total','HTTP request count',['route','status'])
LAT  = Histogram('http_request_latency_seconds','HTTP request latency (s)',['route'])
RATE_LIMIT_PER_MIN = int(os.getenv('RATE_LIMIT_PER_MIN','120'))
def _now(): return datetime.now(timezone.utc).isoformat()
def _trace(req: Request)->str:
    src=f"{time.time()}:{req.client.host if req.client else 'x'}:{req.url.path}:{os.getpid()}";return hashlib.sha1(src.encode()).hexdigest()[:16]
class _Limiter:
    def __init__(self,limit=120): self.limit=limit; self.bucket: Dict[str,Tuple[int,int]]={}
    def allow(self,ip:str)->bool:
        now=int(time.time()//60); win,cnt=self.bucket.get(ip,(now,0))
        if win!=now: win,cnt=now,0
        cnt+=1; self.bucket[ip]=(win,cnt); return cnt<=self.limit
LIMITER=_Limiter(RATE_LIMIT_PER_MIN)
def attach_ops(app: FastAPI, ready_check: Optional[Callable[[], bool]] = None,
               model_name: Optional[str]=None, model_version: Optional[str]=None)->None:
    MODEL = model_name or os.getenv('MODEL_NAME', app.title or 'model')
    MODEL_VER = model_version or os.getenv('MODEL_VERSION','0.1.0')
    STARTED = _now()
    @app.middleware('http')
    async def _mw(request: Request, call_next):
        ip=request.client.host if request.client else 'unknown'
        if not LIMITER.allow(ip):
            REQS.labels(request.url.path,'429').inc();return JSONResponse({'detail':'rate limit'}, status_code=429)
        t0=time.time()
        try:
            resp: Response = await call_next(request)
        finally:
            LAT.labels(request.url.path).observe(time.time()-t0)
        resp.headers['X-Trace-Id']=_trace(request)
        resp.headers['X-Model-Version']=MODEL_VER
        return resp
    @app.get('/health')
    async def _health(): return {'ok': True, 'model': MODEL, 'model_version': MODEL_VER}
    @app.get('/ready')
    async def _ready(): return JSONResponse({'ready': True, 'started': STARTED}, status_code=200)
    @app.get('/meta')
    async def _meta(): return {'model': MODEL, 'model_version': MODEL_VER, 'git_sha': os.getenv('GIT_SHA','dev'), 'last_updated': STARTED}
    @app.get('/metrics')
    async def _metrics(): return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
