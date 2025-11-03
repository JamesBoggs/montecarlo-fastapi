# app/core/registry.py
from typing import Callable, Dict, List
from fastapi import APIRouter
from .format import metric_payload
import time

# --- add this tiny cache ---
_CACHE_TTL = 30  # seconds
_cache: Dict[str, tuple] = {}  # id -> (ts, payload)

def _get_cached(mid: str):
    t = _cache.get(mid)
    if not t: return None
    ts, pay = t
    if time.time() - ts > _CACHE_TTL:
        _cache.pop(mid, None)
        return None
    return pay

def _set_cached(mid: str, payload):
    _cache[mid] = (time.time(), payload)
# ---------------------------

class _Metric:
    def __init__(self, id: str, label: str, fn: Callable):
        self.id = id; self.label = label; self.fn = fn

class Registry:
    def __init__(self, base_prefix: str):
        self.base_prefix = base_prefix.rstrip("/")
        self._items: Dict[str, _Metric] = {}

    def register(self, id: str, label: str):
        def deco(fn: Callable):
            self._items[id] = _Metric(id, label, fn); return fn
        return deco

    @property
    def items(self) -> List[_Metric]:
        return list(self._items.values())

    def describe(self) -> List[Dict[str, str]]:
        return [{"id": m.id, "label": m.label, "endpoint": f"{self.base_prefix}/{m.id}"} for m in self.items]

def mount_metrics(router: APIRouter, reg: Registry):
    for m in reg.items:
        path = f"{reg.base_prefix}/{m.id}"
        async def handler(fn=m.fn, label=m.label, _id=m.id):
            # serve from cache if warm
            got = _get_cached(_id)
            if got is not None:
                return got
            try:
                out = (await fn()) if (getattr(fn, "__code__", None) and fn.__code__.co_flags & 0x80) else fn()
                payload = metric_payload(id=_id, label=label, series=out, status="online")
            except Exception:
                payload = metric_payload(id=_id, label=label, series=[], status="offline")
            _set_cached(_id, payload)
            return payload
        router.add_api_route(path, handler, methods=["GET"])
