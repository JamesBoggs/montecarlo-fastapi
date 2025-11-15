# app/routers/meta.py
from fastapi import APIRouter

from ..core.format import meta_payload
from ..services.montecarlo.calcs import DESCRIPTOR, meta as engine_meta

router = APIRouter()


@router.get("/api/montecarlo/meta")
def get_meta():
    """
    Meta endpoint for the Monte Carlo service.
    Combines static descriptor with engine meta info.
    """
    eng = engine_meta()  # {"engine": "...", "version": "...", "device": "..."}

    return meta_payload(
        name=DESCRIPTOR["name"],
        description=DESCRIPTOR["description"],
        version=DESCRIPTOR["version"],
        engine=eng.get("engine", "montecarlo"),
        device=eng.get("device", "cpu"),
        metrics=DESCRIPTOR.get("metrics", []),
    )
