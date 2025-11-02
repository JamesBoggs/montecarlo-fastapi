from fastapi import APIRouter
from ..services.montecarlo.calcs import DESCRIPTOR, registry
from ..core.format import meta_payload

router = APIRouter()

@router.get("/meta")
def meta():
    return meta_payload(
        name=DESCRIPTOR.get("name", "montecarlo"),
        description=DESCRIPTOR.get("description", ""),
        metrics=registry.describe(),
        version=DESCRIPTOR.get("version", "1.0.0"),
    )
