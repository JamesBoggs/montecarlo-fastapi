from fastapi import APIRouter
from ..core.registry import mount_metrics
from ..services.montecarlo.calcs import registry

router = APIRouter()
mount_metrics(router, registry)
