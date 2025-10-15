# meta.py
import os, platform, datetime as dt
from fastapi import APIRouter

router = APIRouter()

@router.get("/meta")
def meta():
    return {
        "model": os.getenv("MODEL_NAME", "FastAPI"),
        "model_version": os.getenv("MODEL_VERSION", "0.1.0"),
        "git_sha": os.getenv("GIT_SHA", "dev"),
        "last_updated": dt.datetime.utcnow().isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "docker_image": os.getenv("DOCKER_IMAGE", "local/dev"),
        },
        "trained": bool(os.getenv("TRAINED_WEIGHTS", "")),
        "weights_format": os.getenv("WEIGHTS_FORMAT", ""),  # ".pt" or ".pts"
        "weights_uri": os.getenv("TRAINED_WEIGHTS", ""),
    }
