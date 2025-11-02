import os, torch
from typing import Any


class BaseRunner:
    def __init__(self, model_path_env: str = "MODEL_PATH", default_relpath: str = "models/montecarlo.pt"):
        path = os.getenv(model_path_env) or default_relpath
        self.model = self._load(path)

    def _load(self, path: str) -> Any:
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights at {path}: {e}")
