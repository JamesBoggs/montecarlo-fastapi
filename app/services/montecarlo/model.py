from ...core.runner import BaseRunner


class MontecarloRunner(BaseRunner):
    def __init__(self):
        super().__init__("MODEL_PATH", "models/montecarlo.pt")
        w = self.model if isinstance(self.model, dict) else {}
        self.mu = float(w.get("mu", 0.05))
        self.sigma = float(w.get("sigma", 0.20))

runner = MontecarloRunner()
