# app/calcs.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# -------------------------------------------------------------------
# Device / engine setup
# -------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SimulationConfig:
    spot: float = 100.0
    mu: float = 0.05            # annual drift
    sigma: float = 0.20         # annual volatility
    steps: int = 252
    paths: int = 10_000
    dt: float = 1.0 / 252.0
    risk_free: float = 0.02     # annual risk-free rate for Sharpe / options
    seed: Optional[int] = None


@dataclass
class PortfolioConfig:
    spots: Sequence[float]
    mus: Sequence[float]
    sigmas: Sequence[float]
    weights: Sequence[float]
    corr: Optional[Sequence[Sequence[float]]] = None
    steps: int = 252
    paths: int = 10_000
    dt: float = 1.0 / 252.0
    risk_free: float = 0.02
    seed: Optional[int] = None


@dataclass
class OptionConfig:
    spot: float
    strike: float
    maturity_years: float
    mu: float
    sigma: float
    risk_free: float
    steps: int = 252
    paths: int = 50_000
    is_call: bool = True
    seed: Optional[int] = None


# Singleton-style engine so it "loads" once at startup.
class MonteCarloEngine:
    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or DEVICE

    # ---------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)

    # Single-asset GBM paths: shape (paths, steps + 1)
    def simulate_paths(self, cfg: SimulationConfig) -> torch.Tensor:
        self._set_seed(cfg.seed)

        spot = torch.tensor(cfg.spot, device=self.device, dtype=torch.float32)
        mu = torch.tensor(cfg.mu, device=self.device, dtype=torch.float32)
        sigma = torch.tensor(cfg.sigma, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # (paths, steps) normal shocks
            z = torch.randn(cfg.paths, cfg.steps, device=self.device)

            drift = (mu - 0.5 * sigma**2) * cfg.dt
            diffusion = sigma * math.sqrt(cfg.dt) * z
            increments = drift + diffusion  # (paths, steps)

            log_paths = torch.cumsum(increments, dim=1)
            zeros = torch.zeros(cfg.paths, 1, device=self.device)
            log_paths = torch.cat([zeros, log_paths], dim=1)  # (paths, steps + 1)

            paths = spot * torch.exp(log_paths)
            return paths  # (paths, steps + 1)

    # Multi-asset correlated GBM + portfolio aggregation
    def simulate_portfolio_paths(self, cfg: PortfolioConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        self._set_seed(cfg.seed)

        spots = torch.tensor(cfg.spots, device=self.device, dtype=torch.float32)
        mus = torch.tensor(cfg.mus, device=self.device, dtype=torch.float32)
        sigmas = torch.tensor(cfg.sigmas, device=self.device, dtype=torch.float32)
        weights = torch.tensor(cfg.weights, device=self.device, dtype=torch.float32)

        n_assets = spots.shape[0]

        if cfg.corr is not None:
            corr = torch.tensor(cfg.corr, device=self.device, dtype=torch.float32)
        else:
            corr = torch.eye(n_assets, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # Cholesky for correlation
            L = torch.linalg.cholesky(corr)  # (n_assets, n_assets)

            # IID standard normals then correlate: (paths, steps, n_assets)
            z = torch.randn(cfg.paths, cfg.steps, n_assets, device=self.device)
            # reshape to (paths * steps, n_assets) for matmul, then back
            z_flat = z.view(-1, n_assets)
            y_flat = z_flat @ L.T
            y = y_flat.view(cfg.paths, cfg.steps, n_assets)

            drift = (mus - 0.5 * sigmas**2) * cfg.dt  # (n_assets,)
            diffusion_coeff = sigmas * math.sqrt(cfg.dt)  # (n_assets,)

            # broadcast drift & diffusion to (paths, steps, n_assets)
            increments = drift.view(1, 1, n_assets) + diffusion_coeff.view(
                1, 1, n_assets
            ) * y  # (paths, steps, n_assets)

            log_paths = torch.cumsum(increments, dim=1)  # (paths, steps, n_assets)
            zeros = torch.zeros(cfg.paths, 1, n_assets, device=self.device)
            log_paths = torch.cat([zeros, log_paths], dim=1)  # (paths, steps+1, n_assets)

            # asset price paths
            asset_paths = spots.view(1, 1, n_assets) * torch.exp(log_paths)

            # portfolio value = sum(weights * asset_prices) along asset dim
            portfolio_paths = (asset_paths * weights.view(1, 1, n_assets)).sum(dim=2)

            return portfolio_paths, asset_paths

    # ---------------------------------------------------------------
    # Risk metrics
    # ---------------------------------------------------------------

    def compute_returns(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Convert price paths (paths, steps+1) into simple returns from t=0 to t=T:
        r = (S_T / S_0) - 1
        """
        start = paths[:, 0]
        end = paths[:, -1]
        returns = (end / start) - 1.0
        return returns

    def compute_var(self, returns: torch.Tensor, alpha: float = 0.95) -> float:
        """
        VaR at confidence alpha; positive number representing loss.
        """
        with torch.no_grad():
            q = torch.quantile(returns, 1 - alpha)  # e.g. 0.05 quantile for 95% VaR
            var = float(-q.item())
        return var

    def compute_cvar(self, returns: torch.Tensor, alpha: float = 0.95) -> float:
        """
        CVaR (Expected Shortfall) at confidence alpha.
        """
        with torch.no_grad():
            thresh = torch.quantile(returns, 1 - alpha)
            tail = returns[returns <= thresh]
            if tail.numel() == 0:
                return 0.0
            cvar = float(-tail.mean().item())
        return cvar

    def compute_sharpe(
        self, returns: torch.Tensor, risk_free: float = 0.0, periods_per_year: int = 252
    ) -> float:
        """
        Annualized Sharpe ratio from terminal returns.
        We treat 'returns' as a 1-period return at the whole horizon and back out
        a rough per-period equivalent.
        """
        with torch.no_grad():
            r_mean = returns.mean()
            r_std = returns.std(unbiased=False)
            if r_std.abs() < 1e-8:
                return 0.0

            # Approximate per-period mean and std via horizon scaling
            horizon_years = 1.0  # config dt * steps ~= 1 year for a typical setup
            rf_annual = risk_free
            excess = r_mean - rf_annual * horizon_years
            sharpe = excess / r_std
            return float(sharpe.item())

    # ---------------------------------------------------------------
    # Stress testing, options, uncertainty
    # ---------------------------------------------------------------

    def run_stress_test(
        self,
        cfg: SimulationConfig,
        vol_shock: float = 0.5,
        drift_shock: float = -0.5,
    ) -> Dict[str, Any]:
        """
        Simple stress test: shock drift and vol by multipliers and compare VaR / CVaR.
        vol_shock = 0.5 => +50% vol,  drift_shock = -0.5 => -50% drift.
        """
        base_paths = self.simulate_paths(cfg)
        base_returns = self.compute_returns(base_paths)
        base_var = self.compute_var(base_returns)
        base_cvar = self.compute_cvar(base_returns)

        stressed_cfg = SimulationConfig(
            spot=cfg.spot,
            mu=cfg.mu * (1.0 + drift_shock),
            sigma=cfg.sigma * (1.0 + vol_shock),
            steps=cfg.steps,
            paths=cfg.paths,
            dt=cfg.dt,
            risk_free=cfg.risk_free,
            seed=cfg.seed,
        )

        stressed_paths = self.simulate_paths(stressed_cfg)
        stressed_returns = self.compute_returns(stressed_paths)
        stressed_var = self.compute_var(stressed_returns)
        stressed_cvar = self.compute_cvar(stressed_returns)

        return {
            "base": {
                "mu": cfg.mu,
                "sigma": cfg.sigma,
                "var_95": base_var,
                "cvar_95": base_cvar,
            },
            "stressed": {
                "mu": stressed_cfg.mu,
                "sigma": stressed_cfg.sigma,
                "var_95": stressed_var,
                "cvar_95": stressed_cvar,
            },
        }

    def price_european_option_mc(self, cfg: OptionConfig) -> Dict[str, Any]:
        """
        European option pricing via Monte Carlo under GBM.
        """
        sim_cfg = SimulationConfig(
            spot=cfg.spot,
            mu=cfg.mu,
            sigma=cfg.sigma,
            steps=cfg.steps,
            paths=cfg.paths,
            dt=cfg.maturity_years / cfg.steps,
            risk_free=cfg.risk_free,
            seed=cfg.seed,
        )

        paths = self.simulate_paths(sim_cfg)  # (paths, steps+1)
        S_T = paths[:, -1]

        with torch.no_grad():
            if cfg.is_call:
                payoff = torch.clamp(S_T - cfg.strike, min=0.0)
            else:
                payoff = torch.clamp(cfg.strike - S_T, min=0.0)

            disc_factor = math.exp(-cfg.risk_free * cfg.maturity_years)
            price = disc_factor * payoff.mean()
            stderr = disc_factor * payoff.std(unbiased=False) / math.sqrt(cfg.paths)

            return {
                "price": float(price.item()),
                "stderr": float(stderr.item()),
                "is_call": cfg.is_call,
                "strike": cfg.strike,
                "maturity_years": cfg.maturity_years,
                "spot": cfg.spot,
            }

    def compute_uncertainty_bands(
        self,
        cfg: SimulationConfig,
        quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> Dict[str, Any]:
        """
        Confidence bands across time for price paths.
        """
        paths = self.simulate_paths(cfg)  # (paths, steps+1)

        with torch.no_grad():
            qs_tensor = torch.tensor(quantiles, device=self.device)
            bands = torch.quantile(paths, qs_tensor, dim=0)  # (len(q), steps+1)

            time_grid = [float(i * cfg.dt) for i in range(paths.shape[1])]
            bands_dict = {
                str(q): bands[i].cpu().tolist()
                for i, q in enumerate(quantiles)
            }

        return {
            "time": time_grid,
            "bands": bands_dict,
        }

    # ---------------------------------------------------------------
    # Standardized platform surface
    # ---------------------------------------------------------------

    def run_core(self, cfg: SimulationConfig) -> Dict[str, Any]:
        """
        Core simulation used by the main dashboard card:
        - GBM paths for a single asset
        - VaR / CVaR / Sharpe summary
        - Sample of paths for plotting
        """
        paths = self.simulate_paths(cfg)
        returns = self.compute_returns(paths)
        var_95 = self.compute_var(returns, alpha=0.95)
        cvar_95 = self.compute_cvar(returns, alpha=0.95)
        sharpe = self.compute_sharpe(returns, risk_free=cfg.risk_free)

        # Small subset of paths for charting
        max_paths_for_chart = min(50, cfg.paths)
        sample_paths = paths[:max_paths_for_chart].cpu().tolist()

        return {
            "engine": "montecarlo",
            "summary": {
                "expected_return": float(returns.mean().item()),
                "volatility": float(returns.std(unbiased=False).item()),
                "var_95": var_95,
                "cvar_95": cvar_95,
                "sharpe": sharpe,
            },
            "paths_sample": sample_paths,
            "time": [float(i * cfg.dt) for i in range(paths.shape[1])],
            "config": {
                "spot": cfg.spot,
                "mu": cfg.mu,
                "sigma": cfg.sigma,
                "steps": cfg.steps,
                "paths": cfg.paths,
                "dt": cfg.dt,
                "risk_free": cfg.risk_free,
            },
        }

    def run_scenarios(
        self,
        base_cfg: SimulationConfig,
        vol_shocks: Sequence[float],
        drift_shocks: Sequence[float],
    ) -> Dict[str, Any]:
        """
        Scenario grid over vol & drift shocks, returning VaR / CVaR surfaces.
        """
        results: List[Dict[str, Any]] = []

        for dv in drift_shocks:
            for vs in vol_shocks:
                stressed = SimulationConfig(
                    spot=base_cfg.spot,
                    mu=base_cfg.mu * (1.0 + dv),
                    sigma=base_cfg.sigma * (1.0 + vs),
                    steps=base_cfg.steps,
                    paths=base_cfg.paths,
                    dt=base_cfg.dt,
                    risk_free=base_cfg.risk_free,
                    seed=base_cfg.seed,
                )
                paths = self.simulate_paths(stressed)
                returns = self.compute_returns(paths)
                var_95 = self.compute_var(returns, alpha=0.95)
                cvar_95 = self.compute_cvar(returns, alpha=0.95)

                results.append(
                    {
                        "drift_shock": dv,
                        "vol_shock": vs,
                        "var_95": var_95,
                        "cvar_95": cvar_95,
                    }
                )

        return {
            "engine": "montecarlo",
            "base": {
                "mu": base_cfg.mu,
                "sigma": base_cfg.sigma,
            },
            "scenarios": results,
        }

    def run_backtest(
        self,
        realized_returns: Sequence[float],
        alpha: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Sketch of a VaR backtest: given historical realized returns and a fixed
        VaR level, count breaches. This can be extended later.
        """
        if not realized_returns:
            return {
                "engine": "montecarlo",
                "backtest": {
                    "alpha": alpha,
                    "breaches": 0,
                    "n_obs": 0,
                    "breach_rate": 0.0,
                },
            }

        r = torch.tensor(realized_returns, device=self.device, dtype=torch.float32)
        # naive: use historical percentile as a VaR proxy
        var_est = torch.quantile(r, 1 - alpha)
        breaches = (r <= var_est).sum().item()
        n_obs = r.numel()
        breach_rate = breaches / n_obs

        return {
            "engine": "montecarlo",
            "backtest": {
                "alpha": alpha,
                "var_proxy": float(-var_est.item()),
                "breaches": int(breaches),
                "n_obs": int(n_obs),
                "breach_rate": float(breach_rate),
            },
        }

    def get_meta(self) -> Dict[str, Any]:
        """
        Metadata endpoint for the engine.
        """
        return {
            "engine": "montecarlo",
            "version": "2.1.0",
            "device": str(self.device),
            "defaults": {
                "spot": 100.0,
                "mu": 0.05,
                "sigma": 0.20,
                "steps": 252,
                "paths": 10_000,
                "dt": 1.0 / 252.0,
            },
        }


# -------------------------------------------------------------------
# Module-level singleton + convenience wrappers
# -------------------------------------------------------------------

_engine = MonteCarloEngine()


def get_engine() -> MonteCarloEngine:
    return _engine


# Facade functions your routers can import and call.

def core_simulation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Thin wrapper for run_core() taking a dict (e.g. Pydantic model .dict()).
    """
    cfg = SimulationConfig(
        spot=payload.get("spot", 100.0),
        mu=payload.get("mu", 0.05),
        sigma=payload.get("sigma", 0.20),
        steps=payload.get("steps", 252),
        paths=payload.get("paths", 10_000),
        dt=payload.get("dt", 1.0 / 252.0),
        risk_free=payload.get("risk_free", 0.02),
        seed=payload.get("seed"),
    )
    return _engine.run_core(cfg)


def scenarios_simulation(payload: Dict[str, Any]) -> Dict[str, Any]:
    base_cfg = SimulationConfig(
        spot=payload.get("spot", 100.0),
        mu=payload.get("mu", 0.05),
        sigma=payload.get("sigma", 0.20),
        steps=payload.get("steps", 252),
        paths=payload.get("paths", 10_000),
        dt=payload.get("dt", 1.0 / 252.0),
        risk_free=payload.get("risk_free", 0.02),
        seed=payload.get("seed"),
    )
    vol_shocks = payload.get("vol_shocks", [-0.5, -0.25, 0.0, 0.25, 0.5])
    drift_shocks = payload.get("drift_shocks", [-0.5, -0.25, 0.0, 0.25, 0.5])
    return _engine.run_scenarios(base_cfg, vol_shocks=vol_shocks, drift_shocks=drift_shocks)


def uncertainty_bands(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = SimulationConfig(
        spot=payload.get("spot", 100.0),
        mu=payload.get("mu", 0.05),
        sigma=payload.get("sigma", 0.20),
        steps=payload.get("steps", 252),
        paths=payload.get("paths", 50_000),
        dt=payload.get("dt", 1.0 / 252.0),
        risk_free=payload.get("risk_free", 0.02),
        seed=payload.get("seed"),
    )
    quantiles = payload.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])
    return _engine.compute_uncertainty_bands(cfg, quantiles=quantiles)


def stress_test(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = SimulationConfig(
        spot=payload.get("spot", 100.0),
        mu=payload.get("mu", 0.05),
        sigma=payload.get("sigma", 0.20),
        steps=payload.get("steps", 252),
        paths=payload.get("paths", 10_000),
        dt=payload.get("dt", 1.0 / 252.0),
        risk_free=payload.get("risk_free", 0.02),
        seed=payload.get("seed"),
    )
    vol_shock = payload.get("vol_shock", 0.5)
    drift_shock = payload.get("drift_shock", -0.5)
    return _engine.run_stress_test(cfg, vol_shock=vol_shock, drift_shock=drift_shock)


def option_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = OptionConfig(
        spot=payload["spot"],
        strike=payload["strike"],
        maturity_years=payload["maturity_years"],
        mu=payload.get("mu", 0.05),
        sigma=payload.get("sigma", 0.20),
        risk_free=payload.get("risk_free", 0.02),
        steps=payload.get("steps", 252),
        paths=payload.get("paths", 50_000),
        is_call=payload.get("is_call", True),
        seed=payload.get("seed"),
    )
    return _engine.price_european_option_mc(cfg)


def meta() -> Dict[str, Any]:
    return _engine.get_meta()
