# app/services/montecarlo/calcs.py

from ...core.registry import Registry
from .model import runner
import math, random
import numpy as np

registry = Registry(base_prefix="/api/montecarlo")

DESCRIPTOR = {
    "name": "Monte Carlo Simulator",
    "description": "Simulates price paths and volatility shocks using stochastic models.",
    "version": "2.1.0",
}

def simulate_paths(S0, T, steps, paths, mu, sigma):
    dt = T / steps
    results = np.zeros((paths, steps + 1))
    for i in range(paths):
        prices = [S0]
        for _ in range(steps):
            dW = np.random.normal(0, math.sqrt(dt))
            S_next = prices[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            prices.append(S_next)
        results[i] = prices
    return results

@registry.register("var", "Value-at-Risk (VaR)")
def value_at_risk():
    S0 = 100.0
    results = simulate_paths(S0, 1.0, 252, 2000, runner.mu, runner.sigma)
    returns = (results[:, -1] - S0) / S0
    var = np.percentile(returns, 5)
    return [{"x": 0, "y": 0}, {"x": 1, "y": float(var)}]

@registry.register("cvar", "Conditional VaR (CVaR)")
def conditional_var():
    S0 = 100.0
    results = simulate_paths(S0, 1.0, 252, 2000, runner.mu, runner.sigma)
    returns = (results[:, -1] - S0) / S0
    var = np.percentile(returns, 5)
    cvar = returns[returns <= var].mean()
    return [{"x": 0, "y": 0}, {"x": 1, "y": float(cvar)}]

@registry.register("sharpe", "Sharpe Ratio")
def sharpe_ratio():
    S0 = 100.0
    results = simulate_paths(S0, 1.0, 252, 2000, runner.mu, runner.sigma)
    rets = (results[:, -1] - S0) / S0
    sharpe = rets.mean() / rets.std()
    return [{"x": 0, "y": 0}, {"x": 1, "y": float(sharpe)}]

@registry.register("stress", "Stress Testing")
def stress_test():
    S0 = 100.0
    stressed = simulate_paths(S0, 1.0, 252, 1000, runner.mu, runner.sigma * 2)
    mean_return = ((stressed[:, -1] - S0) / S0).mean()
    return [{"x": 0, "y": 0}, {"x": 1, "y": float(mean_return)}]

@registry.register("options", "Option Pricing")
def option_pricing():
    S0 = 100.0; K = 100.0; r = 0.03; T = 1.0
    results = simulate_paths(S0, T, 252, 5000, runner.mu, runner.sigma)
    payoff = np.maximum(results[:, -1] - K, 0)
    price = np.exp(-r * T) * payoff.mean()
    return [{"x": 0, "y": float(price)}]

@registry.register("portfolio", "Portfolio Simulation")
def portfolio_sim():
    S0 = 100.0
    w = np.array([0.5, 0.5])
    assets = [simulate_paths(S0, 1.0, 252, 2000, runner.mu, runner.sigma * (i + 1)) for i in range(len(w))]
    returns = np.vstack([(a[:, -1] - S0) / S0 for a in assets])
    port_ret = (w @ returns).mean()
    port_std = math.sqrt(w @ np.cov(returns) @ w.T)
    return [{"x": 0, "y": float(port_ret)}, {"x": 1, "y": float(port_std)}]

@registry.register("uncertainty", "Forecast Error Bands")
def forecast_uncertainty():
    S0 = 100.0
    results = simulate_paths(S0, 1.0, 252, 2000, runner.mu, runner.sigma)
    final = results[:, -1]
    ci_low, ci_high = np.percentile(final, [5, 95])
    return [{"x": 0, "y": float(ci_low)}, {"x": 1, "y": float(ci_high)}]

@registry.register("drawdown", "Drawdown Analysis")
def drawdown():
    S0 = 100.0
    results = simulate_paths(S0, 1.0, 252, 2000, runner.mu, runner.sigma)
    dd = (results.max(axis=1) - results[:, -1]) / results.max(axis=1)
    return [{"x": 0, "y": float(dd.mean())}]

@registry.register("corrshift", "Correlation Breakdown")
def correlation_shift():
    S0 = 100.0
    a = simulate_paths(S0, 1.0, 252, 1000, runner.mu, runner.sigma)
    b = simulate_paths(S0, 1.0, 252, 1000, runner.mu * 0.8, runner.sigma * 1.2)
    corr = np.corrcoef(a[:, -1], b[:, -1])[0, 1]
    return [{"x": 0, "y": float(corr)}]

@registry.register("tailrisk", "Tail Risk / Skewness")
def tail_risk():
    S0 = 100.0
    results = simulate_paths(S0, 1.0, 252, 2000, runner.mu, runner.sigma)
    returns = (results[:, -1] - S0) / S0
    skew = ((returns - returns.mean())**3).mean() / returns.std()**3
    kurt = ((returns - returns.mean())**4).mean() / returns.std()**4
    return [{"x": 0, "y": float(skew)}, {"x": 1, "y": float(kurt)}]
