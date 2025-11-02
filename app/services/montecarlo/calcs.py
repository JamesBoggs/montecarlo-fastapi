"""
Stub calcs module for montecarlo — add your real calc functions here.
Each calc should return either a list of y values OR a list of dicts {x, y}.
Use the @registry.register("id", "Label") decorator to expose a route.
"""

from ...core.registry import Registry

registry = Registry(base_prefix="/api/montecarlo")

DESCRIPTOR = {
    "name": "Montecarlo API",
    "description": "Replace this with a proper description.",
    "version": "1.0.0",
}

# Example stub (safe to delete):
@registry.register("example", "Example Series")
def _example():
    return [0, 1, 0.5, 1.25, 0.75]
