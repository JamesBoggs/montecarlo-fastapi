#!/usr/bin/env python3
"""
scaffold_monte_repo.py — CLI to scaffold a FastAPI service repo with the unified
structure (core/, routers/, services/<model>/) and standard payload formatting.

It DOES NOT overwrite your calc functions unless you pass --force.
If a calcs.py already exists, it will be left untouched and wired in.

Usage examples:
  # Work on a local repo path
  python scaffold_monte_repo.py --repo-path /path/to/montecarlo-fastapi

  # Clone a remote repo then scaffold (requires git; optional gh for PR)
  python scaffold_monte_repo.py --repo-url https://github.com/JamesBoggs/montecarlo-fastapi.git \
      --push --create-pr --pr-base main

  # Customize names
  python scaffold_monte_repo.py --repo-path . \
      --model-name montecarlo --api-prefix /api/montecarlo --pt-path models/montecarlo.pt
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# -------------------------- helpers --------------------------

def run(cmd, cwd=None, check=True):
    print("$", " ".join(cmd))
    try:
        cp = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if cp.stdout:
            print(cp.stdout)
        if cp.stderr:
            sys.stderr.write(cp.stderr)
        return cp
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        sys.stderr.write(e.stderr)
        if check:
            raise
        return e


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str, overwrite: bool = False):
    if path.exists() and not overwrite:
        print(f"[skip] {path} exists (use --force to overwrite)")
        return False
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
    print(f"[write] {path}")
    return True


def ensure_init_py(folder: Path):
    ensure_dir(folder)
    initp = folder / "__init__.py"
    if not initp.exists():
        initp.write_text("", encoding="utf-8")
        print(f"[write] {initp}")


def merge_requirements(path: Path, required_lines):
    existing = []
    if path.exists():
        existing = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    merged = existing[:]
    for line in required_lines:
        base = line.split("==")[0] if "==" in line else line
        if not any(x.startswith(base) for x in merged):
            merged.append(line)
    write_file(path, "\n".join(merged) + "\n", overwrite=True)


# -------------------------- templates --------------------------

def tpl_core_format():
    return '''from typing import Any, Dict, List, Iterable


def to_series(data: Any) -> List[Dict[str, float]]:
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            data = data.tolist()
    except Exception:
        pass

    if data is None:
        return []

    if isinstance(data, list) and (not data or isinstance(data[0], (int, float))):
        return [{"x": i, "y": float(y)} for i, y in enumerate(data)]

    if isinstance(data, list) and data and isinstance(data[0], dict):
        kx = next((k for k in data[0] if k.lower() in ("x","t","time","index","idx")), "x")
        ky = next((k for k in data[0] if k.lower() in ("y","v","value","val","price")), "y")
        out = []
        for i, d in enumerate(data):
            x = d.get(kx, i)
            y = d.get(ky, 0.0)
            out.append({"x": (float(x) if isinstance(x,(int,float)) else i), "y": float(y)})
        return out
    return []


def metric_payload(id: str, label: str, series: Any, status: str = "online") -> Dict[str, Any]:
    return {"id": id, "label": label, "status": status, "chartData": to_series(series)}


def meta_payload(name: str, description: str, metrics: Iterable[Dict[str, str]], version: str = "1.0.0") -> Dict[str, Any]:
    return {"name": name, "description": description, "version": version, "metrics": list(metrics)}
'''


def tpl_core_registry():
    return '''from typing import Callable, Dict, List
from fastapi import APIRouter
from .format import metric_payload


class _Metric:
    def __init__(self, id: str, label: str, fn: Callable):
        self.id = id
        self.label = label
        self.fn = fn


class Registry:
    def __init__(self, base_prefix: str):
        self.base_prefix = base_prefix.rstrip("/")
        self._items: Dict[str, _Metric] = {}

    def register(self, id: str, label: str):
        def deco(fn: Callable):
            self._items[id] = _Metric(id, label, fn)
            return fn
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
            try:
                if getattr(fn, "__code__", None) and (fn.__code__.co_flags & 0x80):
                    out = await fn()
                else:
                    out = fn()
                return metric_payload(id=_id, label=label, series=out, status="online")
            except Exception:
                return metric_payload(id=_id, label=label, series=[], status="offline")

        router.add_api_route(path, handler, methods=["GET"])\n'''


def tpl_core_runner(model_pt_relpath: str):
    return f'''import os, torch
from typing import Any


class BaseRunner:
    def __init__(self, model_path_env: str = "MODEL_PATH", default_relpath: str = "{model_pt_relpath}"):
        path = os.getenv(model_path_env) or default_relpath
        self.model = self._load(path)

    def _load(self, path: str) -> Any:
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights at {{path}}: {{e}}")
'''


def tpl_services_model(model_name: str, model_pt_relpath: str):
    return f'''from ...core.runner import BaseRunner


class {model_name.capitalize()}Runner(BaseRunner):
    def __init__(self):
        super().__init__("MODEL_PATH", "{model_pt_relpath}")
        w = self.model if isinstance(self.model, dict) else {{}}
        self.mu = float(w.get("mu", 0.05))
        self.sigma = float(w.get("sigma", 0.20))

runner = {model_name.capitalize()}Runner()
'''


def tpl_routers_health():
    return '''from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {"ok": True}
'''


def tpl_routers_meta(model_name: str):
    return f'''from fastapi import APIRouter
from ..services.{model_name}.calcs import DESCRIPTOR, registry
from ..core.format import meta_payload

router = APIRouter()

@router.get("/meta")
def meta():
    return meta_payload(
        name=DESCRIPTOR.get("name", "{model_name}"),
        description=DESCRIPTOR.get("description", ""),
        metrics=registry.describe(),
        version=DESCRIPTOR.get("version", "1.0.0"),
    )
'''


def tpl_routers_metrics(model_name: str):
    return f'''from fastapi import APIRouter
from ..core.registry import mount_metrics
from ..services.{model_name}.calcs import registry

router = APIRouter()
mount_metrics(router, registry)
'''


def tpl_services_calcs_stub(model_name: str, api_prefix: str):
    return f'''"""
Stub calcs module for {model_name} — add your real calc functions here.
Each calc should return either a list of y values OR a list of dicts {{x, y}}.
Use the @registry.register("id", "Label") decorator to expose a route.
"""

from ...core.registry import Registry

registry = Registry(base_prefix="{api_prefix}")

DESCRIPTOR = {{
    "name": "{model_name.capitalize()} API",
    "description": "Replace this with a proper description.",
    "version": "1.0.0",
}}

# Example stub (safe to delete):
@registry.register("example", "Example Series")
def _example():
    return [0, 1, 0.5, 1.25, 0.75]
'''


def tpl_app_main(service_title: str):
    return f'''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, meta, metrics

app = FastAPI(title="{service_title}", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(meta.router)
app.include_router(metrics.router)

@app.get("/__routes")
def __routes():
    return [{{"path": r.path, "methods": list(getattr(r, "methods", []))}} for r in app.router.routes]
'''

# -------------------------- main logic --------------------------

def scaffold(repo_dir: Path, model_name: str, api_prefix: str, pt_path: str, force: bool):
    print(f"[info] Scaffolding in {repo_dir}")

    # app/ package and subpackages
    ensure_init_py(repo_dir / "app")
    ensure_init_py(repo_dir / "app" / "core")
    ensure_init_py(repo_dir / "app" / "routers")
    ensure_init_py(repo_dir / "app" / "services")
    ensure_init_py(repo_dir / "app" / "services" / model_name)

    # models folder
    ensure_dir(repo_dir / "models")
    if not (repo_dir / pt_path).exists():
        print(f"[warn] PT weights expected at {pt_path} (not found). You can add them later.")

    # core files
    write_file(repo_dir / "app/core/format.py", tpl_core_format(), overwrite=force)
    write_file(repo_dir / "app/core/registry.py", tpl_core_registry(), overwrite=force)
    write_file(repo_dir / "app/core/runner.py", tpl_core_runner(pt_path), overwrite=force)

    # routers
    write_file(repo_dir / "app/routers/health.py", tpl_routers_health(), overwrite=force)
    write_file(repo_dir / "app/routers/meta.py", tpl_routers_meta(model_name), overwrite=force)
    write_file(repo_dir / "app/routers/metrics.py", tpl_routers_metrics(model_name), overwrite=force)

    # services/<model>
    write_file(repo_dir / f"app/services/{model_name}/model.py", tpl_services_model(model_name, pt_path), overwrite=force)

    calcs_path = repo_dir / f"app/services/{model_name}/calcs.py"
    if not calcs_path.exists() or force:
        write_file(calcs_path, tpl_services_calcs_stub(model_name, api_prefix), overwrite=force)
    else:
        print(f"[keep] Existing {calcs_path} preserved (use --force to overwrite)")

    # main
    write_file(repo_dir / "app/main.py", tpl_app_main(f"{model_name.capitalize()} API"), overwrite=force)

    # requirements
    merge_requirements(
        repo_dir / "requirements.txt",
        [
            "fastapi==0.115.0",
            "uvicorn==0.30.6",
            "pydantic==2.9.2",
            "numpy==2.1.2",
            "torch==2.6.0",
        ],
    )

    print("[done] Scaffolding complete.")


def ensure_git(repo_dir: Path):
    if not (repo_dir / ".git").exists():
        raise SystemExit(f"No .git directory found at {repo_dir}. Clone the repo first or pass --repo-url.")


def commit_push(repo_dir: Path, branch: str, push: bool, pr: bool, pr_base: str, pr_title: str, pr_body: str):
    ensure_git(repo_dir)
    # create/switch branch
    run(["git", "switch", "-c", branch], cwd=repo_dir, check=False)
    run(["git", "add", "-A"], cwd=repo_dir)
    run(["git", "commit", "-m", pr_title], cwd=repo_dir, check=False)

    if push:
        run(["git", "push", "-u", "origin", branch], cwd=repo_dir, check=False)

    if pr:
        # Try GitHub CLI; if missing, just print instructions
        res = run(["gh", "--version"], cwd=repo_dir, check=False)
        if isinstance(res, subprocess.CalledProcessError):
            print("[info] gh CLI not found. To open a PR manually:")
            print(f"      Create PR from branch '{branch}' -> base '{pr_base}' with title '{pr_title}'")
        else:
            run([
                "gh", "pr", "create",
                "-B", pr_base,
                "-t", pr_title,
                "-b", pr_body,
            ], cwd=repo_dir, check=False)


def main():
    ap = argparse.ArgumentParser(description="Scaffold a FastAPI service repo with standard structure.")
    gsrc = ap.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--repo-path", type=str, help="Path to an existing local git repo")
    gsrc.add_argument("--repo-url", type=str, help="Git URL to clone (https or ssh)")

    ap.add_argument("--model-name", default="montecarlo", help="Service model name (folder under app/services)")
    ap.add_argument("--api-prefix", default="/api/montecarlo", help="Base API prefix for mounted calc routes")
    ap.add_argument("--pt-path", default="models/montecarlo.pt", help="Relative path to .pt weights inside repo")
    ap.add_argument("--branch", default="chore/structure-boilerplate", help="Branch name for changes")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")

    ap.add_argument("--push", action="store_true", help="git push the branch to origin")
    ap.add_argument("--create-pr", action="store_true", help="open a PR using gh CLI if available")
    ap.add_argument("--pr-base", default="main", help="PR base branch")

    args = ap.parse_args()

    tmpdir = None
    try:
        if args.repo_url:
            tmpdir = tempfile.mkdtemp(prefix="monte_scaffold_")
            run(["git", "clone", args.repo_url, tmpdir])
            repo_dir = Path(tmpdir)
        else:
            repo_dir = Path(args.repo_path).resolve()

        scaffold(repo_dir, args.model_name, args.api_prefix, args.pt_path, args.force)

        pr_title = "chore(structure): scaffold core/routers/services layout with registry + meta"
        pr_body = (
            "This PR introduces a standard FastAPI structure with unified payloads,\n"
            "auto-mounted metric routes via a registry, and /meta + /health endpoints.\n\n"
            "- app/core: format.py, registry.py, runner.py\n"
            "- app/routers: health.py, meta.py, metrics.py\n"
            f"- app/services/{args.model_name}: model.py (+ calcs stub if missing)\n"
            "- app/main.py\n"
            "- requirements merged\n\n"
            "Your existing calcs (if any) were preserved. Add/port them to services/<model>/calcs.py and redeploy."
        )
        commit_push(repo_dir, args.branch, args.push, args.create_pr, args.pr_base, pr_title, pr_body)

        print("\n[OK] Done. If you cloned to a temp dir, review, then push/PR as desired.")
    finally:
        if tmpdir:
            # Keep the cloned dir to let user inspect; comment out to auto-clean
            print(f"[info] cloned into {tmpdir}")


if __name__ == "__main__":
    main()
