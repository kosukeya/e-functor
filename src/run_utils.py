from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    runs_root: Path
    run_id: str
    run_dir: Path
    alpha_log: Path
    model_last: Path
    islands_dir: Path
    figures_dir: Path
    derived_dir: Path
    collapse_dir: Path
    config_path: Path
    meta_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(
    run_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[Path, str]:
    root = repo_root()
    if run_dir:
        p = Path(run_dir)
        if not p.is_absolute():
            p = root / p
        rid = run_id or p.name
        return p, rid
    rid = run_id or _default_run_id()
    return root / "runs" / rid, rid


def build_run_paths(run_dir: Path, run_id: str) -> RunPaths:
    root = repo_root()
    runs_root = root / "runs"
    return RunPaths(
        repo_root=root,
        runs_root=runs_root,
        run_id=run_id,
        run_dir=run_dir,
        alpha_log=run_dir / "alpha_log.csv",
        model_last=run_dir / "model_last.pt",
        islands_dir=run_dir / "islands",
        figures_dir=run_dir / "figures",
        derived_dir=run_dir / "derived",
        collapse_dir=run_dir / "collapse",
        config_path=run_dir / "config.json",
        meta_path=run_dir / "metadata.json",
    )


def ensure_run_layout(paths: RunPaths) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.islands_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.derived_dir.mkdir(parents=True, exist_ok=True)
    paths.collapse_dir.mkdir(parents=True, exist_ok=True)
