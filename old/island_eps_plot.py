# island_eps_plot.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root

def _col(df, name):
    if name not in df.columns:
        raise KeyError(f"missing column: {name} (available={list(df.columns)})")
    return df[name]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-name", type=str, default="island_eps_timeseries.png")
    ap.add_argument("--islands", type=int, nargs="*", default=[0, 1])
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    in_path = Path(args.csv) if args.csv else (paths.derived_dir / "island_eps_summary.csv")
    if not in_path.is_absolute():
        in_path = repo_root() / in_path
    if not in_path.exists():
        raise FileNotFoundError(f"not found: {in_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (paths.figures_dir / "island_eps")
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # basic cleanup
    df = df.sort_values("epoch").reset_index(drop=True)
    x = _col(df, "epoch")

    # plot
    plt.figure(figsize=(9, 4))
    plt.plot(x, _col(df, "epsilon_all"), marker="o", label="epsilon_all")

    for k in args.islands:
        col = f"epsilon_{k}"
        if col in df.columns:
            plt.plot(x, df[col], marker="o", label=col)
        else:
            print(f"[warn] missing {col}; skipped")

    plt.axhline(0.0, linestyle="--")
    plt.title("Island epsilon time series")
    plt.xlabel("epoch")
    plt.ylabel("epsilon")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / args.out_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("saved:", out_path)

if __name__ == "__main__":
    main()
