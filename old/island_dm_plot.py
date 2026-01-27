# island_dm_plot.py
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
    ap.add_argument("--out-name", type=str, default="island_dM_components_timeseries.png")
    ap.add_argument("--islands", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--comps", type=str, nargs="*", default=["d_cf", "d_att", "d_self"])
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

    df = pd.read_csv(in_path).sort_values("epoch").reset_index(drop=True)
    x = _col(df, "epoch")

    # 3つ並べて見たいのでサブプロット（縦）にする
    fig, axes = plt.subplots(len(args.comps), 1, figsize=(9, 3.2 * len(args.comps)), sharex=True)
    if len(args.comps) == 1:
        axes = [axes]

    for ax, comp in zip(axes, args.comps):
        # all
        ax.plot(x, _col(df, f"{comp}_all"), marker="o", label=f"{comp}_all")

        # islands
        for k in args.islands:
            col = f"{comp}_{k}"
            if col in df.columns:
                ax.plot(x, df[col], marker="o", label=col)
            else:
                print(f"[warn] missing {col}; skipped")

        ax.axhline(0.0, linestyle="--")
        ax.set_title(f"{comp} time series")
        ax.set_ylabel(comp)
        ax.legend()

    axes[-1].set_xlabel("epoch")
    fig.suptitle("Island dM component time series", y=1.02)
    fig.tight_layout()

    out_path = out_dir / args.out_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved:", out_path)

if __name__ == "__main__":
    main()
