# plot_alpha.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None, help="Path to alpha_log.csv (override)")
    ap.add_argument("--out-dir", type=str, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    log_path = Path(args.csv) if args.csv else paths.alpha_log
    out_dir = Path(args.out_dir) if args.out_dir else (paths.figures_dir / "viz_basic")
    out_dir.mkdir(exist_ok=True, parents=True)

    try:
        df = pd.read_csv(log_path, engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(log_path, engine="python", error_bad_lines=False, warn_bad_lines=True)

    def _to_numeric_compat(series):
        try:
            return pd.to_numeric(series, errors="ignore")
        except ValueError:
            coerced = pd.to_numeric(series, errors="coerce")
            return coerced.where(~coerced.isna(), series)

    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = _to_numeric_compat(df[c])

    def pick_col(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    def plot_if_exists(ycol, fname, title, ylabel, extra=None, ylim=None):
        if "epoch" not in df.columns or ycol is None or ycol not in df.columns:
            print(f"[skip] missing columns: epoch or {ycol}")
            return
        x = pd.to_numeric(df["epoch"], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() == 0:
            print(f"[skip] no numeric data for: {ycol}")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(x[m], y[m], marker="o")
        if extra:
            extra()
        if ylim is not None:
            plt.ylim(*ylim)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()
        print("saved:", out_dir / fname)

    alpha_col = pick_col("alpha_used", "alpha")
    plot_if_exists(alpha_col, "alpha.png", "alpha (gate) over epochs", "alpha", ylim=(-0.05, 1.05))

    plot_if_exists(
        "epsilon",
        "epsilon.png",
        "epsilon over epochs",
        "epsilon",
        extra=lambda: plt.axhline(0.0, linestyle="--"),
    )

    if "epoch" in df.columns:
        x = pd.to_numeric(df["epoch"], errors="coerce")
    else:
        x = None

    def plot_two_if_exists(y1, y2, fname, title):
        if x is None or y1 not in df.columns or y2 not in df.columns:
            print(f"[skip] missing columns for {fname}: epoch/{y1}/{y2}")
            return
        a = pd.to_numeric(df[y1], errors="coerce")
        b = pd.to_numeric(df[y2], errors="coerce")
        m1 = x.notna() & a.notna()
        m2 = x.notna() & b.notna()
        if m1.sum() == 0 and m2.sum() == 0:
            print(f"[skip] no numeric data for {fname}")
            return

        plt.figure(figsize=(8, 4))
        if m1.sum() > 0:
            plt.plot(x[m1], a[m1], marker="o", label=y1)
        if m2.sum() > 0:
            plt.plot(x[m2], b[m2], marker="o", label=y2)
        plt.title(title)
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()
        print("saved:", out_dir / fname)

    plot_two_if_exists("dM", "dC", "dM_dC.png", "dM and dC over epochs")

    components = [c for c in ["d_cf", "d_mono", "d_att", "d_self"] if c in df.columns]
    if x is None or len(components) == 0:
        print("[skip] dM_components.png: missing epoch or all component columns")
    else:
        plt.figure(figsize=(8, 4))
        any_plotted = False
        for c in components:
            y = pd.to_numeric(df[c], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() > 0:
                plt.plot(x[m], y[m], marker="o", label=c)
                any_plotted = True
        if any_plotted:
            plt.title("dM components over epochs")
            plt.xlabel("epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "dM_components.png", dpi=150)
            print("saved:", out_dir / "dM_components.png")
        else:
            print("[skip] dM_components.png: no numeric data")
        plt.close()


if __name__ == "__main__":
    main()
