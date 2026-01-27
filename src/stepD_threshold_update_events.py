# stepD_threshold_update_events.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_utils import resolve_run_dir, build_run_paths, repo_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--threshold-csv", type=str, default=None)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--out-ts", type=str, default=None)
    ap.add_argument("--out-events", type=str, default=None)
    ap.add_argument("--out-png", type=str, default=None)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    thr_csv = Path(args.threshold_csv) if args.threshold_csv else (paths.derived_dir / "island_dt_by_epoch_err_abs_mean.csv")
    if not thr_csv.is_absolute():
        thr_csv = repo_root() / thr_csv
    if not thr_csv.exists():
        raise FileNotFoundError(f"threshold CSV not found: {thr_csv}")

    out_ts = Path(args.out_ts) if args.out_ts else (paths.derived_dir / "threshold_timeseries_with_events.csv")
    out_events = Path(args.out_events) if args.out_events else (paths.derived_dir / "threshold_update_events.csv")
    out_png = Path(args.out_png) if args.out_png else (paths.figures_dir / "threshold" / "threshold_update_events.png")
    if not out_ts.is_absolute():
        out_ts = repo_root() / out_ts
    if not out_events.is_absolute():
        out_events = repo_root() / out_events
    if not out_png.is_absolute():
        out_png = repo_root() / out_png
    out_ts.parent.mkdir(parents=True, exist_ok=True)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(thr_csv)
    required_cols = ["epoch", "threshold"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{thr_csv} missing columns: {missing}")

    df = df.sort_values("epoch").reset_index(drop=True)
    df["threshold_prev"] = df["threshold"].shift(1)
    df["threshold_diff"] = df["threshold"] - df["threshold_prev"]
    df["abs_threshold_diff"] = df["threshold_diff"].abs()
    df["is_event"] = df["abs_threshold_diff"] > float(args.eps)
    df["event_sign"] = np.sign(df["threshold_diff"]).fillna(0).astype(int)

    df.to_csv(out_ts, index=False)
    events = df[df["is_event"]].copy()
    events.to_csv(out_events, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(df["epoch"], df["threshold_diff"], marker="o")
    if len(events) > 0:
        plt.scatter(events["epoch"], events["threshold_diff"], s=80)
    plt.axhline(0.0)
    plt.axhline(args.eps, linestyle="--")
    plt.axhline(-args.eps, linestyle="--")
    plt.title(f"threshold_diff timeseries (event if |diff| > {args.eps})")
    plt.xlabel("epoch")
    plt.ylabel("threshold_diff")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] loaded: {thr_csv} (n={len(df)})")
    print(f"[OK] EPS={args.eps}")
    print(f"[OK] events: {len(events)} epochs -> saved: {out_events}")
    print(f"[OK] saved plot: {out_png}")
    print(f"[OK] saved all-with-flags: {out_ts}")


if __name__ == "__main__":
    main()
