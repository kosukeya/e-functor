# stepE_threshold_modes.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_utils import resolve_run_dir, build_run_paths, repo_root


def build_timeseries_from_events(events_df: pd.DataFrame, eps_event: float) -> pd.DataFrame:
    ev = events_df.sort_values("epoch").reset_index(drop=True).copy()
    first_prev = float(ev.loc[0, "threshold_prev"])
    rows = [{"epoch": 0, "threshold": first_prev}]
    for _, r in ev.iterrows():
        rows.append({"epoch": int(r["epoch"]), "threshold": float(r["threshold"])})
    ts = pd.DataFrame(rows).drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)

    ts["threshold_prev"] = ts["threshold"].shift(1)
    ts["threshold_diff"] = ts["threshold"] - ts["threshold_prev"]
    ts["abs_threshold_diff"] = ts["threshold_diff"].abs()
    ts["is_event"] = ts["abs_threshold_diff"] > eps_event
    ts["event_sign"] = np.sign(ts["threshold_diff"]).fillna(0).astype(int)
    return ts


def discretize_modes_by_events(ts: pd.DataFrame, eps_event: float):
    df = ts.sort_values("epoch").reset_index(drop=True).copy()
    if "threshold_diff" not in df.columns:
        df["threshold_prev"] = df["threshold"].shift(1)
        df["threshold_diff"] = df["threshold"] - df["threshold_prev"]
    if "abs_threshold_diff" not in df.columns:
        df["abs_threshold_diff"] = df["threshold_diff"].abs()
    if "is_event" not in df.columns:
        df["is_event"] = df["abs_threshold_diff"] > eps_event

    mode_id = []
    cur = 0
    for i, is_ev in enumerate(df["is_event"].fillna(False).tolist()):
        if i != 0 and bool(is_ev):
            cur += 1
        mode_id.append(cur)
    df["mode_id"] = mode_id

    seg = (
        df.groupby("mode_id")
        .agg(
            epoch_start=("epoch", "min"),
            epoch_end=("epoch", "max"),
            n_points=("epoch", "count"),
            threshold_level=("threshold", "median"),
            threshold_min=("threshold", "min"),
            threshold_max=("threshold", "max"),
        )
        .reset_index()
    )
    return df, seg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--timeseries", type=str, default=None)
    ap.add_argument("--events", type=str, default=None)
    ap.add_argument("--eps-event", type=float, default=0.01)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--fig-dir", type=str, default=None)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    ts_path = Path(args.timeseries) if args.timeseries else (paths.derived_dir / "threshold_timeseries_with_events.csv")
    events_path = Path(args.events) if args.events else (paths.derived_dir / "threshold_update_events.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "stepE_modes")
    fig_dir = Path(args.fig_dir) if args.fig_dir else (paths.figures_dir / "stepE_modes")

    if not ts_path.is_absolute():
        ts_path = repo_root() / ts_path
    if not events_path.is_absolute():
        events_path = repo_root() / events_path
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    if not fig_dir.is_absolute():
        fig_dir = repo_root() / fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        if "threshold" not in ts.columns and "threshold_value" in ts.columns:
            ts = ts.rename(columns={"threshold_value": "threshold"})
    else:
        if not events_path.exists():
            raise FileNotFoundError(f"Neither {ts_path} nor {events_path} found.")
        events = pd.read_csv(events_path)
        ts = build_timeseries_from_events(events, args.eps_event)

    if "threshold" not in ts.columns:
        raise ValueError("timeseries CSV must contain 'threshold' column.")

    ts_mode, mode_segments = discretize_modes_by_events(ts, args.eps_event)

    out_ts = out_dir / "stepE_threshold_modes.csv"
    out_seg = out_dir / "stepE_threshold_mode_segments.csv"
    ts_mode.to_csv(out_ts, index=False)
    mode_segments.to_csv(out_seg, index=False)
    print("[saved]", out_ts)
    print("[saved]", out_seg)

    plt.figure(figsize=(12, 4.5))
    for m, g in ts_mode.groupby("mode_id"):
        plt.plot(g["epoch"], g["threshold"], marker="o", label=f"mode {m}")

    ev_epochs = ts_mode.loc[ts_mode["is_event"].fillna(False), "epoch"].tolist()
    for e in ev_epochs:
        plt.axvline(e, linestyle="--", alpha=0.4)

    plt.title(f"threshold timeseries with discrete modes (event if |diff|>{args.eps_event})")
    plt.xlabel("epoch")
    plt.ylabel("threshold")
    plt.legend()
    plt.tight_layout()

    out_png = fig_dir / "stepE_threshold_modes.png"
    plt.savefig(out_png, dpi=150)
    print("[saved]", out_png)

    print("\n[mode segments]")
    print(mode_segments)


if __name__ == "__main__":
    main()
