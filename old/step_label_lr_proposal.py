# step_label_lr_proposal.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root


def load_events(event_ts_path: Path) -> np.ndarray:
    ts = pd.read_csv(event_ts_path)
    if "is_event" not in ts.columns:
        raise ValueError(f"{event_ts_path} must contain 'is_event' column")
    event_epochs = ts.loc[ts["is_event"].astype(bool), "epoch"].astype(int).values
    return np.unique(event_epochs)


def window_has_event(event_epochs: np.ndarray, s: int, e: int) -> bool:
    i = np.searchsorted(event_epochs, s, side="left")
    return (i < len(event_epochs)) and (event_epochs[i] <= e)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--aligned", type=str, default=None)
    ap.add_argument("--events-ts", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--label", type=str, default="majority")
    ap.add_argument("--min-agree", type=float, default=0.8)
    ap.add_argument("--lr-base", type=float, default=1.0)
    ap.add_argument("--lr-gain", type=float, default=1.5)
    ap.add_argument("--lr-min", type=float, default=0.5)
    ap.add_argument("--lr-max", type=float, default=2.0)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    aligned_path = Path(args.aligned) if args.aligned else (paths.derived_dir / "stepH_semantic_alignment" / "aligned_cluster_by_window_with_agreement.csv")
    event_ts_path = Path(args.events_ts) if args.events_ts else (paths.derived_dir / "threshold_timeseries_with_events.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "label_lr")

    if not aligned_path.is_absolute():
        aligned_path = repo_root() / aligned_path
    if not event_ts_path.is_absolute():
        event_ts_path = repo_root() / event_ts_path
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_csv(aligned_path)
    if args.label not in aligned.columns:
        raise ValueError(f"{aligned_path} must contain column '{args.label}'")
    if "agree_frac" not in aligned.columns:
        raise ValueError(f"{aligned_path} must contain column 'agree_frac'")

    aligned_f = aligned.loc[aligned["agree_frac"] >= args.min_agree].copy()
    if len(aligned_f) == 0:
        raise ValueError("No windows left after agree_frac filtering. Lower --min-agree.")

    event_epochs = load_events(event_ts_path)

    ev_flags = []
    for _, r in aligned_f.iterrows():
        s, e = int(r["epoch_start"]), int(r["epoch_end"])
        ev_flags.append(window_has_event(event_epochs, s, e))
    aligned_f["is_event_window"] = ev_flags
    aligned_f["cluster"] = aligned_f[args.label].astype(int)

    grp = (
        aligned_f
        .groupby("cluster", as_index=False)
        .agg(
            n_windows=("cluster", "size"),
            event_windows=("is_event_window", "sum"),
            event_rate=("is_event_window", "mean"),
            mean_agree=("agree_frac", "mean"),
        )
    )

    global_rate = aligned_f["is_event_window"].mean()

    def propose(rate: float) -> float:
        mult = args.lr_base * (1.0 + args.lr_gain * (rate - global_rate))
        return float(np.clip(mult, args.lr_min, args.lr_max))

    grp["lr_mult"] = grp["event_rate"].apply(propose)

    w = aligned_f.merge(grp[["cluster", "lr_mult"]], on="cluster", how="left")
    w_out = w[["epoch_start", "epoch_end", "cluster", "agree_frac", "is_event_window", "lr_mult"]].copy()

    grp.to_csv(out_dir / "lr_mult_by_cluster.csv", index=False)
    w_out.to_csv(out_dir / "lr_schedule_by_window.csv", index=False)

    print("[saved]")
    print(" -", out_dir / "lr_mult_by_cluster.csv")
    print(" -", out_dir / "lr_schedule_by_window.csv")
    print("\n[summary]")
    print("global_event_rate:", global_rate)
    print(grp.sort_values("cluster"))


if __name__ == "__main__":
    main()
