# stepE_event_prepost.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_utils import resolve_run_dir, build_run_paths, repo_root


def pivot_wide(df, value_cols, prefix=""):
    wide = df.pivot_table(index="epoch", columns="island", values=value_cols, aggfunc="mean")
    wide.columns = [f"{prefix}{col}_island{isl}" for col, isl in wide.columns]
    wide = wide.reset_index()

    for col in value_cols:
        c0 = f"{prefix}{col}_island0"
        c1 = f"{prefix}{col}_island1"
        if c0 in wide.columns and c1 in wide.columns:
            wide[f"{prefix}{col}_diff_0_minus_1"] = wide[c0] - wide[c1]
    return wide


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--events", type=str, default=None)
    ap.add_argument("--timeseries", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--fig-dir", type=str, default=None)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    env_path = Path(args.env) if args.env else (paths.derived_dir / "island_env_error.csv")
    prof_path = Path(args.profile) if args.profile else (paths.derived_dir / "island_profile.csv")
    events_path = Path(args.events) if args.events else (paths.derived_dir / "threshold_update_events.csv")
    ts_path = Path(args.timeseries) if args.timeseries else (paths.derived_dir / "threshold_timeseries_with_events.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "stepE_event_prepost")
    fig_dir = Path(args.fig_dir) if args.fig_dir else (paths.figures_dir / "stepE_event_prepost")

    if not env_path.is_absolute():
        env_path = repo_root() / env_path
    if not prof_path.is_absolute():
        prof_path = repo_root() / prof_path
    if not events_path.is_absolute():
        events_path = repo_root() / events_path
    if not ts_path.is_absolute():
        ts_path = repo_root() / ts_path
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    if not fig_dir.is_absolute():
        fig_dir = repo_root() / fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    env = pd.read_csv(env_path)
    prof = pd.read_csv(prof_path)
    events = pd.read_csv(events_path)

    env_value_cols = [c for c in env.columns if c not in ["epoch", "island", "n"]]
    prof_value_cols = [c for c in prof.columns if c not in ["epoch", "island", "n"]]

    env_w = pivot_wide(env, env_value_cols, prefix="env__")
    prof_w = pivot_wide(prof, prof_value_cols, prefix="prof__")
    feat = env_w.merge(prof_w, on="epoch", how="inner")

    if ts_path.exists():
        thr_ts = pd.read_csv(ts_path)
        cols = [c for c in ["epoch", "threshold", "threshold_prev", "threshold_diff", "is_event"] if c in thr_ts.columns]
        feat = feat.merge(thr_ts[cols], on="epoch", how="left")

    epoch_list = sorted(feat["epoch"].dropna().unique().tolist())

    def prev_epoch(e):
        idx = np.searchsorted(epoch_list, e) - 1
        return epoch_list[idx] if idx >= 0 else None

    num_cols = []
    for c in feat.columns:
        if c == "epoch":
            continue
        if pd.api.types.is_bool_dtype(feat[c]):
            continue
        if pd.api.types.is_numeric_dtype(feat[c]):
            num_cols.append(c)

    std = feat[num_cols].std(numeric_only=True).replace(0, np.nan)

    diff_rows = []
    for _, ev in events.iterrows():
        e = int(ev["epoch"])
        pre = prev_epoch(e)
        if pre is None:
            continue

        pre_row = feat.loc[feat["epoch"] == pre, num_cols].iloc[0]
        post_row = feat.loc[feat["epoch"] == e, num_cols].iloc[0]

        diff = post_row - pre_row
        z = diff / std

        tmp = pd.DataFrame({
            "event_epoch": e,
            "pre_epoch": pre,
            "feature": diff.index,
            "pre": pre_row.values,
            "post": post_row.values,
            "diff": diff.values,
            "zscore": z.values,
        })
        diff_rows.append(tmp)

    diff_long = pd.concat(diff_rows, ignore_index=True)
    diff_long["abs_z"] = diff_long["zscore"].abs()
    diff_long.to_csv(out_dir / "event_prepost_diff_long.csv", index=False)

    topk = max(1, int(args.topk))
    tops = []
    for e in sorted(diff_long["event_epoch"].unique()):
        sub = diff_long[diff_long["event_epoch"] == e].sort_values("abs_z", ascending=False).head(topk).copy()
        sub["rank"] = np.arange(1, len(sub) + 1)
        tops.append(sub)
    top_long = pd.concat(tops, ignore_index=True)
    top_long.to_csv(out_dir / "event_prepost_top10_by_absz.csv", index=False)

    for e in sorted(diff_long["event_epoch"].unique()):
        sub = diff_long[diff_long["event_epoch"] == e].sort_values("abs_z", ascending=False).head(12)
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(sub)), sub["zscore"].values)
        plt.xticks(range(len(sub)), sub["feature"].values, rotation=75, ha="right", fontsize=8)
        plt.axhline(0, linewidth=1)
        plt.title(f"Event @ epoch={e}: top changes (z-score) post-pre")
        plt.tight_layout()
        plt.savefig(fig_dir / f"event_{e}_top_changes_zscore.png", dpi=160)
        plt.close()

    agg = (
        diff_long.groupby("feature")
        .agg(
            mean_abs_z=("abs_z", "mean"),
            mean_z=("zscore", "mean"),
            max_abs_z=("abs_z", "max"),
            n_events=("event_epoch", "nunique"),
        )
        .reset_index()
        .sort_values("mean_abs_z", ascending=False)
    )
    agg.to_csv(out_dir / "event_prepost_feature_aggregate.csv", index=False)

    print("Saved to:", out_dir)
    print("\nTop features by mean |z| (across events):")
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
