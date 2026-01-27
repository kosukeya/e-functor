# stepA_threshold_sources.py
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_utils import resolve_run_dir, build_run_paths, repo_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--dt", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--top-n", type=int, default=6)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    env_path = Path(args.env) if args.env else (paths.derived_dir / "island_env_error.csv")
    prof_path = Path(args.profile) if args.profile else (paths.derived_dir / "island_profile.csv")
    dt_path = Path(args.dt) if args.dt else (paths.derived_dir / "island_dt_by_epoch_err_abs_mean.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "stepA_threshold_sources")

    if not env_path.is_absolute():
        env_path = repo_root() / env_path
    if not prof_path.is_absolute():
        prof_path = repo_root() / prof_path
    if not dt_path.is_absolute():
        dt_path = repo_root() / dt_path
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = pd.read_csv(env_path)
    prof = pd.read_csv(prof_path)
    dt = pd.read_csv(dt_path)

    need_cols = ["epoch", "threshold", "val_island0", "val_island1", "accuracy"]
    missing = [c for c in need_cols if c not in dt.columns]
    if missing:
        raise ValueError(f"threshold CSV missing columns: {missing}")

    dt = dt[need_cols].copy()
    dt["mid_err"] = (dt["val_island0"] + dt["val_island1"]) / 2
    dt["err_gap"] = dt["val_island0"] - dt["val_island1"]

    key_cols = ["epoch", "island", "n"]
    feature_cols = [c for c in prof.columns if c not in key_cols]

    wide0 = prof[prof["island"] == 0].set_index("epoch")[["n"] + feature_cols].add_suffix("_i0")
    wide1 = prof[prof["island"] == 1].set_index("epoch")[["n"] + feature_cols].add_suffix("_i1")

    wide = wide0.join(wide1, how="inner")
    wide["n_total"] = wide["n_i0"] + wide["n_i1"]
    wide["n_ratio_i0"] = wide["n_i0"] / wide["n_total"]
    wide["n_ratio_i1"] = wide["n_i1"] / wide["n_total"]

    def add_diff(col):
        if f"{col}_i0" in wide.columns and f"{col}_i1" in wide.columns:
            wide[f"{col}_diff_i0_minus_i1"] = wide[f"{col}_i0"] - wide[f"{col}_i1"]

    selected_for_diff = [
        "alpha_used_mean",
        "alpha_used_std",
        "y_true_mean",
        "y_mix_mean",
        "delta_sun_mean",
        "delta_plant_mean",
        "ratio_mean",
        "attn_mass_0_2",
        "attn_mass_3_5",
        "attn_mass_6",
        "err_abs_mean",
        "err_abs_p90",
        "err_signed_mean",
    ]
    for col in selected_for_diff:
        add_diff(col)

    df = dt.set_index("epoch").join(wide, how="inner").reset_index()

    exclude_patterns = [
        r"^val_island",
        r"^mid_err$",
        r"^threshold$",
        r"err_abs_mean",
        r"err_abs_p90",
        r"err_signed_mean",
        r"err_gap",
    ]

    def is_excluded(col):
        return any(re.search(pat, col) for pat in exclude_patterns)

    cand_cols = [c for c in df.columns if c not in ["epoch", "accuracy"] and not is_excluded(c)]
    cand_cols = [c for c in cand_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]

    rows = []
    for c in cand_cols:
        corr = np.corrcoef(df[c].astype(float), df["threshold"])[0, 1]
        corr_mid = np.corrcoef(df[c].astype(float), df["mid_err"])[0, 1]
        rows.append((c, corr, corr_mid, df[c].nunique()))

    rank = (
        pd.DataFrame(rows, columns=["feature", "corr_with_threshold", "corr_with_mid_err", "n_unique"])
        .sort_values(by="corr_with_threshold", key=lambda s: s.abs(), ascending=False)
    )

    rank.to_csv(out_dir / "threshold_candidate_correlations.csv", index=False)
    df.to_csv(out_dir / "threshold_with_candidates_wide.csv", index=False)

    plt.figure()
    plt.plot(df["epoch"], df["threshold"], marker="o", label="threshold")
    plt.plot(df["epoch"], df["mid_err"], marker="o", label="mid_err=(err0+err1)/2")
    plt.xlabel("epoch")
    plt.title("Threshold equals midpoint of island errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_equals_mid_err.png", dpi=150)
    plt.close()

    top = rank.head(max(1, args.top_n))["feature"].tolist()
    for feat in top:
        plt.figure()
        plt.plot(df["epoch"], df["threshold"], marker="o", label="threshold")
        plt.plot(df["epoch"], df[feat], marker="o", label=feat)
        plt.xlabel("epoch")
        plt.title(f"Threshold vs candidate: {feat}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"threshold_vs_{feat}.png", dpi=150)
        plt.close()

    print("Saved outputs to:", out_dir)
    print(rank.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
