# island_dt_by_epoch.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root


def fit_stump_by_epoch(df: pd.DataFrame, feature="err_abs_mean", target="island"):
    rows = []
    for epoch, g in df.groupby("epoch"):
        g = g.dropna(subset=[feature, target]).copy()
        if g[target].nunique() < 2:
            continue

        X = g[[feature]].to_numpy()
        y = g[target].astype(int).to_numpy()

        clf = DecisionTreeClassifier(max_depth=1, random_state=0)
        clf.fit(X, y)

        tree = clf.tree_
        feat_idx = tree.feature[0]
        thr = float(tree.threshold[0]) if feat_idx != -2 else np.nan

        left_node = tree.children_left[0]
        right_node = tree.children_right[0]

        left_counts = tree.value[left_node][0]
        right_counts = tree.value[right_node][0]
        left_class = int(np.argmax(left_counts))
        right_class = int(np.argmax(right_counts))

        yhat = clf.predict(X)
        acc = float(accuracy_score(y, yhat))

        rows.append({
            "epoch": int(epoch),
            "feature": feature,
            "threshold": thr,
            "left_rule": f"{feature} <= {thr:.6g} -> class {left_class}",
            "right_rule": f"{feature} >  {thr:.6g} -> class {right_class}",
            "left_class": left_class,
            "right_class": right_class,
            "accuracy": acc,
            "n_rows": int(len(g)),
            "val_island0": float(g.loc[g[target] == 0, feature].iloc[0]) if (g[target] == 0).any() else np.nan,
            "val_island1": float(g.loc[g[target] == 1, feature].iloc[0]) if (g[target] == 1).any() else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--feature", type=str, default="err_abs_mean")
    ap.add_argument("--target", type=str, default="island")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default="island_dt_by_epoch_err_abs_mean.csv")
    ap.add_argument("--fig-dir", type=str, default=None)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    csv_path = Path(args.csv) if args.csv else (paths.derived_dir / "island_env_error.csv")
    if not csv_path.is_absolute():
        csv_path = repo_root() / csv_path

    df = pd.read_csv(csv_path)
    df["epoch"] = df["epoch"].astype(int)
    df["island"] = df["island"].astype(int)

    out = fit_stump_by_epoch(df, feature=args.feature, target=args.target)

    out_dir = Path(args.out_dir) if args.out_dir else paths.derived_dir
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv_path = out_dir / args.out_csv
    out.to_csv(out_csv_path, index=False)
    print("saved:", out_csv_path)
    print(out[["epoch", "threshold", "left_class", "right_class", "accuracy", "val_island0", "val_island1"]])

    fig_dir = Path(args.fig_dir) if args.fig_dir else (paths.figures_dir / "threshold")
    if not fig_dir.is_absolute():
        fig_dir = repo_root() / fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(out["epoch"], out["threshold"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel(f"DecisionTree threshold ({args.feature})")
    plt.title("Stump threshold by epoch")
    plt.grid(True)
    plt.tight_layout()
    fig_path = fig_dir / "island_dt_threshold_timeseries.png"
    plt.savefig(fig_path, dpi=200)
    print("saved:", fig_path)

    plt.figure()
    plt.plot(out["epoch"], out["val_island0"], marker="o", label="island=0 value")
    plt.plot(out["epoch"], out["val_island1"], marker="o", label="island=1 value")
    plt.xlabel("epoch")
    plt.ylabel(args.feature)
    plt.title(f"{args.feature} by island")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_path = fig_dir / "err_abs_mean_by_island_timeseries.png"
    plt.savefig(fig_path, dpi=200)
    print("saved:", fig_path)


if __name__ == "__main__":
    main()
