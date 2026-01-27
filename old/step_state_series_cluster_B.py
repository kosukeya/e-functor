# step_state_series_cluster_B.py
# Solution B (recommended): cluster "state trajectories" built from epoch-level state vectors
#
# Inputs (same directory by default):
#   - threshold_timeseries_with_events.csv  (epoch-level; NO island column)
#   - island_env_error.csv                  (epoch x island)
#   - island_profile.csv                    (epoch x island)
#
# Outputs:
#   - stepB_state_series_clustering/
#       - state_epoch_table.csv
#       - trajectories_matrix.csv
#       - trajectory_cluster_assignments.csv
#       - cluster_summary.csv
#       - pca_scatter.png
#       - cluster_centroid_heatmap.png
#
# Usage:
#   python step_state_series_cluster_B.py \
#     --thr threshold_timeseries_with_events.csv \
#     --env island_env_error.csv \
#     --prof island_profile.csv \
#     --out stepB_state_series_clustering \
#     --window 5 --stride 1 --k 3

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root


def _check_columns(df: pd.DataFrame, required: list, name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}\n"
                         f"available columns: {df.columns.tolist()}")


def pivot_epoch_island_to_wide(df: pd.DataFrame, value_cols: list, prefix: str) -> pd.DataFrame:
    """
    df: columns include epoch, island, and value_cols
    -> epoch-level wide table:
       {prefix}{col}_island0, {prefix}{col}_island1, plus diffs (0-1) for each col.
    """
    wide = df.pivot_table(index="epoch", columns="island", values=value_cols, aggfunc="mean")
    # multiindex cols -> flat
    wide.columns = [f"{prefix}{col}_island{isl}" for col, isl in wide.columns]
    wide = wide.reset_index()

    # add diffs where both islands exist
    for col in value_cols:
        c0 = f"{prefix}{col}_island0"
        c1 = f"{prefix}{col}_island1"
        if c0 in wide.columns and c1 in wide.columns:
            wide[f"{prefix}{col}_diff_0_minus_1"] = wide[c0] - wide[c1]
    return wide


def build_epoch_state_table(thr: pd.DataFrame, env: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """
    thr: epoch-level
    env/prof: epoch x island
    returns: epoch-level merged state table
    """
    _check_columns(thr, ["epoch", "threshold"], "thr")
    _check_columns(env, ["epoch", "island"], "env")
    _check_columns(prof, ["epoch", "island"], "prof")

    # pick numeric feature columns (exclude identifiers)
    env_value_cols = [c for c in env.columns if c not in ["epoch", "island", "n"] and pd.api.types.is_numeric_dtype(env[c])]
    prof_value_cols = [c for c in prof.columns if c not in ["epoch", "island", "n"] and pd.api.types.is_numeric_dtype(prof[c])]

    env_w = pivot_epoch_island_to_wide(env, env_value_cols, prefix="env__")
    prof_w = pivot_epoch_island_to_wide(prof, prof_value_cols, prefix="prof__")

    # merge on epoch only (IMPORTANT)
    st = thr.merge(env_w, on="epoch", how="inner").merge(prof_w, on="epoch", how="inner")

    # keep epoch sorted
    st = st.sort_values("epoch").reset_index(drop=True)
    return st


def make_trajectories(state_epoch: pd.DataFrame, feature_cols: list, window: int, stride: int):
    """
    Convert epoch-level state vectors into trajectory vectors by sliding window.
    Each trajectory = concat( state[t-window+1 : t+1] ) or forward window.
    Here we do forward windows: [t, t+1, ..., t+window-1]
    """
    epochs = state_epoch["epoch"].to_numpy()
    X0 = state_epoch[feature_cols].to_numpy(dtype=float)

    traj = []
    meta = []
    n = len(state_epoch)
    if n < window:
        return np.zeros((0, 0)), pd.DataFrame()

    for start in range(0, n - window + 1, stride):
        end = start + window
        block = X0[start:end]  # (window, d)
        if np.any(~np.isfinite(block)):
            continue
        traj.append(block.reshape(-1))  # flatten
        meta.append({
            "traj_id": len(meta),
            "epoch_start": int(epochs[start]),
            "epoch_end": int(epochs[end - 1]),
        })

    if len(traj) == 0:
        return np.zeros((0, 0)), pd.DataFrame(meta)

    X = np.vstack(traj)
    meta_df = pd.DataFrame(meta)
    return X, meta_df


def auto_choose_k(X, k_min=2, k_max=8, random_state=0):
    """
    Choose k by silhouette on a small range.
    """
    if X.shape[0] < 3:
        return 1, None

    best_k = None
    best_s = -1
    scores = []
    for k in range(k_min, min(k_max, X.shape[0] - 1) + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X)
        # silhouette requires >=2 clusters and no singleton-only issues; try/except defensively
        try:
            s = silhouette_score(X, labels)
        except Exception:
            s = np.nan
        scores.append((k, s))
        if np.isfinite(s) and s > best_s:
            best_s = s
            best_k = k
    return best_k if best_k is not None else 2, scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--thr", type=str, default=None)
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--prof", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--k", type=int, default=0, help="0 means auto-choose by silhouette")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    thr_path = Path(args.thr) if args.thr else (paths.derived_dir / "threshold_timeseries_with_events.csv")
    env_path = Path(args.env) if args.env else (paths.derived_dir / "island_env_error.csv")
    prof_path = Path(args.prof) if args.prof else (paths.derived_dir / "island_profile.csv")
    out_dir = Path(args.out) if args.out else (paths.derived_dir / "stepB_state_series_clustering")

    if not thr_path.is_absolute():
        thr_path = repo_root() / thr_path
    if not env_path.is_absolute():
        env_path = repo_root() / env_path
    if not prof_path.is_absolute():
        prof_path = repo_root() / prof_path
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir

    os.makedirs(out_dir, exist_ok=True)

    thr = pd.read_csv(thr_path)
    env = pd.read_csv(env_path)
    prof = pd.read_csv(prof_path)

    # --- build epoch-level table (NO island join for thr) ---
    state_epoch = build_epoch_state_table(thr, env, prof)
    state_epoch.to_csv(os.path.join(out_dir, "state_epoch_table.csv"), index=False)

    # --- feature selection ---
    # drop purely meta columns that are not "state" (keep threshold + (optional) event flags)
    drop_cols = {"feature", "left_rule", "right_rule"}  # if present
    feature_cols = []
    for c in state_epoch.columns:
        if c in ["epoch"]:
            continue
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(state_epoch[c]):
            feature_cols.append(c)

    # If you want to exclude labels/targets for “pure state”, uncomment:
    # for col in ["is_event", "event_sign", "threshold_diff", "abs_threshold_diff"]:
    #     if col in feature_cols: feature_cols.remove(col)

    # --- trajectories ---
    X_traj, meta = make_trajectories(state_epoch, feature_cols, window=args.window, stride=args.stride)

    if X_traj.size == 0 or X_traj.shape[0] == 0:
        raise RuntimeError(
            "No trajectories were created. "
            "Likely causes:\n"
            "  - not enough epochs for the chosen window\n"
            "  - too many NaN/inf after merge\n"
            "Try smaller --window (e.g., 3) or inspect state_epoch_table.csv."
        )

    # scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_traj)

    # choose k
    if args.k and args.k >= 2:
        k = args.k
        sil_scores = None
    else:
        k, sil_scores = auto_choose_k(Xs, k_min=2, k_max=8, random_state=args.seed)

    km = KMeans(n_clusters=k, random_state=args.seed, n_init=50)
    labels = km.fit_predict(Xs)

    # save matrices
    pd.DataFrame(X_traj).to_csv(os.path.join(out_dir, "trajectories_matrix.csv"), index=False)
    meta = meta.copy()
    meta["cluster"] = labels
    meta.to_csv(os.path.join(out_dir, "trajectory_cluster_assignments.csv"), index=False)

    # cluster summary
    summ = (meta.groupby("cluster")
            .agg(n_traj=("traj_id", "count"),
                 epoch_start_min=("epoch_start", "min"),
                 epoch_end_max=("epoch_end", "max"))
            .reset_index()
            .sort_values("cluster"))
    summ.to_csv(os.path.join(out_dir, "cluster_summary.csv"), index=False)

    # --- viz: PCA scatter of trajectories ---
    pca = PCA(n_components=2, random_state=args.seed)
    Z = pca.fit_transform(Xs)

    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels)
    plt.title(f"Trajectory clustering (window={args.window}, k={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=160)
    plt.close()

    # --- viz: centroid heatmap (flattened) ---
    C = km.cluster_centers_  # in scaled space
    plt.figure(figsize=(10, 4))
    plt.imshow(C, aspect="auto")
    plt.title("Cluster centroids (scaled, flattened trajectory features)")
    plt.xlabel("flattened feature index")
    plt.ylabel("cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_centroid_heatmap.png"), dpi=160)
    plt.close()

    # optional: silhouette scores dump
    if sil_scores is not None:
        pd.DataFrame(sil_scores, columns=["k", "silhouette"]).to_csv(
            os.path.join(out_dir, "silhouette_scores.csv"), index=False
        )

    print(f"[ok] saved to: {out_dir}")
    print(f"  epochs in state table: {state_epoch.shape[0]}")
    print(f"  trajectories: {X_traj.shape[0]}  dim: {X_traj.shape[1]}")
    print(f"  chosen k: {k}")


if __name__ == "__main__":
    main()
