# stepH_semantic_cluster_alignment.py
# Semantic alignment of trajectory clusters across multiple runs
# - reference = run1
# - align clusters by centroid distance (default: cityblock / L1)
# - evaluate agreement across runs on the same windows (epoch_start, epoch_end)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ----------------------------
# Config
# ----------------------------
RUNS = [1, 2, 3, 4, 5]   # edit if needed
BASE_DIR = "."
OUT_DIR = "stepH_semantic_alignment"
DIST_METRIC = "cityblock"  # recommended here ("euclidean" also ok)

# ----------------------------
# IO helpers
# ----------------------------
def path_for(run: int, kind: str) -> str:
    """
    kind: traj | assign
    run1 uses no prefix, others use run{r}_ prefix.
    """
    if run == 1:
        if kind == "traj":
            return os.path.join(BASE_DIR, "trajectories_matrix.csv")
        if kind == "assign":
            return os.path.join(BASE_DIR, "trajectory_cluster_assignments.csv")
    else:
        if kind == "traj":
            return os.path.join(BASE_DIR, f"run{run}_trajectories_matrix.csv")
        if kind == "assign":
            return os.path.join(BASE_DIR, f"run{run}_trajectory_cluster_assignments.csv")
    raise ValueError(f"unknown kind={kind}")

def load_run(run: int):
    traj = pd.read_csv(path_for(run, "traj"))
    assign = pd.read_csv(path_for(run, "assign"))
    # keep numeric matrix only
    X = traj.select_dtypes(include=[np.number]).to_numpy()
    return traj, assign, X

# ----------------------------
# Core
# ----------------------------
def compute_centroids(X: np.ndarray, clusters: np.ndarray):
    cent = {}
    for k in sorted(np.unique(clusters)):
        cent[k] = X[clusters == k].mean(axis=0)
    return cent

def align_to_reference(ref_centroids: dict, tgt_centroids: dict, metric: str):
    ref_keys = sorted(ref_centroids.keys())
    tgt_keys = sorted(tgt_centroids.keys())

    ref_mat = np.vstack([ref_centroids[k] for k in ref_keys])
    tgt_mat = np.vstack([tgt_centroids[k] for k in tgt_keys])

    cost = cdist(tgt_mat, ref_mat, metric=metric)
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {tgt_keys[i]: ref_keys[j] for i, j in zip(row_ind, col_ind)}
    map_rows = []
    for i, j in zip(row_ind, col_ind):
        map_rows.append({
            "tgt_cluster": tgt_keys[i],
            "ref_cluster": ref_keys[j],
            "distance": float(cost[i, j]),
        })
    cost_df = pd.DataFrame(
        cost,
        index=[f"tgt_c{c}" for c in tgt_keys],
        columns=[f"ref_c{c}" for c in ref_keys],
    )
    return mapping, pd.DataFrame(map_rows), cost_df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # load all runs
    data = {}
    for r in RUNS:
        traj, assign, X = load_run(r)
        data[r] = {"traj": traj, "assign": assign, "X": X}

    # reference
    ref_assign = data[1]["assign"]
    ref_X = data[1]["X"]
    ref_clusters = ref_assign["cluster"].to_numpy()
    ref_cent = compute_centroids(ref_X, ref_clusters)

    # align each run to run1
    mapping_all = []
    for r in RUNS:
        if r == 1:
            # identity
            data[r]["assign_aligned"] = data[r]["assign"].copy()
            data[r]["assign_aligned"]["cluster_aligned"] = data[r]["assign_aligned"]["cluster"]
            continue

        tgt_assign = data[r]["assign"]
        tgt_X = data[r]["X"]
        tgt_clusters = tgt_assign["cluster"].to_numpy()
        tgt_cent = compute_centroids(tgt_X, tgt_clusters)

        mapping, map_df, cost_df = align_to_reference(ref_cent, tgt_cent, metric=DIST_METRIC)
        map_df.insert(0, "run", r)
        mapping_all.append(map_df)

        # save mapping & cost
        cost_df.to_csv(os.path.join(OUT_DIR, f"run{r}_to_run1_cost_matrix.csv"), index=True)

        # apply
        aligned = tgt_assign.copy()
        aligned["cluster_aligned"] = aligned["cluster"].map(mapping)
        data[r]["assign_aligned"] = aligned
        aligned.to_csv(os.path.join(OUT_DIR, f"run{r}_aligned_assignments.csv"), index=False)

        # heatmap
        plt.figure(figsize=(5, 3))
        plt.imshow(cost_df.values, aspect="auto")
        plt.xticks(range(cost_df.shape[1]), cost_df.columns, rotation=45, ha="right", fontsize=7)
        plt.yticks(range(cost_df.shape[0]), cost_df.index, fontsize=7)
        plt.colorbar(label=f"{DIST_METRIC} distance")
        plt.title(f"run{r} -> run1 cost ({DIST_METRIC})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"run{r}_cost_heatmap.png"), dpi=160)
        plt.close()

    if mapping_all:
        mapping_df = pd.concat(mapping_all, ignore_index=True).sort_values(["run", "ref_cluster"])
    else:
        mapping_df = pd.DataFrame(columns=["run", "tgt_cluster", "ref_cluster", "distance"])
    mapping_df.to_csv(os.path.join(OUT_DIR, "cluster_alignment_mapping_to_run1.csv"), index=False)

    # cross-run agreement on the same windows
    key = ["epoch_start", "epoch_end"]
    merged = data[1]["assign_aligned"][key + ["cluster_aligned"]].rename(columns={"cluster_aligned": "run1"})

    for r in RUNS:
        if r == 1:
            continue
        merged = merged.merge(
            data[r]["assign_aligned"][key + ["cluster_aligned"]].rename(columns={"cluster_aligned": f"run{r}"}),
            on=key, how="inner"
        )

    merged.to_csv(os.path.join(OUT_DIR, "aligned_cluster_by_window_across_runs.csv"), index=False)

    label_cols = [c for c in merged.columns if c.startswith("run")]
    majority = merged[label_cols].mode(axis=1)[0]
    agree_frac = merged[label_cols].eq(majority, axis=0).sum(axis=1) / len(label_cols)

    merged2 = merged.copy()
    merged2["majority"] = majority
    merged2["agree_frac"] = agree_frac
    merged2.to_csv(os.path.join(OUT_DIR, "aligned_cluster_by_window_with_agreement.csv"), index=False)

    overall = float(agree_frac.mean())
    per_run = {}
    for r in RUNS:
        if r == 1:
            continue
        per_run[f"agree_with_run1_run{r}"] = float((merged[f"run{r}"] == merged["run1"]).mean())

    metrics = pd.DataFrame([{
        "dist_metric": DIST_METRIC,
        "n_windows_compared": int(len(merged)),
        "overall_mean_agree_frac": overall,
        **per_run
    }])
    metrics.to_csv(os.path.join(OUT_DIR, "cross_run_alignment_metrics.csv"), index=False)

    # agreement histogram
    plt.figure(figsize=(8, 3))
    plt.hist(agree_frac, bins=np.linspace(0, 1, 11))
    plt.xlabel("Agreement fraction across runs (per window)")
    plt.ylabel("count")
    plt.title("Cross-run cluster agreement (after semantic alignment)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "agreement_hist.png"), dpi=160)
    plt.close()

    print("[saved]", OUT_DIR)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()