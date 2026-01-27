# stepH_semantic_cluster_alignment.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import repo_root


def _resolve_run_base(spec: str) -> Path:
    p = Path(spec)
    if p.exists():
        if p.is_dir():
            if (p / "trajectories_matrix.csv").exists():
                return p
            if (p / "derived" / "stepB_state_series_clustering").exists():
                return p / "derived" / "stepB_state_series_clustering"
        if p.suffix == ".csv":
            return p.parent
    # treat as run_id
    return repo_root() / "runs" / spec / "derived" / "stepB_state_series_clustering"


def _load_run(base_dir: Path):
    traj_path = base_dir / "trajectories_matrix.csv"
    assign_path = base_dir / "trajectory_cluster_assignments.csv"
    if not traj_path.exists() or not assign_path.exists():
        raise FileNotFoundError(f"Missing trajectories/assignments in {base_dir}")
    traj = pd.read_csv(traj_path)
    assign = pd.read_csv(assign_path)
    X = traj.select_dtypes(include=[np.number]).to_numpy()
    return traj, assign, X


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs="*", default=None,
                    help="Run ids or directories containing stepB_state_series_clustering outputs")
    ap.add_argument("--ref", type=str, default=None, help="Reference run id or directory (defaults to first)")
    ap.add_argument("--dist-metric", type=str, default="cityblock")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args(argv)

    run_specs = args.runs
    if not run_specs:
        run_specs = [p.parent for p in (repo_root() / "runs").glob("*/derived/stepB_state_series_clustering/trajectories_matrix.csv")]
        run_specs = [str(p) for p in run_specs]

    run_bases = [(_resolve_run_base(spec), spec) for spec in run_specs]
    if not run_bases:
        raise RuntimeError("No runs found. Provide --runs ...")

    if args.ref:
        ref_base = _resolve_run_base(args.ref)
    else:
        ref_base = run_bases[0][0]

    out_dir = Path(args.out_dir) if args.out_dir else (ref_base.parent / "stepH_semantic_alignment")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for base_dir, spec in run_bases:
        traj, assign, X = _load_run(base_dir)
        name = base_dir.parent.parent.name if (base_dir.parent.parent).name else base_dir.name
        data[name] = {"traj": traj, "assign": assign, "X": X, "base": base_dir}

    ref_name = None
    for name, d in data.items():
        if d["base"] == ref_base:
            ref_name = name
            break
    if ref_name is None:
        ref_name = list(data.keys())[0]
        ref_base = data[ref_name]["base"]

    ref_assign = data[ref_name]["assign"]
    ref_X = data[ref_name]["X"]
    ref_clusters = ref_assign["cluster"].to_numpy()
    ref_cent = compute_centroids(ref_X, ref_clusters)

    mapping_all = []
    for name, d in data.items():
        if name == ref_name:
            aligned = d["assign"].copy()
            aligned["cluster_aligned"] = aligned["cluster"]
            data[name]["assign_aligned"] = aligned
            continue

        tgt_assign = d["assign"]
        tgt_X = d["X"]
        tgt_clusters = tgt_assign["cluster"].to_numpy()
        tgt_cent = compute_centroids(tgt_X, tgt_clusters)

        mapping, map_df, cost_df = align_to_reference(ref_cent, tgt_cent, metric=args.dist_metric)
        map_df.insert(0, "run", name)
        mapping_all.append(map_df)

        cost_df.to_csv(out_dir / f"{name}_to_{ref_name}_cost_matrix.csv", index=True)

        aligned = tgt_assign.copy()
        aligned["cluster_aligned"] = aligned["cluster"].map(mapping)
        data[name]["assign_aligned"] = aligned
        aligned.to_csv(out_dir / f"{name}_aligned_assignments.csv", index=False)

        plt.figure(figsize=(5, 3))
        plt.imshow(cost_df.values, aspect="auto")
        plt.xticks(range(cost_df.shape[1]), cost_df.columns, rotation=45, ha="right", fontsize=7)
        plt.yticks(range(cost_df.shape[0]), cost_df.index, fontsize=7)
        plt.colorbar(label=f"{args.dist_metric} distance")
        plt.title(f"{name} -> {ref_name} cost ({args.dist_metric})")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_cost_heatmap.png", dpi=160)
        plt.close()

    if mapping_all:
        mapping_df = pd.concat(mapping_all, ignore_index=True).sort_values(["run", "ref_cluster"])
    else:
        mapping_df = pd.DataFrame(columns=["run", "tgt_cluster", "ref_cluster", "distance"])
    mapping_df.to_csv(out_dir / "cluster_alignment_mapping_to_ref.csv", index=False)

    key = ["epoch_start", "epoch_end"]
    merged = data[ref_name]["assign_aligned"][key + ["cluster_aligned"]].rename(columns={"cluster_aligned": ref_name})

    for name, d in data.items():
        if name == ref_name:
            continue
        merged = merged.merge(
            d["assign_aligned"][key + ["cluster_aligned"]].rename(columns={"cluster_aligned": name}),
            on=key,
            how="inner",
        )

    merged.to_csv(out_dir / "aligned_cluster_by_window_across_runs.csv", index=False)

    label_cols = [c for c in merged.columns if c not in key]
    majority = merged[label_cols].mode(axis=1)[0]
    agree_frac = merged[label_cols].eq(majority, axis=0).sum(axis=1) / len(label_cols)

    merged2 = merged.copy()
    merged2["majority"] = majority
    merged2["agree_frac"] = agree_frac
    merged2.to_csv(out_dir / "aligned_cluster_by_window_with_agreement.csv", index=False)

    overall = float(agree_frac.mean())
    per_run = {}
    for name in label_cols:
        if name == ref_name:
            continue
        per_run[f"agree_with_{ref_name}_{name}"] = float((merged[name] == merged[ref_name]).mean())

    metrics = pd.DataFrame([{
        "dist_metric": args.dist_metric,
        "n_windows_compared": int(len(merged)),
        "overall_mean_agree_frac": overall,
        **per_run,
    }])
    metrics.to_csv(out_dir / "cross_run_alignment_metrics.csv", index=False)

    plt.figure(figsize=(8, 3))
    plt.hist(agree_frac, bins=np.linspace(0, 1, 11))
    plt.xlabel("Agreement fraction across runs (per window)")
    plt.ylabel("count")
    plt.title("Cross-run cluster agreement (after semantic alignment)")
    plt.tight_layout()
    plt.savefig(out_dir / "agreement_hist.png", dpi=160)
    plt.close()

    print("[saved]", out_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
