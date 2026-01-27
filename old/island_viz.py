# island_viz.py
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths


# --------------------------
# Utilities
# --------------------------
def load_snapshot(pt_path: Path) -> dict:
    d = torch.load(pt_path, map_location="cpu")
    # sanity checks
    for k in ["epoch", "I", "attn_growth_row", "y_mix", "y_sun0_mix", "y_plant0_mix"]:
        if k not in d:
            raise KeyError(f"{pt_path} missing key: {k}")
    return d


def make_features(d: dict, use_attn: bool = True) -> np.ndarray:
    """
    Feature = [flatten(I: N x 3 x 32) , attn: N x 7]
    """
    I = d["I"]
    if isinstance(I, torch.Tensor):
        I = I.numpy()
    I_flat = I.reshape(I.shape[0], -1)  # N x (3*32)

    if not use_attn:
        return I_flat

    attn = d["attn_growth_row"]
    if isinstance(attn, torch.Tensor):
        attn = attn.numpy()

    return np.concatenate([I_flat, attn], axis=1)


def auto_k_by_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 8, random_state: int = 0) -> int:
    """
    Choose K by silhouette. If it fails (e.g., all points identical), fallback to k_min.
    """
    best_k = k_min
    best_s = -1.0
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(X)
            # silhouette requires >1 cluster and not all same label
            if len(set(labels)) < 2:
                continue
            s = silhouette_score(X, labels)
            if s > best_s:
                best_s = s
                best_k = k
        except Exception:
            continue
    return best_k


def align_labels(prev_centers: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Align cluster labels between epochs by nearest-center matching (greedy).
    Returns a mapping array new_label -> aligned_label.
    (Simple & small K: greedy is fine; if you want strict optimum, use Hungarian.)
    """
    K_prev = prev_centers.shape[0]
    K = centers.shape[0]
    K_use = min(K_prev, K)

    # distance matrix
    D = np.linalg.norm(prev_centers[:, None, :] - centers[None, :, :], axis=-1)  # K_prev x K
    mapping = -np.ones(K, dtype=int)

    used_prev = set()
    used_new = set()

    # greedy smallest distances
    flat = [(D[i, j], i, j) for i in range(K_prev) for j in range(K)]
    flat.sort(key=lambda x: x[0])

    aligned_id = 0
    for _, i, j in flat:
        if i in used_prev or j in used_new:
            continue
        mapping[j] = i  # map new j -> prev i
        used_prev.add(i)
        used_new.add(j)
        aligned_id += 1
        if aligned_id >= K_use:
            break

    # any unmatched new clusters get new ids after prev max
    next_id = K_prev
    for j in range(K):
        if mapping[j] == -1:
            mapping[j] = next_id
            next_id += 1
    return mapping


def compute_dwell_times(label_matrix: np.ndarray, epochs: np.ndarray) -> pd.DataFrame:
    """
    label_matrix: [T, N] aligned labels over time
    epochs: [T]
    Returns per-sample dwell metrics.
    """
    T, N = label_matrix.shape
    dwell_max = np.zeros(N, dtype=int)
    dwell_mean = np.zeros(N, dtype=float)

    for n in range(N):
        runs = []
        cur = label_matrix[0, n]
        length = 1
        for t in range(1, T):
            if label_matrix[t, n] == cur:
                length += 1
            else:
                runs.append(length)
                cur = label_matrix[t, n]
                length = 1
        runs.append(length)
        dwell_max[n] = max(runs)
        dwell_mean[n] = float(np.mean(runs))

    return pd.DataFrame({
        "sample": np.arange(N),
        "dwell_max_steps": dwell_max,
        "dwell_mean_steps": dwell_mean,
    })


def safe_numpy(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# --------------------------
# Main
# --------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--island-dir", type=str, default=None,
                    help="Override islands directory (defaults to <run_dir>/islands)")
    ap.add_argument("--pt-paths", type=str, nargs="*", default=None,
                    help="List of snapshot .pt paths. If omitted, uses --pt-glob in island dir.")
    ap.add_argument("--pt-glob", type=str, default="island_epoch*.pt")
    ap.add_argument("--use-attn", action="store_true", help="Concatenate attn_growth_row to I embedding.")
    ap.add_argument("--k", type=int, default=0, help="KMeans K. If 0, auto by silhouette per-epoch.")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    island_dir = Path(args.island_dir) if args.island_dir else paths.islands_dir
    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "island_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    # gather snapshot paths
    if args.pt_paths and len(args.pt_paths) > 0:
        pt_list = [Path(p) for p in args.pt_paths]
    else:
        pt_list = sorted(island_dir.glob(args.pt_glob))

    if not pt_list:
        raise FileNotFoundError("No snapshot .pt files found.")

    # load all
    snaps = [load_snapshot(p) for p in pt_list]
    epochs = np.array([int(s["epoch"]) for s in snaps], dtype=int)

    # sort by epoch (just in case)
    order = np.argsort(epochs)
    snaps = [snaps[i] for i in order]
    pt_list = [pt_list[i] for i in order]
    epochs = epochs[order]

    # per-epoch clustering + aligned labels
    all_epoch_rows = []
    label_aligned_list = []
    centers_prev = None

    for t, (pt_path, d) in enumerate(zip(pt_list, snaps)):
        X = make_features(d, use_attn=args.use_attn)
        Xs = StandardScaler().fit_transform(X)

        K = args.k if args.k and args.k > 0 else auto_k_by_silhouette(Xs, args.k_min, args.k_max, args.seed)
        km = KMeans(n_clusters=K, n_init="auto", random_state=args.seed)
        labels = km.fit_predict(Xs)
        centers = km.cluster_centers_

        # align labels across epochs (to make dwell times meaningful)
        if centers_prev is None:
            labels_aligned = labels.copy()
            centers_prev = centers.copy()
            label_map = None
        else:
            mapping = align_labels(centers_prev, centers)
            labels_aligned = np.array([mapping[l] for l in labels], dtype=int)
            max_id = labels_aligned.max()
            centers_new = np.zeros((max_id + 1, centers.shape[1]), dtype=float)
            centers_new[:] = np.nan
            for j in range(K):
                centers_new[mapping[j]] = centers[j]
            centers_prev = centers_new

        label_aligned_list.append(labels_aligned)

        y_mix = safe_numpy(d["y_mix"]).reshape(-1)
        y_s0 = safe_numpy(d["y_sun0_mix"]).reshape(-1)
        y_p0 = safe_numpy(d["y_plant0_mix"]).reshape(-1)
        d_sun = y_mix - y_s0
        d_plant = y_mix - y_p0

        for cid in np.unique(labels_aligned):
            idx = (labels_aligned == cid)
            n = int(idx.sum())
            row = {
                "epoch": int(d["epoch"]),
                "alpha_used": float(d.get("alpha_used", np.nan)),
                "island_id": int(cid),
                "n": n,
                "y_mix_mean": float(np.mean(y_mix[idx])),
                "delta_sun_mean": float(np.mean(d_sun[idx])),
                "delta_plant_mean": float(np.mean(d_plant[idx])),
                "delta_sun_std": float(np.std(d_sun[idx])),
                "delta_plant_std": float(np.std(d_plant[idx])),
            }
            all_epoch_rows.append(row)

        # quick visualization (PCA 2D)
        Z = PCA(n_components=2, random_state=args.seed).fit_transform(Xs)
        plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], c=labels_aligned, s=8)
        plt.title(f"Islands (epoch={int(d['epoch'])}, K={K}, use_attn={args.use_attn})")
        plt.tight_layout()
        out_png = out_dir / f"islands_epoch{int(d['epoch']):05d}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

    label_mat = np.stack(label_aligned_list, axis=0)
    dwell_df = compute_dwell_times(label_mat, epochs)
    dwell_df.to_csv(out_dir / "island_dwell.csv", index=False)

    island_df = pd.DataFrame(all_epoch_rows)
    island_df.to_csv(out_dir / "island_summary.csv", index=False)

    island_count = island_df.groupby("epoch")["island_id"].nunique().reset_index(name="n_islands")
    plt.figure(figsize=(7, 4))
    plt.plot(island_count["epoch"], island_count["n_islands"], marker="o")
    plt.title("Number of islands over epochs")
    plt.xlabel("epoch")
    plt.ylabel("#islands")
    plt.tight_layout()
    plt.savefig(out_dir / "island_count.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(dwell_df["dwell_max_steps"], bins=20)
    plt.title("Dwell time (max consecutive steps) histogram")
    plt.xlabel("max dwell steps")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "island_dwell_hist.png", dpi=150)
    plt.close()

    print("Saved:")
    print(" -", out_dir / "island_summary.csv")
    print(" -", out_dir / "island_dwell.csv")
    print(" -", out_dir / "island_count.png")
    print(" -", out_dir / "island_dwell_hist.png")
    print(" -", out_dir / "islands_epochXXXXX.png (per snapshot)")

if __name__ == "__main__":
    main()
