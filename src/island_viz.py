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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_paths", type=str, nargs="*", default=None,
                    help="List of snapshot .pt paths. If omitted, uses --pt_glob in --runs_dir.")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--pt_glob", type=str, default="island_epoch*.pt")
    ap.add_argument("--use_attn", action="store_true", help="Concatenate attn_growth_row to I embedding.")
    ap.add_argument("--k", type=int, default=0, help="KMeans K. If 0, auto by silhouette per-epoch.")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # gather snapshot paths
    if args.pt_paths and len(args.pt_paths) > 0:
        pt_list = [Path(p) for p in args.pt_paths]
    else:
        pt_list = sorted(Path(args.runs_dir).glob(args.pt_glob))

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
            # update "prev centers" in aligned index space:
            # simplest: keep current centers but reorder to aligned ids
            # (unmatched ones will extend ids; that's OK)
            max_id = labels_aligned.max()
            centers_new = np.zeros((max_id + 1, centers.shape[1]), dtype=float)
            # fill with nan first
            centers_new[:] = np.nan
            for j in range(K):
                centers_new[mapping[j]] = centers[j]
            centers_prev = centers_new

        label_aligned_list.append(labels_aligned)

        # compute cluster stats (island metrics)
        y_mix = safe_numpy(d["y_mix"]).reshape(-1)
        y_s0  = safe_numpy(d["y_sun0_mix"]).reshape(-1)
        y_p0  = safe_numpy(d["y_plant0_mix"]).reshape(-1)
        d_sun   = y_mix - y_s0
        d_plant = y_mix - y_p0

        # per-island summary
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

    # stack labels for dwell
    label_mat = np.stack(label_aligned_list, axis=0)  # T x N
    dwell_df = compute_dwell_times(label_mat, epochs)
    dwell_df.to_csv(out_dir / "island_dwell.csv", index=False)

    # island summary table
    island_df = pd.DataFrame(all_epoch_rows)
    island_df.to_csv(out_dir / "island_summary.csv", index=False)

    # island count over epochs
    island_count = island_df.groupby("epoch")["island_id"].nunique().reset_index(name="n_islands")
    plt.figure(figsize=(7, 4))
    plt.plot(island_count["epoch"], island_count["n_islands"], marker="o")
    plt.title("Number of islands over epochs")
    plt.xlabel("epoch")
    plt.ylabel("#islands")
    plt.tight_layout()
    plt.savefig(out_dir / "island_count.png", dpi=150)
    plt.close()

    # dwell histogram
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

# --- island epsilon / dM decomposition ---
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from model import MultiIWorldModel
import config as C
from metrics import epsilon_between_models

def _get_any(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def _load_snapshot(pt_path: Path):
    snap = torch.load(pt_path, map_location="cpu")

    # try to find epoch
    epoch = _get_any(snap, ["epoch", "step", "it"], None)
    if epoch is None:
        # try parse from filename
        m = re.search(r"epoch(\d+)", pt_path.name)
        epoch = int(m.group(1)) if m else -1

    # model state
    state = _get_any(snap, ["model_state", "state_dict", "model_state_dict"], None)
    if state is None:
        raise KeyError(f"model state not found in {pt_path}")

    # data tensors (expect p,s,w,y on cpu)
    # your snapshot might store packed data or raw tensors
    data = _get_any(snap, ["train_data", "data", "packed", "batch"], None)
    if data is None:
        # try individual keys
        p = _get_any(snap, ["plant", "p"], None)
        s = _get_any(snap, ["sun", "s"], None)
        w = _get_any(snap, ["water", "w"], None)
        y = _get_any(snap, ["growth", "y"], None)
        if any(v is None for v in [p,s,w,y]):
            raise KeyError(f"data not found in {pt_path}")
        data = (p, s, w, y)

    # cluster labels computed by your island_viz earlier?
    # if not stored, you can recompute here from embedding+attn as you already do
    labels = _get_any(snap, ["labels", "cluster", "clusters"], None)

    # embedding / attn for label recompute fallback
    I = _get_any(snap, ["I_embed", "I", "I_emb", "I_embedding"], None)
    attn = _get_any(snap, ["attn", "attn_mean", "attn_avg"], None)

    return {
        "epoch": int(epoch),
        "state": state,
        "data": data,
        "labels": labels,
        "I": I,
        "attn": attn,
        "raw": snap,
    }

def _ensure_labels(snap, k=2, use_attn=True, seed=0):
    """If labels not in snapshot, recompute using (I + attn) like your island plotting."""
    if snap["labels"] is not None:
        lab = snap["labels"]
        if isinstance(lab, torch.Tensor):
            lab = lab.cpu().numpy()
        return lab.astype(int)

    from sklearn.cluster import KMeans

    I = snap["I"]
    if I is None:
        raise KeyError("labels missing and I_embed missing; cannot cluster")

    if isinstance(I, torch.Tensor):
        X = I.detach().cpu().numpy()
    else:
        X = np.asarray(I)

    if use_attn:
        attn = snap["attn"]
        if attn is None:
            raise KeyError("use_attn=True but attn missing")
        if isinstance(attn, torch.Tensor):
            A = attn.detach().cpu().numpy()
        else:
            A = np.asarray(attn)
        X = np.concatenate([X, A], axis=1)

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    lab = km.fit_predict(X)
    return lab.astype(int)

def _match_labels(prev_X, prev_lab, cur_X, cur_lab):
    """
    Align cluster ids between prev and cur by centroid distance.
    Returns a remapped cur_lab so that cluster 0/1 correspond across epochs.
    """
    # centroids
    prev_c = [prev_X[prev_lab == i].mean(axis=0) for i in np.unique(prev_lab)]
    cur_c  = [cur_X[cur_lab  == i].mean(axis=0) for i in np.unique(cur_lab)]
    prev_c = np.stack(prev_c, axis=0)
    cur_c  = np.stack(cur_c, axis=0)

    # distance matrix
    D = ((prev_c[:, None, :] - cur_c[None, :, :])**2).sum(axis=2)
    # for K=2, brute mapping
    if D.shape == (2,2):
        # choose best assignment
        if D[0,0] + D[1,1] <= D[0,1] + D[1,0]:
            mapping = {0:0, 1:1}
        else:
            mapping = {0:1, 1:0}
    else:
        # generic fallback: greedy
        mapping = {}
        used = set()
        for i in range(D.shape[0]):
            j = int(np.argmin(D[i]))
            while j in used:
                D[i, j] = np.inf
                j = int(np.argmin(D[i]))
            mapping[i] = j
            used.add(j)
        # invert to map cur->prev id
        mapping = {v:k for k,v in mapping.items()}

    cur_lab2 = np.array([mapping[int(c)] for c in cur_lab], dtype=int)
    return cur_lab2

@torch.no_grad()
def _eps_info_for_subset(state_f, state_g, data, mask, device):
    """
    Compute epsilon_between_models on subset of samples given boolean mask (N,).
    """
    p, s, w, y = data
    # ensure torch tensors on device
    def to_dev(t):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return torch.tensor(t, device=device)

    p = to_dev(p); s = to_dev(s); w = to_dev(w); y = to_dev(y)

    m = torch.tensor(mask, device=device, dtype=torch.bool)
    p2, s2, w2, y2 = p[m], s[m], w[m], y[m]
    data2 = (p2, s2, w2, y2)

    model_f = MultiIWorldModel(d_model=C.D_MODEL, n_heads=C.N_HEADS).to(device)
    model_g = MultiIWorldModel(d_model=C.D_MODEL, n_heads=C.N_HEADS).to(device)
    model_f.load_state_dict(state_f)
    model_g.load_state_dict(state_g)
    model_f.eval(); model_g.eval()

    info = epsilon_between_models(
        model_f=model_f, model_g=model_g, data=data2, device=device,
        w_cf=1.0, w_mono=1.0, w_att=1.0, w_self=1.0, sym_att=True
    )
    return info

def run_island_eps(
    runs_dir="runs",
    pattern="island_epoch*.pt",
    k=2,
    use_attn=True,
    device=None,
    out_csv="island_eps_summary.csv",
):
    runs_dir = Path(runs_dir)
    paths = sorted(runs_dir.glob(pattern))
    if len(paths) < 2:
        raise RuntimeError("need at least 2 snapshots")

    device = device or (C.device if hasattr(C, "device") else "cpu")

    # load all snaps
    snaps = [_load_snapshot(p) for p in paths]
    snaps.sort(key=lambda x: x["epoch"])

    rows = []
    prev = snaps[0]
    # labels & features for prev
    prev_lab = _ensure_labels(prev, k=k, use_attn=use_attn)
    prev_I = prev["I"].detach().cpu().numpy() if isinstance(prev["I"], torch.Tensor) else np.asarray(prev["I"])
    prev_A = prev["attn"].detach().cpu().numpy() if use_attn and isinstance(prev["attn"], torch.Tensor) else (np.asarray(prev["attn"]) if use_attn else None)
    prev_X = np.concatenate([prev_I, prev_A], axis=1) if use_attn else prev_I

    for cur in snaps[1:]:
        cur_lab = _ensure_labels(cur, k=k, use_attn=use_attn)

        cur_I = cur["I"].detach().cpu().numpy() if isinstance(cur["I"], torch.Tensor) else np.asarray(cur["I"])
        cur_A = cur["attn"].detach().cpu().numpy() if use_attn and isinstance(cur["attn"], torch.Tensor) else (np.asarray(cur["attn"]) if use_attn else None)
        cur_X = np.concatenate([cur_I, cur_A], axis=1) if use_attn else cur_I

        # align current labels to previous labels
        cur_lab_aligned = _match_labels(prev_X, prev_lab, cur_X, cur_lab)

        # compute eps for total + each island using current labels (aligned)
        N = len(cur_lab_aligned)
        all_mask = np.ones(N, dtype=bool)

        info_all = _eps_info_for_subset(cur["state"], prev["state"], cur["data"], all_mask, device=device)
        row = {
            "epoch": cur["epoch"],
            "prev_epoch": prev["epoch"],
            "epsilon_all": info_all["epsilon"],
            "dM_all": info_all["dM"],
            "dC_all": info_all["dC"],
            "d_cf_all": info_all["d_cf"],
            "d_mono_all": info_all["d_mono"],
            "d_att_all": info_all["d_att"],
            "d_self_all": info_all["d_self"],
        }

        for isl in range(k):
            mask = (cur_lab_aligned == isl)
            if mask.sum() < 2:
                # too small -> skip
                row.update({
                    f"n_{isl}": int(mask.sum()),
                    f"epsilon_{isl}": np.nan,
                    f"dM_{isl}": np.nan,
                    f"dC_{isl}": np.nan,
                    f"d_cf_{isl}": np.nan,
                    f"d_mono_{isl}": np.nan,
                    f"d_att_{isl}": np.nan,
                    f"d_self_{isl}": np.nan,
                })
                continue

            info = _eps_info_for_subset(cur["state"], prev["state"], cur["data"], mask, device=device)
            row.update({
                f"n_{isl}": int(mask.sum()),
                f"epsilon_{isl}": info["epsilon"],
                f"dM_{isl}": info["dM"],
                f"dC_{isl}": info["dC"],
                f"d_cf_{isl}": info["d_cf"],
                f"d_mono_{isl}": info["d_mono"],
                f"d_att_{isl}": info["d_att"],
                f"d_self_{isl}": info["d_self"],
            })

        rows.append(row)

        # advance
        prev = cur
        prev_lab = cur_lab_aligned
        prev_X = cur_X

    df = pd.DataFrame(rows)
    out_path = Path(runs_dir) / out_csv
    df.to_csv(out_path, index=False)
    print("saved:", out_path)

if __name__ == "__main__":
    run_island_eps()

