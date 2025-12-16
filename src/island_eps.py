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
    run_island_eps(
        runs_dir="runs",
        pattern="islands/island_epoch*.pt",  # ★ここだけ変更
    )

