# island_env_error.py
# Island-wise env distribution (plant/sun/water) + error distribution, per (epoch x island)

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def _get_any(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _parse_epoch_from_name(name: str) -> int:
    m = re.search(r"epoch(\d+)", name)
    return int(m.group(1)) if m else -1


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_snapshot(pt_path: Path) -> dict:
    snap = torch.load(pt_path, map_location="cpu")

    epoch = _get_any(snap, ["epoch", "step", "it"], None)
    if epoch is None:
        epoch = _parse_epoch_from_name(pt_path.name)

    env = _get_any(snap, ["env"], None)  # expected (N,3) for (plant,sun,water)
    y_true = _get_any(snap, ["y_true", "y"], None)
    y_mix = _get_any(snap, ["y_mix"], None)

    I = _get_any(snap, ["I", "I_embed", "I_emb", "I_embedding"], None)
    attn = _get_any(snap, ["attn_growth_row", "attn", "attn_mean"], None)

    env = _to_numpy(env)
    y_true = _to_numpy(y_true)
    y_mix = _to_numpy(y_mix)
    I = _to_numpy(I)
    attn = _to_numpy(attn)

    if env is None or y_true is None or y_mix is None:
        raise KeyError(f"snapshot missing env/y_true/y_mix: {pt_path}")

    # Ensure shapes:
    # env: (N,3)
    if env.ndim != 2 or env.shape[1] != 3:
        raise ValueError(f"env should be (N,3) but got {env.shape} in {pt_path}")

    # y_*: (N,) or (N,1)
    y_true = y_true.reshape(-1)
    y_mix = y_mix.reshape(-1)

    if len(y_true) != env.shape[0] or len(y_mix) != env.shape[0]:
        raise ValueError(f"N mismatch: env {env.shape[0]}, y_true {len(y_true)}, y_mix {len(y_mix)} in {pt_path}")

    return {
        "epoch": int(epoch),
        "env": env,
        "y_true": y_true,
        "y_mix": y_mix,
        "I": I,
        "attn": attn,
        "raw": snap,
        "path": str(pt_path),
    }


def _make_features(I, attn, use_attn: bool) -> np.ndarray:
    if I is None:
        raise KeyError("I missing; cannot cluster islands")

    # I could be (N,3,D) -> flatten to (N, 3*D)
    if I.ndim == 3:
        X = I.reshape(I.shape[0], -1)
    elif I.ndim == 2:
        X = I
    else:
        raise ValueError(f"Unexpected I shape: {I.shape}")

    if use_attn:
        if attn is None:
            raise KeyError("use_attn=True but attn_growth_row missing")
        if attn.ndim != 2:
            raise ValueError(f"attn should be (N,7) but got {attn.shape}")
        if attn.shape[0] != X.shape[0]:
            raise ValueError(f"N mismatch: I {X.shape[0]}, attn {attn.shape[0]}")
        X = np.concatenate([X, attn], axis=1)

    return X.astype(np.float32)


def _kmeans_labels(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    lab = km.fit_predict(X)
    return lab.astype(int)


def _align_labels_by_centroids(prev_X, prev_lab, cur_X, cur_lab):
    """
    Align current cluster ids to previous using centroid distance.
    Works best for small K. For K=2 it checks both swaps.
    """
    prev_ids = np.unique(prev_lab)
    cur_ids = np.unique(cur_lab)

    prev_c = np.stack([prev_X[prev_lab == i].mean(axis=0) for i in prev_ids], axis=0)
    cur_c  = np.stack([cur_X[cur_lab  == i].mean(axis=0) for i in cur_ids], axis=0)

    D = ((prev_c[:, None, :] - cur_c[None, :, :]) ** 2).sum(axis=2)

    if D.shape == (2, 2):
        if D[0, 0] + D[1, 1] <= D[0, 1] + D[1, 0]:
            mapping = {int(cur_ids[0]): int(prev_ids[0]), int(cur_ids[1]): int(prev_ids[1])}
        else:
            mapping = {int(cur_ids[0]): int(prev_ids[1]), int(cur_ids[1]): int(prev_ids[0])}
    else:
        # greedy assignment (ok for modest K)
        mapping = {}
        used = set()
        for i, pid in enumerate(prev_ids):
            j = int(np.argmin(D[i]))
            while j in used:
                D[i, j] = np.inf
                j = int(np.argmin(D[i]))
            mapping[int(cur_ids[j])] = int(pid)
            used.add(j)

    cur_aligned = np.array([mapping[int(c)] for c in cur_lab], dtype=int)
    return cur_aligned


def _stats_1d(x: np.ndarray, prefix: str) -> dict:
    x = x.astype(np.float64)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p25": float(np.quantile(x, 0.25)),
        f"{prefix}_p50": float(np.quantile(x, 0.50)),
        f"{prefix}_p75": float(np.quantile(x, 0.75)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
        f"{prefix}_max": float(np.max(x)),
    }


def _plot_timeseries(df: pd.DataFrame, out_dir: Path, k: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_metric(metric, title, ylabel):
        plt.figure()
        for isl in range(k):
            sub = df[df["island"] == isl].sort_values("epoch")
            plt.plot(sub["epoch"].values, sub[metric].values, marker="o", label=f"island_{isl}")
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_timeseries.png", dpi=160)
        plt.close()

    # env means
    plot_metric("plant_mean", "Island env mean: plant", "plant_mean")
    plot_metric("sun_mean",   "Island env mean: sun",   "sun_mean")
    plot_metric("water_mean", "Island env mean: water", "water_mean")

    # env std
    plot_metric("plant_std", "Island env std: plant", "plant_std")
    plot_metric("sun_std",   "Island env std: sun",   "sun_std")
    plot_metric("water_std", "Island env std: water", "water_std")

    # errors
    plot_metric("err_abs_mean", "Island mean absolute error |y_mix - y_true|", "MAE")
    plot_metric("err_signed_mean", "Island signed error mean (y_mix - y_true)", "signed error")
    plot_metric("err_abs_p90", "Island error abs p90", "abs error p90")


def run(
    runs_dir="runs",
    pattern="islands/island_epoch*.pt",
    k=2,
    use_attn=True,
    seed=0,
    out_csv="island_env_error.csv",
    out_plots_dir="island_env_error_plots",
):
    runs_dir = Path(runs_dir)
    paths = sorted(runs_dir.glob(pattern))
    if len(paths) == 0:
        raise RuntimeError(f"no snapshots found: {runs_dir}/{pattern}")

    snaps = [_load_snapshot(p) for p in paths]
    snaps.sort(key=lambda d: d["epoch"])

    rows = []

    prev_X = None
    prev_lab = None

    for snap in snaps:
        env = snap["env"]
        y_true = snap["y_true"]
        y_mix = snap["y_mix"]

        # features -> cluster
        X = _make_features(snap["I"], snap["attn"], use_attn=use_attn)
        lab = _kmeans_labels(X, k=k, seed=seed)

        # align label ids across epochs
        if prev_X is not None and prev_lab is not None:
            lab = _align_labels_by_centroids(prev_X, prev_lab, X, lab)

        # compute per-island stats
        for isl in range(k):
            m = (lab == isl)
            n = int(m.sum())
            if n == 0:
                continue

            plant = env[m, 0]
            sun = env[m, 1]
            water = env[m, 2]

            err_signed = (y_mix[m] - y_true[m])
            err_abs = np.abs(err_signed)

            row = {
                "epoch": int(snap["epoch"]),
                "island": int(isl),
                "n": n,
            }
            row.update(_stats_1d(plant, "plant"))
            row.update(_stats_1d(sun, "sun"))
            row.update(_stats_1d(water, "water"))

            row.update(_stats_1d(err_abs, "err_abs"))
            row.update(_stats_1d(err_signed, "err_signed"))

            rows.append(row)

        prev_X = X
        prev_lab = lab

    df = pd.DataFrame(rows).sort_values(["epoch", "island"]).reset_index(drop=True)

    out_path = runs_dir / out_csv
    df.to_csv(out_path, index=False)
    print("saved:", out_path)

    plots_dir = runs_dir / out_plots_dir
    _plot_timeseries(df, plots_dir, k=k)
    print("saved plots to:", plots_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--pattern", type=str, default="islands/island_epoch*.pt")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--use_attn", type=int, default=1)  # 1/0
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="island_env_error.csv")
    ap.add_argument("--out_plots_dir", type=str, default="island_env_error_plots")
    args = ap.parse_args()

    run(
        runs_dir=args.runs_dir,
        pattern=args.pattern,
        k=args.k,
        use_attn=bool(args.use_attn),
        seed=args.seed,
        out_csv=args.out_csv,
        out_plots_dir=args.out_plots_dir,
    )


if __name__ == "__main__":
    main()
