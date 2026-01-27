# island_profile.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# sklearn is used only for KMeans
from sklearn.cluster import KMeans

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths

# -------------------------
# small utilities
# -------------------------
def _get_any(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _parse_epoch(path: Path, snap: dict) -> int:
    e = _get_any(snap, ["epoch", "step", "it"], None)
    if e is not None:
        return int(e)
    m = re.search(r"epoch(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pearson(a, b, eps=1e-12):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std() * b0.std()) + eps
    return float((a0 * b0).mean() / denom)


def _make_features(I, attn, use_attn=True):
    """
    I: (N, 3, D)  or (N, 3*D) etc.
    attn: (N, 7)
    -> X: (N, F)
    """
    I = _to_numpy(I)
    if I is None:
        raise KeyError("snapshot missing I")

    # flatten if 3D (N,3,D)
    if I.ndim == 3:
        N, a, b = I.shape
        I2 = I.reshape(N, a * b)
    elif I.ndim == 2:
        I2 = I
    else:
        raise ValueError(f"unexpected I shape: {I.shape}")

    if not use_attn:
        return I2

    attn = _to_numpy(attn)
    if attn is None:
        raise KeyError("use_attn=True but snapshot missing attn_growth_row")
    if attn.ndim != 2:
        raise ValueError(f"unexpected attn shape: {attn.shape}")

    # concat along feature axis
    return np.concatenate([I2, attn], axis=1)


def _match_labels(prev_X, prev_lab, cur_X, cur_lab):
    """
    Align cluster ids between prev and cur by centroid distance.
    Returns remapped cur_lab so that label ids correspond over time.
    Works best for K=2 (our case).
    """
    prev_ids = np.unique(prev_lab)
    cur_ids = np.unique(cur_lab)

    prev_c = np.stack([prev_X[prev_lab == i].mean(axis=0) for i in prev_ids], axis=0)
    cur_c = np.stack([cur_X[cur_lab == i].mean(axis=0) for i in cur_ids], axis=0)

    D = ((prev_c[:, None, :] - cur_c[None, :, :]) ** 2).sum(axis=2)

    if D.shape == (2, 2):
        if D[0, 0] + D[1, 1] <= D[0, 1] + D[1, 0]:
            mapping = {int(cur_ids[0]): int(prev_ids[0]), int(cur_ids[1]): int(prev_ids[1])}
        else:
            mapping = {int(cur_ids[0]): int(prev_ids[1]), int(cur_ids[1]): int(prev_ids[0])}
    else:
        # generic greedy fallback
        mapping = {}
        used_prev = set()
        for j, cur_id in enumerate(cur_ids):
            i = int(np.argmin(D[:, j]))
            while i in used_prev:
                D[i, j] = np.inf
                i = int(np.argmin(D[:, j]))
            mapping[int(cur_id)] = int(prev_ids[i])
            used_prev.add(i)

    return np.array([mapping[int(c)] for c in cur_lab], dtype=int)


def _load_snapshot(pt_path: Path):
    snap = torch.load(pt_path, map_location="cpu")
    epoch = _parse_epoch(pt_path, snap)

    # required for profiling
    I = _get_any(snap, ["I", "I_embed", "I_emb", "I_embedding"], None)
    attn = _get_any(snap, ["attn_growth_row", "attn", "attn_mean", "attn_avg"], None)

    y_true = _get_any(snap, ["y_true", "y"], None)
    y_mix  = _get_any(snap, ["y_mix"], None)
    y_stat = _get_any(snap, ["y_stat"], None)
    y_sem  = _get_any(snap, ["y_sem"], None)

    y_sun0 = _get_any(snap, ["y_sun0_mix"], None)
    y_p0   = _get_any(snap, ["y_plant0_mix"], None)

    return {
        "path": pt_path,
        "epoch": int(epoch),
        "I": I,
        "attn": attn,
        "y_true": y_true,
        "y_mix": y_mix,
        "y_stat": y_stat,
        "y_sem": y_sem,
        "y_sun0_mix": y_sun0,
        "y_plant0_mix": y_p0,
        "raw": snap,
    }


# -------------------------
# main
# -------------------------
def run_island_profile(
    island_dir,
    pattern="island_epoch*.pt",
    k=2,
    use_attn=True,
    seed=0,
    max_snapshots=None,
    out_csv="island_profile.csv",
    out_dir=None,
):
    island_dir = Path(island_dir)
    out_dir = Path(out_dir) if out_dir is not None else island_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(island_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"no snapshots found: {island_dir}/{pattern}")

    snaps = [_load_snapshot(p) for p in paths]
    snaps.sort(key=lambda x: x["epoch"])
    if max_snapshots is not None:
        snaps = snaps[: int(max_snapshots)]

    rows = []

    prev_X = None
    prev_lab = None

    for idx, s in enumerate(snaps):
        X = _make_features(s["I"], s["attn"], use_attn=use_attn)

        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        lab = km.fit_predict(X).astype(int)

        if idx > 0:
            lab = _match_labels(prev_X, prev_lab, X, lab)

        # cache for alignment
        prev_X = X
        prev_lab = lab

        # tensors -> numpy
        y_true = _to_numpy(s["y_true"])
        y_mix  = _to_numpy(s["y_mix"])
        y_stat = _to_numpy(s["y_stat"])
        y_sem  = _to_numpy(s["y_sem"])
        y_sun0 = _to_numpy(s["y_sun0_mix"])
        y_p0   = _to_numpy(s["y_plant0_mix"])
        attn   = _to_numpy(s["attn"])

        if y_true is None or y_mix is None:
            raise KeyError(f"{s['path']} missing y_true or y_mix (needed)")

        for isl in range(k):
            m = (lab == isl)
            n = int(m.sum())
            if n == 0:
                continue

            ym = y_mix[m]
            yt = y_true[m]

            # output stats
            r = {
                "epoch": s["epoch"],
                "island": isl,
                "n": n,
                "y_mix_mean": float(ym.mean()),
                "y_mix_std": float(ym.std()),
                "corr_y_mix_true": _pearson(ym, yt),
            }

            if y_stat is not None:
                r["y_stat_mean"] = float(y_stat[m].mean())
            if y_sem is not None:
                r["y_sem_mean"] = float(y_sem[m].mean())

            # counterfactual sensitivity (bigger = that factor mattered)
            if y_sun0 is not None:
                r["delta_sun_mean"] = float((y_mix[m] - y_sun0[m]).mean())
            else:
                r["delta_sun_mean"] = np.nan

            if y_p0 is not None:
                r["delta_plant_mean"] = float((y_mix[m] - y_p0[m]).mean())
            else:
                r["delta_plant_mean"] = np.nan

            # ratio (sun vs plant)
            ds = r["delta_sun_mean"]
            dp = r["delta_plant_mean"]
            if np.isfinite(ds) and np.isfinite(dp):
                r["delta_ratio_sun_over_plant"] = float(ds / (abs(dp) + 1e-8))
            else:
                r["delta_ratio_sun_over_plant"] = np.nan

            # attention mean (7 dims)
            if attn is not None:
                a = attn[m].mean(axis=0)  # (7,)
                for j in range(a.shape[0]):
                    r[f"attn_mean_{j}"] = float(a[j])

                # handy coarse masses (assumes 7 tokens, we just group by index)
                # You can rename later once your token mapping is fixed.
                r["attn_mass_0_2"] = float(a[0:3].sum())   # e.g., env-ish
                r["attn_mass_3_5"] = float(a[3:6].sum())   # e.g., I-ish
                r["attn_mass_6"]   = float(a[6])          # self-ish

            rows.append(r)

    df = pd.DataFrame(rows).sort_values(["epoch", "island"]).reset_index(drop=True)
    csv_path = out_dir / out_csv
    df.to_csv(csv_path, index=False)
    print("saved:", csv_path)

    # -------------
    # plots
    # -------------
    def _plot_timeseries(value_col, title, ylabel, fname, hline0=True):
        plt.figure(figsize=(9, 4))
        for isl in sorted(df["island"].unique()):
            sub = df[df["island"] == isl]
            plt.plot(sub["epoch"], sub[value_col], marker="o", label=f"island_{isl}")
        if hline0:
            plt.axhline(0.0, linestyle="--")
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        out = out_dir / fname
        plt.savefig(out, dpi=150)
        plt.close()
        print("saved:", out)

    # counterfactual sensitivities
    if "delta_sun_mean" in df.columns:
        _plot_timeseries(
            "delta_sun_mean",
            "Island counterfactual sensitivity: sun",
            "mean(y_mix - y_sun0_mix)",
            "island_delta_sun_timeseries.png",
            hline0=True,
        )
    if "delta_plant_mean" in df.columns:
        _plot_timeseries(
            "delta_plant_mean",
            "Island counterfactual sensitivity: plant",
            "mean(y_mix - y_plant0_mix)",
            "island_delta_plant_timeseries.png",
            hline0=True,
        )

    # output stats
    _plot_timeseries(
        "y_mix_mean",
        "Island output mean (y_mix)",
        "mean y_mix",
        "island_y_mix_mean_timeseries.png",
        hline0=False,
    )

    # attention masses (coarse)
    if "attn_mass_0_2" in df.columns:
        _plot_timeseries(
            "attn_mass_0_2",
            "Island attention mass (indices 0-2)",
            "sum(attn[0:3])",
            "island_attn_mass_0_2_timeseries.png",
            hline0=False,
        )
    if "attn_mass_3_5" in df.columns:
        _plot_timeseries(
            "attn_mass_3_5",
            "Island attention mass (indices 3-5)",
            "sum(attn[3:6])",
            "island_attn_mass_3_5_timeseries.png",
            hline0=False,
        )
    if "attn_mass_6" in df.columns:
        _plot_timeseries(
            "attn_mass_6",
            "Island attention mass (index 6)",
            "attn[6]",
            "island_attn_mass_6_timeseries.png",
            hline0=False,
        )




def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--island-dir", type=str, default=None,
                    help="Override islands directory (defaults to <run_dir>/islands)")
    ap.add_argument("--pattern", type=str, default="island_epoch*.pt")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--use-attn", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default="island_profile.csv")
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    island_dir = Path(args.island_dir) if args.island_dir else paths.islands_dir
    out_dir = Path(args.out_dir) if args.out_dir else paths.derived_dir

    run_island_profile(
        island_dir=str(island_dir),
        pattern=args.pattern,
        k=args.k,
        use_attn=bool(args.use_attn),
        seed=args.seed,
        max_snapshots=args.max_snapshots,
        out_csv=args.out_csv,
        out_dir=str(out_dir),
    )
if __name__ == "__main__":
    main()
