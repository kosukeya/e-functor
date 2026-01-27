# stepG_cluster_stability.py
import argparse
import glob
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root


def load_run_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "event_epoch" not in df.columns:
        raise ValueError(f"event_epoch column not found in {path}")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"emb_* columns not found in {path}")
    df = df.sort_values("event_epoch").reset_index(drop=True)
    return df


def cluster_labels(X: np.ndarray, method: str, k: int, seed: int = 0) -> np.ndarray:
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        return model.fit_predict(X)
    if method == "agglo":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(X)
    raise ValueError("method must be one of {'kmeans','agglo'}")


def centroid_matrix(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    C = np.zeros((k, X.shape[1]), dtype=float)
    for i in range(k):
        C[i] = X[y == i].mean(axis=0)
    return C


def match_centroids(Ca: np.ndarray, Cb: np.ndarray) -> dict:
    def cos_dist(a, b):
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return 1.0 - (a @ b) / (na * nb)

    k = Ca.shape[0]
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost[i, j] = cos_dist(Ca[i], Cb[j])

    r, c = linear_sum_assignment(cost)
    return {
        "row": r.tolist(),
        "col": c.tolist(),
        "cost": cost[r, c].tolist(),
        "mean_cost": float(np.mean(cost[r, c])),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--pattern", type=str, default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "agglo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    pattern = args.pattern or str(repo_root() / "runs" / "*" / "derived" / "step2_epsilon_event_embedding" / "epsilon_event_embeddings.csv")
    paths_list = sorted(glob.glob(pattern))
    if len(paths_list) < 2:
        raise RuntimeError(f"Need >=2 runs. No files matched: {pattern}")

    out_csv = Path(args.out) if args.out else (paths.derived_dir / "stepG_cluster_stability_pairwise.csv")
    if not out_csv.is_absolute():
        out_csv = repo_root() / out_csv

    runs = []
    for p in paths_list:
        df = load_run_csv(p)
        run_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

        X = df[emb_cols].to_numpy()
        Xs = StandardScaler().fit_transform(X)
        y = cluster_labels(Xs, method=args.method, k=args.k, seed=args.seed)

        runs.append({
            "name": run_name,
            "path": p,
            "df": df,
            "emb_cols": emb_cols,
            "X": X,
            "Xs": Xs,
            "y": y,
        })

    rows = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            ri, rj = runs[i], runs[j]
            ei = ri["df"]["event_epoch"].to_numpy()
            ej = rj["df"]["event_epoch"].to_numpy()
            if not np.array_equal(ei, ej):
                rows.append({
                    "run_a": ri["name"],
                    "run_b": rj["name"],
                    "ARI": np.nan,
                    "NMI": np.nan,
                    "centroid_match_mean_cost": np.nan,
                    "note": "event_epoch mismatch (need alignment)",
                })
                continue

            ari = adjusted_rand_score(ri["y"], rj["y"])
            nmi = normalized_mutual_info_score(ri["y"], rj["y"])

            Ca = centroid_matrix(ri["Xs"], ri["y"], k=args.k)
            Cb = centroid_matrix(rj["Xs"], rj["y"], k=args.k)
            match = match_centroids(Ca, Cb)

            rows.append({
                "run_a": ri["name"],
                "run_b": rj["name"],
                "ARI": float(ari),
                "NMI": float(nmi),
                "centroid_match_mean_cost": match["mean_cost"],
                "note": "",
            })

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    print(out.head(10))


if __name__ == "__main__":
    main()
