# stepG_cluster_stability.py
import glob
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def load_run_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 必須列チェック
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
    elif method == "agglo":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(X)
    else:
        raise ValueError("method must be one of {'kmeans','agglo'}")

def centroid_matrix(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    C = np.zeros((k, X.shape[1]), dtype=float)
    for i in range(k):
        C[i] = X[y == i].mean(axis=0)
    return C

def match_centroids(Ca: np.ndarray, Cb: np.ndarray) -> dict:
    # コサイン距離で対応（小さいほど近い）
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
    return {"row": r.tolist(), "col": c.tolist(), "cost": cost[r, c].tolist(), "mean_cost": float(np.mean(cost[r, c]))}

def main(
    pattern: str = "runs/*/epsilon_event_embeddings.csv",
    k: int = 2,
    method: str = "kmeans",
    seed: int = 0,
    out_csv: str = "stepG_cluster_stability_pairwise.csv",
):
    paths = sorted(glob.glob(pattern))
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 runs. No files matched: {pattern}")

    runs = []
    for p in paths:
        df = load_run_csv(p)
        run_name = os.path.basename(os.path.dirname(p))
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

        # 重要：runごとにスケールが変わるのを防ぐため標準化
        X = df[emb_cols].to_numpy()
        Xs = StandardScaler().fit_transform(X)

        y = cluster_labels(Xs, method=method, k=k, seed=seed)

        runs.append({
            "name": run_name,
            "path": p,
            "df": df,
            "emb_cols": emb_cols,
            "X": X,
            "Xs": Xs,
            "y": y
        })

    # --- pairwise stability (ARI/NMI) ---
    rows = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            ri, rj = runs[i], runs[j]

            # 注意：イベント集合が一致していることが前提
            ei = ri["df"]["event_epoch"].to_numpy()
            ej = rj["df"]["event_epoch"].to_numpy()
            if not np.array_equal(ei, ej):
                # ここで落とす（必要なら “イベント集合の揃え” を別途実装）
                rows.append({
                    "run_a": ri["name"],
                    "run_b": rj["name"],
                    "ARI": np.nan,
                    "NMI": np.nan,
                    "centroid_match_mean_cost": np.nan,
                    "note": "event_epoch mismatch (need alignment)"
                })
                continue

            ari = adjusted_rand_score(ri["y"], rj["y"])
            nmi = normalized_mutual_info_score(ri["y"], rj["y"])

            Ca = centroid_matrix(ri["Xs"], ri["y"], k=k)
            Cb = centroid_matrix(rj["Xs"], rj["y"], k=k)
            match = match_centroids(Ca, Cb)

            rows.append({
                "run_a": ri["name"],
                "run_b": rj["name"],
                "ARI": float(ari),
                "NMI": float(nmi),
                "centroid_match_mean_cost": match["mean_cost"],
                "note": ""
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    print(out.head(10))

if __name__ == "__main__":
    main()