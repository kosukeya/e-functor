# attn_common_pca_and_loadings.py
from pathlib import Path
import torch
import numpy as np
from sklearn.decomposition import PCA

# ---- 7次元 attention の各次元ラベル（あなたのコードの意味に合わせて調整OK）----
# attn は「growth query 行（head平均）」で、列は 7トークンへの注意。
# 例: [plant, sun, water, (他の内部/補助トークン...), self/growth ...] のような並び。
# ※正確な並びは model 側のトークン順に依存します。
ATTN_LABELS = [f"tok{i}" for i in range(7)]

def load_attn(pt_path: str, key: str = "attn"):
    d = torch.load(pt_path, map_location="cpu")
    if key not in d:
        raise KeyError(f"'{key}' not found. available keys: {sorted(list(d.keys()))}")

    A = d[key]  # (n,7)
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)
    A = A.detach().cpu().numpy()

    epoch = int(d.get("epoch", -1))
    alpha_used = float(d.get("alpha_used", float("nan")))
    return epoch, alpha_used, A

def print_loadings(pca_obj: PCA, labels):
    # components_: (n_components, n_features)
    comps = pca_obj.components_
    for pc in range(comps.shape[0]):
        v = comps[pc]
        order = np.argsort(-np.abs(v))
        print(f"\n=== attn PC{pc+1} loadings (sorted by |loading|) ===")
        for j in order:
            lab = labels[j] if j < len(labels) else f"dim{j}"
            print(f"{lab:>8s}: {v[j]:+.4f}")
        print(f"  (sum abs) {np.abs(v).sum():.4f} / (L2) {np.linalg.norm(v):.4f}")

def main():
    run_dir = Path("runs/test1/islands")
    p400  = run_dir / "island_epoch00400.pt"
    p1600 = run_dir / "island_epoch01600.pt"

    e1, a1, A1 = load_attn(str(p400), key="attn")   # or "attn_growth_row" でも同じはず
    e2, a2, A2 = load_attn(str(p1600), key="attn")

    # ---- 共通PCA（concat-fit）----
    A_all = np.concatenate([A1, A2], axis=0)  # (2n,7)
    pca_attn = PCA(n_components=2, random_state=0)
    pca_attn.fit(A_all)

    print("A explained_variance_ratio:", pca_attn.explained_variance_ratio_)
    print_loadings(pca_attn, ATTN_LABELS)

    # ついでに各epochのPC座標も取れる（後で I_PC と相関を見るのに使う）
    Z1 = pca_attn.transform(A1)
    Z2 = pca_attn.transform(A2)
    print(f"\n[epoch {e1}] attn_PC mean:", Z1.mean(axis=0))
    print(f"[epoch {e2}] attn_PC mean:", Z2.mean(axis=0))

if __name__ == "__main__":
    main()
