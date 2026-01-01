# stable_I_compare.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

RUN_DIR = Path("runs/test1")
ISLAND_DIR = RUN_DIR / "islands"

# ---- 手で決める：ゼロクロス前後（おすすめ）----
PRE  = [600, 800]
POST = [1000, 1200]
# 余力があれば：
# POST2 = [1400, 1600]

def list_island_pts(island_dir: Path):
    pts = {}
    for p in island_dir.glob("island_epoch*.pt"):
        m = re.search(r"epoch(\d+)\.pt$", p.name)
        if m:
            pts[int(m.group(1))] = p
    return pts

def load_I_flat(pt_path: Path):
    d = torch.load(pt_path, map_location="cpu")
    I = d["I"]
    if not isinstance(I, torch.Tensor):
        I = torch.tensor(I)
    I = I.detach().cpu()  # (N,3,32)
    assert I.dim() == 3 and I.shape[1] == 3, f"Unexpected I shape: {tuple(I.shape)}"
    N, K, D = I.shape
    X = I.reshape(N, K*D).numpy()  # (N,96)
    # 参考用：スロット別ノルム
    slot = I.numpy()  # (N,3,32)
    slot_norm = np.linalg.norm(slot, axis=2)  # (N,3)
    # alpha_used なども一応
    alpha_used = float(d.get("alpha_used", np.nan))
    return X, slot_norm, alpha_used

def effective_rank_from_spectrum(var):
    # var: 各次元の分散（>=0）や特異値^2相当
    v = np.asarray(var, dtype=np.float64)
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v / s
    p = p[p > 1e-12]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))

def per_epoch_stats(ep, X, slot_norm, alpha_used, pca_common):
    # center (within epoch)
    Xc = X - X.mean(axis=0, keepdims=True)

    # 共通PCA座標
    Z = pca_common.transform(X)  # (N,2)
    pc_mean = Z.mean(axis=0)
    pc_var  = Z.var(axis=0)

    # epoch固有の分散スペクトル（有効ランク用）
    # 96次元での分散
    var96 = Xc.var(axis=0)
    eff_rank = effective_rank_from_spectrum(var96)

    # 「共通PCでどれだけ説明できているか」を rough に見る指標
    total_var = var96.sum() + 1e-12
    evr_pc1 = float(pc_var[0] / total_var)
    evr_pc2 = float(pc_var[1] / total_var)

    out = {
        "epoch": ep,
        "alpha_used_pt": alpha_used,
        "I_PC1_mean": float(pc_mean[0]),
        "I_PC2_mean": float(pc_mean[1]),
        "I_PC1_var":  float(pc_var[0]),
        "I_PC2_var":  float(pc_var[1]),
        "EVR_in_epoch_PC1": evr_pc1,
        "EVR_in_epoch_PC2": evr_pc2,
        "eff_rank_I": eff_rank,
        "slot1_norm_mean": float(slot_norm[:,0].mean()),
        "slot2_norm_mean": float(slot_norm[:,1].mean()),
        "slot3_norm_mean": float(slot_norm[:,2].mean()),
        "slot1_norm_std":  float(slot_norm[:,0].std()),
        "slot2_norm_std":  float(slot_norm[:,1].std()),
        "slot3_norm_std":  float(slot_norm[:,2].std()),
    }
    return out

def main():
    pts = list_island_pts(ISLAND_DIR)
    need = sorted(set(PRE + POST))
    missing = [ep for ep in need if ep not in pts]
    if missing:
        raise FileNotFoundError(f"Missing island pt for epochs: {missing}")

    # --- load all selected epochs
    X_by_ep = {}
    slot_by_ep = {}
    alpha_by_ep = {}
    for ep in need:
        X, slot_norm, a_used = load_I_flat(pts[ep])
        X_by_ep[ep] = X
        slot_by_ep[ep] = slot_norm
        alpha_by_ep[ep] = a_used

    # --- common PCA (concat-fit) on selected epochs
    X_all = np.concatenate([X_by_ep[ep] for ep in need], axis=0)
    pca = PCA(n_components=2, random_state=0).fit(X_all)
    print("I common PCA explained_variance_ratio:", pca.explained_variance_ratio_)

    # --- stats table
    rows = []
    for ep in need:
        rows.append(per_epoch_stats(ep, X_by_ep[ep], slot_by_ep[ep], alpha_by_ep[ep], pca))
    df = pd.DataFrame(rows).set_index("epoch").sort_index()

    # --- pre/post summary
    df_pre  = df.loc[PRE]
    df_post = df.loc[POST]
    summary = pd.DataFrame({
        "pre_mean":  df_pre.mean(numeric_only=True),
        "post_mean": df_post.mean(numeric_only=True),
    })
    summary["post_minus_pre"] = summary["post_mean"] - summary["pre_mean"]

    print("\n=== per-epoch I stats (selected) ===")
    print(df.round(6))

    print("\n=== pre vs post summary ===")
    print(summary.round(6))

    # save
    out_csv1 = RUN_DIR / "I_stats_zero_cross_selected.csv"
    out_csv2 = RUN_DIR / "I_stats_zero_cross_prepost_summary.csv"
    df.to_csv(out_csv1)
    summary.to_csv(out_csv2)
    print(f"\n[saved] {out_csv1}")
    print(f"[saved] {out_csv2}")

if __name__ == "__main__":
    main()
