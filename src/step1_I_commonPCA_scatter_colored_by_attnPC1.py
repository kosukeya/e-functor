# step1_I_commonPCA_scatter_colored_by_attnPC1.py
from pathlib import Path
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RUN_DIR = Path("runs/test1")
ISLAND_DIR = RUN_DIR / "islands"

# ゼロクロス前後（例）
EPOCHS = [600, 800, 1000, 1200]

# model.py で確定した token順（attn列の意味）
ATTN_LABELS = ["plant","sun","water","I1","I2","I3","self(growth)"]
IDX = {k:i for i,k in enumerate(ATTN_LABELS)}

def list_island_pts(island_dir: Path):
    pts = {}
    for p in island_dir.glob("island_epoch*.pt"):
        m = re.search(r"epoch(\d+)\.pt$", p.name)
        if m:
            pts[int(m.group(1))] = p
    return pts

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std() * b0.std()) + 1e-12
    return float((a0 * b0).mean() / denom)

def load_I_env_attn(pt_path: Path, attn_key="attn"):
    d = torch.load(pt_path, map_location="cpu")

    I = d["I"]                          # (n,3,32)
    env = d["env"]                      # (n,3)  plant/sun/water
    A = d[attn_key]                     # (n,7) 共に growth行（head平均）想定

    I = to_np(I)
    env = to_np(env)
    A = to_np(A)

    # I を flatten -> (n, 3*32)
    if I.ndim != 3:
        raise ValueError(f"Unexpected I shape: {I.shape}")
    n, k, D = I.shape
    X = I.reshape(n, k * D)

    return X, env, A, d

def anchor_pca_by_feature_loading(pca: PCA, feature_index: int, pc_index: int = 0):
    """指定特徴の loading が + になるようにPC符号を固定（components_ を反転）"""
    if pca.components_[pc_index, feature_index] < 0:
        pca.components_[pc_index, :] *= -1
    return pca

def main():
    pt_map = list_island_pts(ISLAND_DIR)
    for ep in EPOCHS:
        if ep not in pt_map:
            raise FileNotFoundError(f"island for epoch={ep} not found in {ISLAND_DIR}")

    # ---- 1) まとめてロード（I / env / attn）----
    X_by_ep, env_by_ep, A_by_ep = {}, {}, {}
    for ep in EPOCHS:
        X, env, A, _ = load_I_env_attn(pt_map[ep], attn_key="attn")
        X_by_ep[ep] = X
        env_by_ep[ep] = env
        A_by_ep[ep] = A

    # ---- 2) 共通PCA（I） concat-fit ----
    X_all = np.concatenate([X_by_ep[ep] for ep in EPOCHS], axis=0)
    pca_I = PCA(n_components=2, random_state=0)
    ZI_all = pca_I.fit_transform(X_all)
    # I-PC符号アンカー：PC1が plant(env[:,0]) と正相関になるように固定
    # （envと相関を見て、必要なら components_ を反転）
    # ※ pca_I.components_ を反転しても transform の結果は自動では反転しないので、Zも反転する
    #    最小なので、ここは「相関で判定してZを反転」方式にする
    # まず代表として全体のenvも結合
    env_all = np.concatenate([env_by_ep[ep] for ep in EPOCHS], axis=0)
    plant_all = env_all[:, 0]
    if corr(ZI_all[:, 0], plant_all) < 0:
        pca_I.components_[0, :] *= -1
        ZI_all[:, 0] *= -1
    if corr(ZI_all[:, 1], plant_all) < 0:
        # PC2まで plant 正に寄せたい場合（好み）。不要ならコメントアウトOK
        pass

    print("I common PCA explained_variance_ratio:", pca_I.explained_variance_ratio_)
    print("I: corr(PC1, plant)=", f"{corr(ZI_all[:,0], plant_all):+.3f}")

    # ---- 3) 共通PCA（attn） concat-fit ----
    A_all = np.concatenate([A_by_ep[ep] for ep in EPOCHS], axis=0)
    pca_A = PCA(n_components=2, random_state=0)
    ZA_all = pca_A.fit_transform(A_all)

    # attn-PC符号アンカー：PC1は I1 loading が + になるように固定（あなたの方針）
    if pca_A.components_[0, IDX["I1"]] < 0:
        pca_A.components_[0, :] *= -1
        ZA_all[:, 0] *= -1
    if pca_A.components_[1, IDX["I3"]] < 0:
        pca_A.components_[1, :] *= -1
        ZA_all[:, 1] *= -1

    print("A common PCA explained_variance_ratio:", pca_A.explained_variance_ratio_)
    print("A: PC1 loadings:", {ATTN_LABELS[i]: float(pca_A.components_[0,i]) for i in range(7)})

    # ---- 4) epochごとに I座標 & attn_PC1 を作って可視化 ----
    # 共通軸レンジで描く（比較しやすい）
    # まず epochごとの ZI と attn_PC1 を作る
    ZI_by_ep, attnPC1_by_ep = {}, {}
    for ep in EPOCHS:
        ZI = pca_I.transform(X_by_ep[ep])
        if corr(ZI[:, 0], env_by_ep[ep][:, 0]) < 0:
            # 念のためepoch単位でも符号ズレが出たら反転（基本は起きない想定）
            ZI[:, 0] *= -1

        ZA = pca_A.transform(A_by_ep[ep])
        if pca_A.components_[0, IDX["I1"]] < 0:
            ZA[:, 0] *= -1

        ZI_by_ep[ep] = ZI
        attnPC1_by_ep[ep] = ZA[:, 0]  # 色

    x_min = min(ZI_by_ep[ep][:,0].min() for ep in EPOCHS)
    x_max = max(ZI_by_ep[ep][:,0].max() for ep in EPOCHS)
    y_min = min(ZI_by_ep[ep][:,1].min() for ep in EPOCHS)
    y_max = max(ZI_by_ep[ep][:,1].max() for ep in EPOCHS)

    # 2x2想定（epochが4つのとき）。数が違う場合は自動で並べる
    n = len(EPOCHS)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    # 色レンジを揃える（比較しやすい）
    c_min = min(attnPC1_by_ep[ep].min() for ep in EPOCHS)
    c_max = max(attnPC1_by_ep[ep].max() for ep in EPOCHS)

    last_sc = None
    for i, ep in enumerate(EPOCHS):
        ax = axes[i]
        ZI = ZI_by_ep[ep]
        c = attnPC1_by_ep[ep]

        last_sc = ax.scatter(
            ZI[:,0], ZI[:,1],
            c=c,
            s=10,
            vmin=c_min, vmax=c_max
        )
        ax.set_title(f"I common PCA (epoch={ep}) colored by attn_PC1")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xlabel("I_PC1"); ax.set_ylabel("I_PC2")

    # 余ったaxesを消す
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.colorbar(last_sc, ax=axes[:n].tolist(), label="attn_PC1 (common PCA)")
    plt.show()

if __name__ == "__main__":
    main()
