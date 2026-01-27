# attn_pc1_zero_contour_fixed.py
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

RUN_DIR = Path("runs/test1")
ISLAND_DIR = RUN_DIR / "islands"

E800 = 800
E1000 = 1000

# model.py より token順は確定：
# [plant, sun, water, I1, I2, I3, growth]
ATTN_LABELS = ["plant","sun","water","I1","I2","I3","self(growth)"]
IDX = {k:i for i,k in enumerate(ATTN_LABELS)}

def load_pt(epoch: int):
    p = ISLAND_DIR / f"island_epoch{epoch:05d}.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    d = torch.load(p, map_location="cpu")
    I = d["I"]  # (N,3,32)
    A = d["attn"]  # (N,7) growth行のattention（平均済み）
    if isinstance(I, torch.Tensor):
        I = I.detach().cpu().numpy()
    else:
        I = np.asarray(I)
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    else:
        A = np.asarray(A)

    # flatten I -> (N,96)
    N = I.shape[0]
    X = I.reshape(N, -1)
    return X, A, d

def corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    return float((a0 * b0).mean() / ((a0.std() * b0.std()) + 1e-12))

def zscore(x):
    x = np.asarray(x).reshape(-1)
    return (x - x.mean()) / (x.std() + 1e-12)

def align_sign_to_ref(a_ref, a_other):
    """
    a_other の符号を、a_ref と相関が最大(正方向)になる向きに揃える
    """
    c1 = corr(a_other, a_ref)
    c2 = corr(-a_other, a_ref)
    if c2 > c1:
        return -a_other, -1, c2
    return a_other, +1, c1

def fit_and_plot_contour(Z, target_z, title, ax):
    """
    Z: (N,2) I-PCA座標
    target_z: (N,) 回帰ターゲット（zscore(attn_PC1)）
    """
    reg = LinearRegression()
    reg.fit(Z, target_z)
    r2 = reg.score(Z, target_z)

    w1, w2 = reg.coef_
    c0 = reg.intercept_

    # 散布図：色は target_z（= zscore(attn_PC1)）
    sc = ax.scatter(Z[:,0], Z[:,1], c=target_z, s=12)
    ax.set_title(f"{title}\nfit z(attn_PC1) = w1*I_PC1 + w2*I_PC2 + c | R²={r2:.3f}")
    ax.set_xlabel("I_PC1 (common)")
    ax.set_ylabel("I_PC2 (common)")
    plt.colorbar(sc, ax=ax, label="z(attn_PC1)")

    # 等値線 z=0（線形なので直線）
    x_min, x_max = Z[:,0].min(), Z[:,0].max()
    xs = np.linspace(x_min, x_max, 200)

    # w1*x + w2*y + c0 = 0 -> y = -(w1*x + c0)/w2
    # w2 がほぼ0の場合は縦線 x = -c0/w1
    eps = 1e-12
    if abs(w2) > 1e-8:
        ys = -(w1 * xs + c0) / (w2 + eps)
        ax.plot(xs, ys, "--", linewidth=2, label="z(attn_PC1)=0")
    elif abs(w1) > 1e-8:
        x0 = -c0 / (w1 + eps)
        ax.axvline(x0, linestyle="--", linewidth=2, label="z(attn_PC1)=0")
    else:
        # ほぼ定数：線が定義できない
        ax.text(0.02, 0.02, "degenerate fit (w1≈w2≈0)", transform=ax.transAxes)

    ax.legend(loc="lower left")

    # 追加で係数も返す
    return {"w1": float(w1), "w2": float(w2), "c": float(c0), "r2": float(r2)}

def main():
    # ---- load ----
    X800, A800, d800 = load_pt(E800)
    X1000, A1000, d1000 = load_pt(E1000)

    # ---- common I PCA (fit on both epochs) ----
    X_all = np.concatenate([X800, X1000], axis=0)
    pca_I = PCA(n_components=2, random_state=0)
    Z_all = pca_I.fit_transform(X_all)
    Z800 = pca_I.transform(X800)
    Z1000 = pca_I.transform(X1000)
    print("I common PCA explained_variance_ratio:", pca_I.explained_variance_ratio_)

    # ---- define attn_PC1 (use the SAME pca_attn used in your pipeline) ----
    # ここでは「attnそのものからPC1を作る」のではなく、
    # すでにあなたが作っている attn_PC1（共通attn PCAのPC1）を使うのが理想。
    # しかし最小コードとして、ここでは attn(7次元)の共通PCAからPC1を作ります。
    # ※あなたの "anchor" 基準(I1+)も再現します。
    A_all = np.concatenate([A800, A1000], axis=0)
    pca_A = PCA(n_components=2, random_state=0)
    ZA_all = pca_A.fit_transform(A_all)

    # アンカー：PC1のI1 loading を + にする（model順の I1 は index=3）
    if pca_A.components_[0, IDX["I1"]] < 0:
        pca_A.components_[0,:] *= -1
        ZA_all[:,0] *= -1

    ZA800 = pca_A.transform(A800)
    ZA1000 = pca_A.transform(A1000)
    if pca_A.components_[0, IDX["I1"]] < 0:  # 念のため
        ZA800[:,0] *= -1
        ZA1000[:,0] *= -1

    attn_pc1_800 = ZA800[:,0].copy()
    attn_pc1_1000 = ZA1000[:,0].copy()

    # ---- sign auto alignment: make 800 aligned to 1000 ----
    attn_pc1_800_aligned, sgn, cc = align_sign_to_ref(attn_pc1_1000, attn_pc1_800)
    print(f"[sign auto] chosen sign for epoch800: {sgn:+d}, corr(after align)={cc:+.6f}")
    print(f"[check] corr(attn_pc1_800, attn_pc1_1000)    ={corr(attn_pc1_800, attn_pc1_1000):+.6f}")
    print(f"[check] corr(-attn_pc1_800, attn_pc1_1000)   ={corr(-attn_pc1_800, attn_pc1_1000):+.6f}")

    # ---- zscore target (each epoch individually) ----
    # 「z=0 を平均境界」にしたいので、各epochでzscore
    z800 = zscore(attn_pc1_800_aligned)
    z1000 = zscore(attn_pc1_1000)

    # ---- plot: two panels (800 / 1000) with each z=0 line ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    info800 = fit_and_plot_contour(
        Z800, z800,
        title=f"I-PCA (epoch {E800}) colored by z(attn_PC1) [aligned sign {sgn:+d}]",
        ax=axes[0]
    )
    info1000 = fit_and_plot_contour(
        Z1000, z1000,
        title=f"I-PCA (epoch {E1000}) colored by z(attn_PC1)",
        ax=axes[1]
    )

    print("\n=== linear fit params (z(attn_PC1) ~ I_PC1,I_PC2) ===")
    print(f"epoch {E800}:  w1={info800['w1']:+.6f} w2={info800['w2']:+.6f} c={info800['c']:+.6f}  R²={info800['r2']:.4f}")
    print(f"epoch {E1000}: w1={info1000['w1']:+.6f} w2={info1000['w2']:+.6f} c={info1000['c']:+.6f} R²={info1000['r2']:.4f}")

    plt.show()

if __name__ == "__main__":
    main()
