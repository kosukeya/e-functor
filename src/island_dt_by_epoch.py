# island_dt_by_epoch.py
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

CSV_PATH = "runs/island_env_error.csv"  # 必要ならパス変更


def fit_stump_by_epoch(df: pd.DataFrame, feature="err_abs_mean", target="island"):
    rows = []
    for epoch, g in df.groupby("epoch"):
        # epoch×island 粒度なので通常2行（island=0/1）
        g = g.dropna(subset=[feature, target]).copy()
        if g[target].nunique() < 2:
            # 片方の島しか無いepochはスキップ
            continue

        X = g[[feature]].to_numpy()
        y = g[target].astype(int).to_numpy()

        clf = DecisionTreeClassifier(max_depth=1, random_state=0)
        clf.fit(X, y)

        # sklearn stump parameters
        tree = clf.tree_
        feat_idx = tree.feature[0]
        thr = float(tree.threshold[0]) if feat_idx != -2 else np.nan  # -2 = leaf

        # どっち側がどのクラスか（left/rightの予測クラス）
        left_node = tree.children_left[0]
        right_node = tree.children_right[0]

        left_counts = tree.value[left_node][0]
        right_counts = tree.value[right_node][0]
        left_class = int(np.argmax(left_counts))
        right_class = int(np.argmax(right_counts))

        # 精度（このepoch内の2点に対する当てはまり）
        yhat = clf.predict(X)
        acc = float(accuracy_score(y, yhat))

        rows.append({
            "epoch": int(epoch),
            "feature": feature,
            "threshold": thr,
            "left_rule": f"{feature} <= {thr:.6g} -> class {left_class}",
            "right_rule": f"{feature} >  {thr:.6g} -> class {right_class}",
            "left_class": left_class,
            "right_class": right_class,
            "accuracy": acc,
            "n_rows": int(len(g)),
            # 参考：そのepochの島ごとの値も残す
            "val_island0": float(g.loc[g[target]==0, feature].iloc[0]) if (g[target]==0).any() else np.nan,
            "val_island1": float(g.loc[g[target]==1, feature].iloc[0]) if (g[target]==1).any() else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return out


def main():
    df = pd.read_csv(CSV_PATH)
    # 念のため型を整える
    df["epoch"] = df["epoch"].astype(int)
    df["island"] = df["island"].astype(int)

    out = fit_stump_by_epoch(df, feature="err_abs_mean", target="island")
    out.to_csv("island_dt_by_epoch_err_abs_mean.csv", index=False)
    print("saved: island_dt_by_epoch_err_abs_mean.csv")
    print(out[["epoch","threshold","left_class","right_class","accuracy","val_island0","val_island1"]])

    # 図：thresholdの時系列
    plt.figure()
    plt.plot(out["epoch"], out["threshold"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("DecisionTree threshold (err_abs_mean)")
    plt.title("Stump threshold by epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("island_dt_threshold_timeseries.png", dpi=200)
    print("saved: island_dt_threshold_timeseries.png")

    # 図：島ごとの値（参考）
    plt.figure()
    plt.plot(out["epoch"], out["val_island0"], marker="o", label="island=0 value")
    plt.plot(out["epoch"], out["val_island1"], marker="o", label="island=1 value")
    plt.xlabel("epoch")
    plt.ylabel("err_abs_mean")
    plt.title("err_abs_mean by island (epoch×island)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("err_abs_mean_by_island_timeseries.png", dpi=200)
    print("saved: err_abs_mean_by_island_timeseries.png")


if __name__ == "__main__":
    main()
