# stepG_feature_stability.py
# Step G-1: "効いている特徴"の安定性チェック（bootstrap + LOO）
#
# Inputs:
#   - island_env_error.csv
#   - island_profile.csv
#   - threshold_timeseries_with_events.csv
# Outputs:
#   - stepG_outputs/
#       event_binary_stability.csv
#       event_sign_stability.csv
#       event_binary_topfreq.png
#       event_sign_topfreq.png
#       event_binary_coef_violin_top.png (任意)
#       event_sign_coef_violin_top.png   (任意)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut


# -------------------------
# 1) Load & build dataset
# -------------------------
def pivot_wide(df: pd.DataFrame, value_cols, prefix: str):
    # epoch×island -> wide columns
    wide = df.pivot_table(index="epoch", columns="island", values=value_cols, aggfunc="mean")
    wide.columns = [f"{prefix}{col}_island{isl}" for col, isl in wide.columns]
    wide = wide.reset_index()

    # add diff (island0 - island1) for each base col
    for col in value_cols:
        c0 = f"{prefix}{col}_island0"
        c1 = f"{prefix}{col}_island1"
        if c0 in wide.columns and c1 in wide.columns:
            wide[f"{prefix}{col}_diff_0_minus_1"] = wide[c0] - wide[c1]
    return wide


def build_epoch_table(env_path, prof_path, thr_path):
    env = pd.read_csv(env_path)
    prof = pd.read_csv(prof_path)
    thr = pd.read_csv(thr_path)

    # columns to pivot
    env_cols = [c for c in env.columns if c not in ["epoch", "island", "n"]]
    prof_cols = [c for c in prof.columns if c not in ["epoch", "island", "n"]]

    env_w = pivot_wide(env, env_cols, prefix="env__")
    prof_w = pivot_wide(prof, prof_cols, prefix="prof__")

    feat = env_w.merge(prof_w, on="epoch", how="inner")

    # merge threshold series (has is_event, threshold_diff, etc.)
    thr_keep = [c for c in thr.columns if c in ["epoch", "threshold", "threshold_prev", "threshold_diff", "is_event"]]
    feat = feat.merge(thr[thr_keep], on="epoch", how="left")

    feat = feat.sort_values("epoch").reset_index(drop=True)
    return feat


def build_prev_dataset(feat: pd.DataFrame):
    """
    Predict y(t) using Xprev = features at epoch(t-1).
    """
    epochs = feat["epoch"].values
    # shift features by 1 step
    X_cols = [c for c in feat.columns if c not in ["epoch"]]

    # Make Xprev columns
    Xprev = feat[X_cols].shift(1)
    Xprev.columns = [f"Xprev__{c}" for c in X_cols]

    out = pd.concat([feat[["epoch"]], Xprev, feat[["is_event", "threshold_diff"]]], axis=1)

    # drop first row (no prev)
    out = out.dropna(subset=[c for c in Xprev.columns]).reset_index(drop=True)

    # binary label: event or not
    y_event = out["is_event"].astype(int).values

    # sign label among events: up(1) vs down(0)
    # define sign only for rows where is_event==1
    out["event_sign"] = np.nan
    mask_ev = out["is_event"].astype(int) == 1
    # threshold_diff > 0 => up
    out.loc[mask_ev, "event_sign"] = (out.loc[mask_ev, "threshold_diff"] > 0).astype(int)

    return out, y_event


# -------------------------
# 2) Models
# -------------------------
def make_logistic(C=1.0, penalty="l2", solver=None, max_iter=5000):
    if solver is None:
        solver = "liblinear" if penalty in ["l1", "l2"] else "lbfgs"

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=None
        ))
    ])


def fit_get_coef(model, X, y, feature_names):
    model.fit(X, y)
    clf = model.named_steps["clf"]
    coef = clf.coef_.reshape(-1)
    return pd.Series(coef, index=feature_names)


# -------------------------
# 3) Stability checks
# -------------------------
def bootstrap_coef_stability(
    model, X, y, feature_names,
    n_boot=300, topk_list=(10, 20), seed=0
):
    """
    Bootstrap resampling:
      - coef mean/std
      - sign consistency
      - top-k selection frequency
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    coefs = []
    ok = 0
    while ok < n_boot:
        idx = rng.integers(0, n, size=n)  # sample with replacement
        yb = y[idx]
        # need both classes
        if len(np.unique(yb)) < 2:
            continue
        cb = fit_get_coef(model, X[idx], yb, feature_names)
        coefs.append(cb.values)
        ok += 1

    coefs = np.vstack(coefs)  # (n_boot, n_features)

    coef_mean = coefs.mean(axis=0)
    coef_std = coefs.std(axis=0)

    # sign consistency: fraction of bootstraps with same sign as mean (ignoring near-zero)
    eps = 1e-12
    mean_sign = np.sign(coef_mean + eps)
    sign_same = (np.sign(coefs + eps) == mean_sign[None, :]).mean(axis=0)

    # top-k selection frequency by abs(coef)
    topk_freq = {}
    abs_coefs = np.abs(coefs)
    for k in topk_list:
        hits = np.zeros(abs_coefs.shape[1], dtype=float)
        for b in range(abs_coefs.shape[0]):
            top_idx = np.argsort(abs_coefs[b])[::-1][:k]
            hits[top_idx] += 1.0
        topk_freq[k] = hits / abs_coefs.shape[0]

    rows = []
    for j, f in enumerate(feature_names):
        row = {
            "feature": f,
            "coef_mean": coef_mean[j],
            "coef_std": coef_std[j],
            "abs_coef_mean": abs(coef_mean[j]),
            "sign_consistency": sign_same[j],
        }
        for k in topk_list:
            row[f"top{k}_freq"] = topk_freq[k][j]
        rows.append(row)

    st = pd.DataFrame(rows).sort_values("abs_coef_mean", ascending=False).reset_index(drop=True)
    return st, coefs


def loo_topk_frequency(model, X, y, feature_names, topk=10):
    loo = LeaveOneOut()
    hits = np.zeros(len(feature_names), dtype=float)
    used = 0
    for tr, te in loo.split(X):
        yt = y[tr]
        if len(np.unique(yt)) < 2:
            # skip degenerate fold
            continue
        coef = fit_get_coef(model, X[tr], yt, feature_names).values
        top_idx = np.argsort(np.abs(coef))[::-1][:topk]
        hits[top_idx] += 1.0
        used += 1
    if used == 0:
        return pd.Series(np.zeros(len(feature_names)), index=feature_names), used
    return pd.Series(hits / used, index=feature_names), used


def plot_topfreq(st_df, out_png, title, topn=25, freq_col="top10_freq"):
    sub = st_df.sort_values(freq_col, ascending=False).head(topn)
    plt.figure(figsize=(12, 6))
    plt.barh(sub["feature"][::-1], sub[freq_col][::-1])
    plt.xlabel(freq_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_violin_top(coefs_mat, feature_names, st_df, out_png, topn=12, title="coef distribution (top)"):
    top_feats = st_df.head(topn)["feature"].tolist()
    idx = [feature_names.index(f) for f in top_feats]

    data = [coefs_mat[:, j] for j in idx]

    plt.figure(figsize=(12, 5))
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(range(1, len(top_feats) + 1), top_feats, rotation=75, ha="right", fontsize=8)
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -------------------------
# 4) Main
# -------------------------
def main():
    env_path = "/content/island_env_error.csv"
    prof_path = "/content/island_profile.csv"
    thr_path = "/content/threshold_timeseries_with_events.csv"

    outdir = "stepG_outputs"
    os.makedirs(outdir, exist_ok=True)

    feat = build_epoch_table(env_path, prof_path, thr_path)
    ds, y_event = build_prev_dataset(feat)

    # Build X matrix
    X_cols = [c for c in ds.columns if c.startswith("Xprev__")]
    X = ds[X_cols].to_numpy(dtype=float)
    feature_names = X_cols

    # -------------------------
    # A) event binary stability
    # -------------------------
    model_bin = make_logistic(C=1.0, penalty="l2", solver="liblinear")

    st_bin, coefs_bin = bootstrap_coef_stability(
        model_bin, X, y_event, feature_names,
        n_boot=400, topk_list=(10, 20), seed=42
    )

    # add LOO top10 frequency
    loo_freq10, used = loo_topk_frequency(model_bin, X, y_event, feature_names, topk=10)
    st_bin["loo_top10_freq"] = st_bin["feature"].map(loo_freq10.to_dict()).fillna(0.0)
    st_bin["loo_used_folds"] = used

    st_bin.to_csv(f"{outdir}/event_binary_stability.csv", index=False)
    plot_topfreq(st_bin, f"{outdir}/event_binary_topfreq.png",
                 title="Event(0/1): top10 selection frequency (bootstrap)",
                 topn=25, freq_col="top10_freq")
    plot_violin_top(coefs_bin, feature_names, st_bin,
                    f"{outdir}/event_binary_coef_violin_top.png",
                    topn=12, title="Event(0/1): coef distribution (bootstrap, top features)")

    # -------------------------
    # B) event sign stability (among events only)
    # -------------------------
    ds_ev = ds[ds["is_event"].astype(int) == 1].copy()
    if ds_ev.shape[0] >= 4 and ds_ev["event_sign"].notna().all():
        y_sign = ds_ev["event_sign"].astype(int).values
        Xs = ds_ev[X_cols].to_numpy(dtype=float)

        # if all same sign, can't do classification
        if len(np.unique(y_sign)) >= 2:
            model_sign = make_logistic(C=1.0, penalty="l2", solver="liblinear")

            st_sign, coefs_sign = bootstrap_coef_stability(
                model_sign, Xs, y_sign, feature_names,
                n_boot=400, topk_list=(10, 20), seed=7
            )

            loo_freq10_s, used_s = loo_topk_frequency(model_sign, Xs, y_sign, feature_names, topk=10)
            st_sign["loo_top10_freq"] = st_sign["feature"].map(loo_freq10_s.to_dict()).fillna(0.0)
            st_sign["loo_used_folds"] = used_s

            st_sign.to_csv(f"{outdir}/event_sign_stability.csv", index=False)
            plot_topfreq(st_sign, f"{outdir}/event_sign_topfreq.png",
                         title="Event SIGN(↑/↓): top10 selection frequency (bootstrap, among events)",
                         topn=25, freq_col="top10_freq")
            plot_violin_top(coefs_sign, feature_names, st_sign,
                            f"{outdir}/event_sign_coef_violin_top.png",
                            topn=12, title="Event SIGN(↑/↓): coef distribution (bootstrap, top features)")
        else:
            # write a small note
            with open(f"{outdir}/event_sign_stability.txt", "w", encoding="utf-8") as f:
                f.write("event_sign has only one class in the data. Cannot fit sign classifier.\n")
    else:
        with open(f"{outdir}/event_sign_stability.txt", "w", encoding="utf-8") as f:
            f.write("Not enough event samples to analyze sign stability.\n")

    print(f"[done] outputs saved to: {outdir}")


if __name__ == "__main__":
    main()