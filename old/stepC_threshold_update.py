# stepC_threshold_update.py
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def _try_load(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in ["epoch", "prev_epoch", "island"]:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _diff_by_epoch(df: pd.DataFrame, key="epoch", prev_key="prev_epoch"):
    df = df.sort_values(key).reset_index(drop=True)
    if prev_key in df.columns:
        prev_map = df.set_index(key)
        delta = df.copy()
        for c in df.columns:
            if c in [key, prev_key]:
                continue
            delta[c] = df.apply(
                lambda r: r[c] - (prev_map.loc[r[prev_key], c] if r[prev_key] in prev_map.index else np.nan),
                axis=1,
            )
        return delta
    delta = df.copy()
    for c in df.columns:
        if c == key:
            continue
        delta[c] = df[c].diff()
    return delta


def _plot_timeseries(df, x, y, title, outpath: Path):
    plt.figure()
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def make_island_contrast(df, value_cols):
    out = []
    for col in value_cols:
        w = df.pivot(index="epoch", columns="island", values=col)
        w.columns = [f"{col}_island{int(i)}" for i in w.columns]
        out.append(w)
    wide = pd.concat(out, axis=1).reset_index()

    islands = sorted(df["island"].unique())
    if len(islands) >= 2:
        a, b = islands[0], islands[1]
        for col in value_cols:
            wide[f"{col}_diff_{int(a)}_minus_{int(b)}"] = (
                wide[f"{col}_island{int(a)}"] - wide[f"{col}_island{int(b)}"]
            )
    return wide


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--threshold-csv", type=str, default=None)
    ap.add_argument("--env-csv", type=str, default=None)
    ap.add_argument("--profile-csv", type=str, default=None)
    ap.add_argument("--eps-csv", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--fig-dir", type=str, default=None)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    thresh_path = Path(args.threshold_csv) if args.threshold_csv else (paths.derived_dir / "island_dt_by_epoch_err_abs_mean.csv")
    env_path = Path(args.env_csv) if args.env_csv else (paths.derived_dir / "island_env_error.csv")
    profile_path = Path(args.profile_csv) if args.profile_csv else (paths.derived_dir / "island_profile.csv")
    eps_path = Path(args.eps_csv) if args.eps_csv else (paths.derived_dir / "island_eps_summary.csv")

    for name, p in [("threshold", thresh_path), ("env", env_path), ("profile", profile_path), ("eps", eps_path)]:
        if not p.is_absolute():
            if name == "eps":
                eps_path = repo_root() / eps_path
            elif name == "threshold":
                thresh_path = repo_root() / thresh_path
            elif name == "env":
                env_path = repo_root() / env_path
            elif name == "profile":
                profile_path = repo_root() / profile_path

    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "stepC_threshold_update")
    fig_dir = Path(args.fig_dir) if args.fig_dir else (paths.figures_dir / "stepC_threshold_update")
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    if not fig_dir.is_absolute():
        fig_dir = repo_root() / fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dt = _load_csv(thresh_path)
    need_cols = [c for c in ["epoch", "threshold", "left_class", "right_class", "accuracy"] if c in dt.columns]
    thr = dt[need_cols].copy().sort_values("epoch").reset_index(drop=True)
    thr["prev_epoch"] = thr["epoch"].shift(1)
    thr.loc[thr.index[0], "prev_epoch"] = np.nan
    thr = _safe_numeric(thr)

    thr_delta = _diff_by_epoch(thr, key="epoch", prev_key="prev_epoch")
    thr_delta = thr_delta.rename(columns={"threshold": "delta_threshold"})
    thr_delta["threshold"] = thr["threshold"]

    env = _safe_numeric(_load_csv(env_path))
    env_cols = [
        "plant_mean", "plant_std", "sun_mean", "sun_std", "water_mean", "water_std",
        "err_abs_mean", "err_abs_p90", "err_signed_mean",
    ]
    env_cols = [c for c in env_cols if c in env.columns]
    env_epoch = make_island_contrast(env, env_cols)

    prof = _safe_numeric(_load_csv(profile_path))
    prof_cols = [
        "y_mix_mean", "y_mix_std", "corr_y_mix_true", "y_stat_mean", "y_sem_mean",
        "delta_sun_mean", "delta_plant_mean", "delta_ratio_sun_over_plant",
        "attn_mass_0_2", "attn_mass_3_5", "attn_mass_6",
    ]
    prof_cols = [c for c in prof_cols if c in prof.columns]
    prof_epoch = make_island_contrast(prof, prof_cols)

    eps = _try_load(eps_path)
    eps_epoch = None
    if eps is not None:
        eps = _safe_numeric(eps)
        keep = [c for c in eps.columns if c in ["epoch", "prev_epoch", "epsilon_all", "dM_all", "dC_all", "d_cf_all", "d_att_all", "d_self_all"]]
        keep += [c for c in eps.columns if c.startswith("epsilon_") or c.startswith("d_cf_") or c.startswith("d_att_") or c.startswith("d_self_")]
        keep = list(dict.fromkeys(keep))
        eps_epoch = eps[keep].copy()
        if "prev_epoch" not in eps_epoch.columns:
            eps_epoch["prev_epoch"] = eps_epoch["epoch"].shift(1)

    feat = env_epoch.merge(prof_epoch, on="epoch", how="outer")
    if eps_epoch is not None:
        feat = feat.merge(eps_epoch, on="epoch", how="outer")

    data = feat.merge(thr_delta[["epoch", "delta_threshold", "threshold"]], on="epoch", how="inner")
    data = data.sort_values("epoch").reset_index(drop=True)

    data["prev_epoch"] = data["epoch"].shift(1)
    delta = _diff_by_epoch(data.drop(columns=["threshold"], errors="ignore"), key="epoch", prev_key="prev_epoch")
    delta_cols = [c for c in delta.columns if c not in ["epoch", "prev_epoch", "delta_threshold"]]
    delta = delta.rename(columns={c: f"delta_{c}" for c in delta_cols})

    model_df = delta.dropna(subset=["delta_threshold"]).reset_index(drop=True)
    feature_cols = [c for c in model_df.columns if c.startswith("delta_") and c != "delta_threshold"]
    X = model_df[feature_cols].copy().dropna(axis=1, how="all").fillna(0.0)
    y = model_df["delta_threshold"].copy()

    model_df.to_csv(out_dir / "stepC_delta_table.csv", index=False)

    _plot_timeseries(
        thr_delta.dropna(subset=["delta_threshold"]),
        x="epoch",
        y="delta_threshold",
        title="Delta threshold over epochs",
        outpath=fig_dir / "delta_threshold_timeseries.png",
    )

    corr = pd.DataFrame({
        "feature": X.columns,
        "corr_with_delta_threshold": [
            np.corrcoef(X[c].values, y.values)[0, 1] if np.std(X[c].values) > 0 else np.nan
            for c in X.columns
        ],
    }).sort_values("corr_with_delta_threshold", key=lambda s: s.abs(), ascending=False)
    corr.to_csv(out_dir / "stepC_feature_correlations.csv", index=False)

    loo = LeaveOneOut()

    def fit_eval(model, name):
        y_pred = cross_val_predict(model, X, y, cv=loo)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        model.fit(X, y)
        try:
            pim = permutation_importance(model, X, y, n_repeats=50, random_state=0)
            imp = pd.DataFrame({
                "feature": X.columns,
                "perm_importance_mean": pim.importances_mean,
                "perm_importance_std": pim.importances_std,
            }).sort_values("perm_importance_mean", ascending=False)
            imp.to_csv(out_dir / f"stepC_perm_importance_{name}.csv", index=False)
        except Exception as e:
            print(f"[warn] permutation importance failed: {name}: {e}")
        return model, r2, mae

    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=0)
    tree, r2_tree, mae_tree = fit_eval(tree, "tree_depth2")
    rule_text = export_text(tree, feature_names=list(X.columns))
    (out_dir / "stepC_tree_rule.txt").write_text(rule_text, encoding="utf-8")

    rf = RandomForestRegressor(n_estimators=400, max_depth=3, min_samples_leaf=2, random_state=0)
    rf, r2_rf, mae_rf = fit_eval(rf, "rf_depth3")

    summary = pd.DataFrame([
        {"model": "DecisionTreeRegressor(depth=2)", "r2_loo": r2_tree, "mae_loo": mae_tree, "n_samples": len(model_df), "n_features": X.shape[1]},
        {"model": "RandomForestRegressor(depth=3)", "r2_loo": r2_rf, "mae_loo": mae_rf, "n_samples": len(model_df), "n_features": X.shape[1]},
    ])
    summary.to_csv(out_dir / "stepC_model_summary.csv", index=False)

    def plot_pred(y_true, y_pred, title, outpath: Path):
        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel("actual delta_threshold")
        plt.ylabel("predicted delta_threshold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    y_pred_tree = cross_val_predict(DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=0), X, y, cv=loo)
    y_pred_rf = cross_val_predict(RandomForestRegressor(n_estimators=400, max_depth=3, min_samples_leaf=2, random_state=0), X, y, cv=loo)

    plot_pred(y, y_pred_tree, "LOO: Tree depth=2 (delta_threshold)", fig_dir / "pred_vs_actual_tree.png")
    plot_pred(y, y_pred_rf, "LOO: RF depth=3 (delta_threshold)", fig_dir / "pred_vs_actual_rf.png")

    print(f"Saved outputs to: {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
