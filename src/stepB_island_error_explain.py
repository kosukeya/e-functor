# stepB_island_error_explain.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from run_utils import resolve_run_dir, build_run_paths, repo_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--targets", type=str, nargs="*", default=["err_abs_mean", "err_abs_p90", "err_signed_mean"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
        category=UserWarning,
    )

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    env_path = Path(args.env) if args.env else (paths.derived_dir / "island_env_error.csv")
    prof_path = Path(args.profile) if args.profile else (paths.derived_dir / "island_profile.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "stepB_island_error_explain")

    if not env_path.is_absolute():
        env_path = repo_root() / env_path
    if not prof_path.is_absolute():
        prof_path = repo_root() / prof_path
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = pd.read_csv(env_path)
    prof = pd.read_csv(prof_path)

    key = ["epoch", "island", "n"]
    df = env.merge(prof, on=key, how="inner", suffixes=("", "_prof"))

    df["island_cat"] = df["island"].astype(str)
    df["plant_is_zero"] = (df["plant_mean"] == 0.0).astype(int) if "plant_mean" in df.columns else 0
    if "sun_mean" in df.columns and "water_mean" in df.columns:
        df["sun_minus_water_mean"] = df["sun_mean"] - df["water_mean"]
        df["sun_plus_water_mean"] = df["sun_mean"] + df["water_mean"]

    features_no_island = [
        "plant_mean", "plant_std",
        "sun_mean", "sun_std",
        "water_mean", "water_std",
        "plant_is_zero", "sun_minus_water_mean", "sun_plus_water_mean",
        "y_mix_mean", "y_mix_std",
        "y_stat_mean", "y_sem_mean",
        "delta_sun_mean", "delta_plant_mean", "delta_ratio_sun_over_plant",
        "attn_mass_0_2", "attn_mass_3_5", "attn_mass_6",
    ]
    features_with_island = features_no_island + ["island_cat"]

    def keep_existing(cols):
        return [c for c in cols if c in df.columns]

    features_no_island = keep_existing(features_no_island)
    features_with_island = keep_existing(features_with_island)

    def make_preprocess(feature_cols):
        cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), cat_cols),
            ],
            remainder="drop",
        )
        return pre, num_cols, cat_cols

    models = {
        "ridge": Ridge(alpha=1.0, random_state=args.seed),
        "rf": RandomForestRegressor(
            n_estimators=500,
            random_state=args.seed,
            min_samples_leaf=1,
            n_jobs=-1,
        ),
    }

    groups = df["epoch"].values
    n_splits = min(5, df["epoch"].nunique())
    gkf = GroupKFold(n_splits=max(2, n_splits))

    def get_feature_names(preprocess, num_cols, cat_cols):
        names = []
        names += num_cols
        if len(cat_cols) > 0:
            ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
            names += list(ohe.get_feature_names_out(cat_cols))
        return names

    all_results = []
    for target in args.targets:
        if target not in df.columns:
            print(f"[skip] target not found: {target}")
            continue

        for feat_name, feat_cols in [
            ("no_island", features_no_island),
            ("with_island", features_with_island),
        ]:
            X = df[feat_cols].copy()
            y = df[target].astype(float).values

            preprocess, num_cols, cat_cols = make_preprocess(feat_cols)

            for model_name, model in models.items():
                pipe = Pipeline([("pre", preprocess), ("model", model)])
                y_pred = cross_val_predict(pipe, X, y, cv=gkf, groups=groups)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)

                all_results.append({
                    "target": target,
                    "feature_set": feat_name,
                    "model": model_name,
                    "r2_oof": r2,
                    "mae_oof": mae,
                    "n_rows": len(df),
                    "n_epochs": df["epoch"].nunique(),
                    "n_features_raw": len(feat_cols),
                })

                pipe.fit(X, y)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
                            category=UserWarning,
                        )
                        perm = permutation_importance(
                            pipe, X, y,
                            n_repeats=50,
                            random_state=args.seed,
                            n_jobs=-1,
                            scoring="neg_mean_absolute_error",
                        )
                    feat_names = get_feature_names(pipe.named_steps["pre"], num_cols, cat_cols)
                    imp = pd.DataFrame({
                        "feature": feat_names,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std,
                    }).sort_values("importance_mean", ascending=False)
                    imp_path = out_dir / f"perm_importance__{target}__{feat_name}__{model_name}.csv"
                    imp.to_csv(imp_path, index=False)
                except Exception as e:
                    print(f"[warn] permutation importance failed: {target}/{feat_name}/{model_name}: {e}")

                plt.figure()
                plt.scatter(y, y_pred)
                plt.xlabel(f"actual {target}")
                plt.ylabel(f"predicted {target} (OOF)")
                plt.title(f"{target} | {feat_name} | {model_name} | R2={r2:.3f} MAE={mae:.4f}")
                plt.tight_layout()
                plt.savefig(out_dir / f"pred_vs_actual__{target}__{feat_name}__{model_name}.png", dpi=150)
                plt.close()

    res = pd.DataFrame(all_results).sort_values(["target", "r2_oof"], ascending=[True, False])
    res.to_csv(out_dir / "stepB_model_summary.csv", index=False)

    print("Saved outputs to:", out_dir)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
