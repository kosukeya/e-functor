# stepC_threshold_update.py
# ---------------------------------------------
# Step C: "threshold is reacting to what?"
#   - Load threshold-by-epoch (from previous DT runs)
#   - Join candidate signals from island_env_error / island_profile / (optional) others
#   - Build delta features and explain delta_threshold with:
#       (1) correlations
#       (2) decision tree regressor (interpretable rule)
#       (3) random forest regressor (robust, non-linear)
#   - Save CSV + plots + a readable "rule" summary
# ---------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ----------------------------
# Helpers
# ----------------------------
def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)

def _try_load(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in ["epoch", "prev_epoch", "island"]:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _diff_by_epoch(df: pd.DataFrame, key="epoch", prev_key="prev_epoch"):
    """
    Compute deltas using prev_epoch mapping if available,
    otherwise uses simple sort-by-epoch diff.
    """
    df = df.sort_values(key).reset_index(drop=True)
    if prev_key in df.columns:
        # map prev values by epoch
        prev_map = df.set_index(key)
        delta = df.copy()
        for c in df.columns:
            if c in [key, prev_key]:
                continue
            delta[c] = df.apply(
                lambda r: r[c] - (prev_map.loc[r[prev_key], c] if r[prev_key] in prev_map.index else np.nan),
                axis=1
            )
        return delta
    else:
        delta = df.copy()
        for c in df.columns:
            if c == key:
                continue
            delta[c] = df[c].diff()
        return delta

def _plot_timeseries(df, x, y, title, outpath):
    plt.figure()
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ----------------------------
# Paths
# ----------------------------
THRESH_PATH = "/content/island_dt_by_epoch_err_abs_mean.csv"  # produced earlier
ENVERR_PATH = "/content/runs/island_env_error.csv"
PROFILE_PATH = "/content/runs/island_profile.csv"

# Optional extras if present:
EPS_PATH   = "/content/runs/island_eps_summary.csv"
SUM_PATH   = "/content/runs/island_summary.csv"
DWELL_PATH = "/content/runs/island_dwell.csv"

OUTDIR = "stepC_threshold_update"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# Load threshold series
# ----------------------------
dt = _load_csv(THRESH_PATH)

# Expect columns: epoch, threshold, left_class/right_class, accuracy, ...
# Keep what we need
need_cols = [c for c in ["epoch", "threshold", "left_class", "right_class", "accuracy"] if c in dt.columns]
thr = dt[need_cols].copy().sort_values("epoch").reset_index(drop=True)

# Create prev_epoch for consistent delta computation
thr["prev_epoch"] = thr["epoch"].shift(1)
thr.loc[thr.index[0], "prev_epoch"] = np.nan

thr = _safe_numeric(thr)

# Delta threshold (target)
thr_delta = _diff_by_epoch(thr, key="epoch", prev_key="prev_epoch")
thr_delta = thr_delta.rename(columns={"threshold": "delta_threshold"})
# also keep raw threshold for context
thr_delta["threshold"] = thr["threshold"]

# ----------------------------
# Build candidate feature table at epoch-level
# We want one row per epoch.
# ----------------------------

# 1) env+error stats are already epoch×island.
env = _load_csv(ENVERR_PATH)
env = _safe_numeric(env)

# Use a simple & interpretable aggregation:
# - difference between islands for key signals
# - plus global (weighted) mean across islands
# Here islands are typically 0/1. Works even if more.
def make_island_contrast(df, value_cols, weight_col="n"):
    # wide: epoch x island for each value col
    out = []
    for col in value_cols:
        w = df.pivot(index="epoch", columns="island", values=col)
        w.columns = [f"{col}_island{int(i)}" for i in w.columns]
        out.append(w)
    wide = pd.concat(out, axis=1).reset_index()

    # contrasts for first two islands if they exist
    islands = sorted(df["island"].unique())
    if len(islands) >= 2:
        a, b = islands[0], islands[1]
        for col in value_cols:
            wide[f"{col}_diff_{int(a)}_minus_{int(b)}"] = wide[f"{col}_island{int(a)}"] - wide[f"{col}_island{int(b)}"]
    return wide

# Pick candidate env/error columns (edit freely)
env_cols = [
    # env distribution
    "plant_mean","plant_std","sun_mean","sun_std","water_mean","water_std",
    # error distribution
    "err_abs_mean","err_abs_p90","err_signed_mean"
]
env_cols = [c for c in env_cols if c in env.columns]
env_epoch = make_island_contrast(env, env_cols)

# 2) profile stats (epoch×island) also
prof = _load_csv(PROFILE_PATH)
prof = _safe_numeric(prof)

prof_cols = [
    # output statistics
    "y_mix_mean","y_mix_std","corr_y_mix_true","y_stat_mean","y_sem_mean",
    # counterfactual sensitivity proxy
    "delta_sun_mean","delta_plant_mean","delta_ratio_sun_over_plant",
    # attention summary
    "attn_mass_0_2","attn_mass_3_5","attn_mass_6",
]
prof_cols = [c for c in prof_cols if c in prof.columns]
prof_epoch = make_island_contrast(prof, prof_cols)

# 3) optional eps summary (epoch-level already, or epoch with per-island suffix)
eps = _try_load(EPS_PATH)
eps_epoch = None
if eps is not None:
    eps = _safe_numeric(eps)
    # keep some plausible drivers if present
    keep = [c for c in eps.columns if c in ["epoch","prev_epoch",
                                           "epsilon_all","dM_all","dC_all",
                                           "d_cf_all","d_att_all","d_self_all"]]
    # also keep per-island epsilon_i etc (optional)
    keep += [c for c in eps.columns if c.startswith("epsilon_") or c.startswith("d_cf_") or c.startswith("d_att_") or c.startswith("d_self_")]
    keep = list(dict.fromkeys(keep))
    eps_epoch = eps[keep].copy()
    if "prev_epoch" not in eps_epoch.columns:
        eps_epoch["prev_epoch"] = eps_epoch["epoch"].shift(1)

# ----------------------------
# Merge all features at epoch-level
# ----------------------------
feat = env_epoch.merge(prof_epoch, on="epoch", how="outer")

if eps_epoch is not None:
    feat = feat.merge(eps_epoch, on="epoch", how="outer")

# attach threshold target
data = feat.merge(thr_delta[["epoch","delta_threshold","threshold"]], on="epoch", how="inner")
data = data.sort_values("epoch").reset_index(drop=True)

# Compute delta features for everything except epoch and raw threshold
data["prev_epoch"] = data["epoch"].shift(1)
delta = _diff_by_epoch(data.drop(columns=["threshold"], errors="ignore"), key="epoch", prev_key="prev_epoch")
# rename delta columns
delta_cols = []
for c in delta.columns:
    if c in ["epoch","prev_epoch","delta_threshold"]:
        continue
    delta_cols.append(c)
delta = delta.rename(columns={c: f"delta_{c}" for c in delta_cols})

# Final modeling table: target = delta_threshold, features = delta_*
model_df = delta.copy()
model_df = model_df.dropna(subset=["delta_threshold"]).reset_index(drop=True)

# Build feature matrix
feature_cols = [c for c in model_df.columns if c.startswith("delta_") and c != "delta_threshold"]
X = model_df[feature_cols].copy()
y = model_df["delta_threshold"].copy()

# Drop all-NaN columns and fill remaining NaNs with 0 (small-sample pragmatic)
X = X.dropna(axis=1, how="all")
X = X.fillna(0.0)

# ----------------------------
# Quick diagnostics / outputs
# ----------------------------
model_df.to_csv(os.path.join(OUTDIR, "stepC_delta_table.csv"), index=False)

_plot_timeseries(
    thr_delta.dropna(subset=["delta_threshold"]),
    x="epoch", y="delta_threshold",
    title="Delta threshold over epochs",
    outpath=os.path.join(OUTDIR, "delta_threshold_timeseries.png")
)

# Correlations (with tiny n, use as heuristic)
corr = pd.DataFrame({
    "feature": X.columns,
    "corr_with_delta_threshold": [np.corrcoef(X[c].values, y.values)[0,1] if np.std(X[c].values) > 0 else np.nan for c in X.columns]
}).sort_values("corr_with_delta_threshold", key=lambda s: s.abs(), ascending=False)
corr.to_csv(os.path.join(OUTDIR, "stepC_feature_correlations.csv"), index=False)

# ----------------------------
# Models
# ----------------------------
loo = LeaveOneOut()

def fit_eval(model, name):
    y_pred = cross_val_predict(model, X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # fit on full data for interpretability outputs
    model.fit(X, y)

    # permutation importance (works with fixed X/y lengths)
    try:
        pim = permutation_importance(model, X, y, n_repeats=50, random_state=0)
        imp = pd.DataFrame({
            "feature": X.columns,
            "perm_importance_mean": pim.importances_mean,
            "perm_importance_std": pim.importances_std
        }).sort_values("perm_importance_mean", ascending=False)
        imp.to_csv(os.path.join(OUTDIR, f"stepC_perm_importance_{name}.csv"), index=False)
    except Exception as e:
        print(f"[warn] permutation importance failed: {name}: {e}")

    return model, r2, mae

# 1) small decision tree: "update rule" extraction
tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=0)
tree, r2_tree, mae_tree = fit_eval(tree, "tree_depth2")

rule_text = export_text(tree, feature_names=list(X.columns))
with open(os.path.join(OUTDIR, "stepC_tree_rule.txt"), "w", encoding="utf-8") as f:
    f.write(rule_text)

# 2) random forest: more flexible (still tiny n, so keep small)
rf = RandomForestRegressor(
    n_estimators=400, max_depth=3, min_samples_leaf=2, random_state=0
)
rf, r2_rf, mae_rf = fit_eval(rf, "rf_depth3")

summary = pd.DataFrame([
    {"model":"DecisionTreeRegressor(depth=2)", "r2_loo": r2_tree, "mae_loo": mae_tree, "n_samples": len(model_df), "n_features": X.shape[1]},
    {"model":"RandomForestRegressor(depth=3)", "r2_loo": r2_rf, "mae_loo": mae_rf, "n_samples": len(model_df), "n_features": X.shape[1]},
])
summary.to_csv(os.path.join(OUTDIR, "stepC_model_summary.csv"), index=False)

# Plot predicted vs actual for both
def plot_pred(y_true, y_pred, title, outpath):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel("actual delta_threshold")
    plt.ylabel("predicted delta_threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# LOO preds again for plots
y_pred_tree = cross_val_predict(DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=0), X, y, cv=loo)
y_pred_rf   = cross_val_predict(RandomForestRegressor(n_estimators=400, max_depth=3, min_samples_leaf=2, random_state=0), X, y, cv=loo)

plot_pred(y, y_pred_tree, "LOO: Tree depth=2 (delta_threshold)", os.path.join(OUTDIR, "pred_vs_actual_tree.png"))
plot_pred(y, y_pred_rf,   "LOO: RF depth=3 (delta_threshold)",   os.path.join(OUTDIR, "pred_vs_actual_rf.png"))

print(f"Saved outputs to: {OUTDIR}")
print(summary.to_string(index=False))
print("\n[Tree rule]\n", rule_text[:1500], "...\n")