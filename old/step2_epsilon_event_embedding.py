# step2_epsilon_event_embedding.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root

PREFERRED_PROFILE_COLS = [
    "attn_mean_3", "attn_mean_4", "attn_mean_5",
    "attn_mass_0_2", "attn_mass_3_5",
    "y_mix_std",
    "delta_ratio_sun_over_plant",
]

PREFERRED_ENVERR_COLS = [
    "err_abs_mean", "err_abs_p90",
    "err_signed_mean", "err_signed_min", "err_signed_max",
    "plant_mean", "sun_mean", "water_mean",
]


def pivot_wide(df, value_cols, prefix):
    wide = df.pivot_table(index="epoch", columns="island", values=value_cols, aggfunc="mean")
    wide.columns = [f"{prefix}{col}_island{isl}" for col, isl in wide.columns]
    wide = wide.reset_index()
    for col in value_cols:
        c0 = f"{prefix}{col}_island0"
        c1 = f"{prefix}{col}_island1"
        if c0 in wide.columns and c1 in wide.columns:
            wide[f"{prefix}{col}_diff_0_minus_1"] = wide[c0] - wide[c1]
    return wide


def safe_select_existing(cols, df_cols):
    return [c for c in cols if c in df_cols]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--thr", type=str, default=None)
    ap.add_argument("--events", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--enverr", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--emb-dim", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    thr_path = Path(args.thr) if args.thr else (paths.derived_dir / "threshold_timeseries_with_events.csv")
    events_path = Path(args.events) if args.events else (paths.derived_dir / "threshold_update_events.csv")
    profile_path = Path(args.profile) if args.profile else (paths.derived_dir / "island_profile.csv")
    enverr_path = Path(args.enverr) if args.enverr else (paths.derived_dir / "island_env_error.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "step2_epsilon_event_embedding")

    for p in [thr_path, events_path, profile_path, enverr_path, out_dir]:
        if not p.is_absolute():
            p = repo_root() / p

    if not thr_path.is_absolute():
        thr_path = repo_root() / thr_path
    if not events_path.is_absolute():
        events_path = repo_root() / events_path
    if not profile_path.is_absolute():
        profile_path = repo_root() / profile_path
    if not enverr_path.is_absolute():
        enverr_path = repo_root() / enverr_path
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    thr = pd.read_csv(thr_path)
    events = pd.read_csv(events_path)
    prof = pd.read_csv(profile_path)
    env = pd.read_csv(enverr_path)

    prof_cols = safe_select_existing(PREFERRED_PROFILE_COLS, prof.columns)
    env_cols = safe_select_existing(PREFERRED_ENVERR_COLS, env.columns)

    prof_cols = [c for c in prof_cols if c not in ["epoch", "island", "n"]]
    env_cols = [c for c in env_cols if c not in ["epoch", "island", "n"]]

    if len(prof_cols) == 0 and len(env_cols) == 0:
        raise ValueError("No preferred columns found in profile/enverr CSVs.")

    prof_w = pivot_wide(prof, prof_cols, prefix="prof__")
    env_w = pivot_wide(env, env_cols, prefix="env__")
    feat = prof_w.merge(env_w, on="epoch", how="inner")

    epochs = np.array(sorted(feat["epoch"].unique()))
    def prev_epoch(e):
        idx = np.searchsorted(epochs, e) - 1
        return int(epochs[idx]) if idx >= 0 else None

    ev_epochs = sorted(events["epoch"].astype(int).unique().tolist())

    rows = []
    feature_cols = [c for c in feat.columns if c != "epoch"]
    feat_idx = feat.set_index("epoch")

    for e in ev_epochs:
        pre = prev_epoch(e)
        if pre is None or pre not in feat_idx.index or e not in feat_idx.index:
            continue

        x_pre = feat_idx.loc[pre, feature_cols]
        x_post = feat_idx.loc[e, feature_cols]
        x_delta = x_post - x_pre

        ev_row = events.loc[events["epoch"].astype(int) == e].iloc[0].to_dict()

        rows.append({
            "event_epoch": e,
            "pre_epoch": pre,
            "event_sign": int(ev_row.get("event_sign", 0)),
            "threshold_prev": float(ev_row.get("threshold_prev", np.nan)),
            "threshold": float(ev_row.get("threshold", np.nan)),
            "threshold_diff": float(ev_row.get("threshold_diff", np.nan)),
            "_x_pre": x_pre.to_dict(),
            "_x_delta": x_delta.to_dict(),
        })

    ev_df = pd.DataFrame(rows)
    if len(ev_df) == 0:
        raise ValueError("No event rows could be built from inputs.")

    X_pre = pd.DataFrame(ev_df["_x_pre"].tolist()).fillna(0.0)
    X_delta = pd.DataFrame(ev_df["_x_delta"].tolist()).fillna(0.0)

    X_pre_plus_delta = pd.concat(
        [X_pre.add_prefix("Xprev__"), X_delta.add_prefix("D__")],
        axis=1,
    )

    scaler = StandardScaler()
    Z = scaler.fit_transform(X_pre_plus_delta.values)

    use_pca = True
    pca_skip_reason = None
    if Z.shape[0] < 2:
        use_pca = False
        pca_skip_reason = "n_samples < 2"
    elif np.allclose(Z.var(axis=0), 0.0):
        use_pca = False
        pca_skip_reason = "zero variance features"

    if use_pca:
        pca = PCA(n_components=min(args.emb_dim, Z.shape[1], Z.shape[0]), random_state=args.seed)
        E = pca.fit_transform(Z)
        explained_var = pca.explained_variance_ratio_.tolist()
    else:
        pca = None
        emb_dim = min(args.emb_dim, Z.shape[1])
        E = Z[:, :emb_dim]
        explained_var = []

    emb_cols = [f"emb_{i}" for i in range(E.shape[1])]
    out = ev_df[["pre_epoch", "event_epoch", "event_sign", "threshold_prev", "threshold", "threshold_diff"]].copy()
    for i, c in enumerate(emb_cols):
        out[c] = E[:, i]

    out.to_csv(out_dir / "epsilon_event_embeddings.csv", index=False)

    X_pre_only = X_pre.add_prefix("Xprev__")
    X_pre_only.insert(0, "pre_epoch", ev_df["pre_epoch"].values)
    X_pre_only.insert(1, "event_epoch", ev_df["event_epoch"].values)
    X_pre_only.to_csv(out_dir / "epsilon_event_features_pre_only.csv", index=False)

    X_pre_plus_delta.insert(0, "pre_epoch", ev_df["pre_epoch"].values)
    X_pre_plus_delta.insert(1, "event_epoch", ev_df["event_epoch"].values)
    X_pre_plus_delta.to_csv(out_dir / "epsilon_event_features_pre_plus_delta.csv", index=False)

    meta = {
        "embedding": {
            "method": "StandardScaler + PCA" if use_pca else "StandardScaler (PCA skipped)",
            "emb_dim": int(E.shape[1]),
            "explained_variance_ratio": explained_var,
            "pca_skipped": (not use_pca),
            "pca_skip_reason": pca_skip_reason,
        },
        "features": {
            "profile_cols_used": prof_cols,
            "enverr_cols_used": env_cols,
            "wide_columns_count": int(X_pre.shape[1]),
            "pre_plus_delta_columns_count": int(X_pre_plus_delta.shape[1] - 2),
        },
    }
    with open(out_dir / "epsilon_event_embedding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved to: {out_dir}")
    print(f"[ok] events: {len(out)} | emb_dim={E.shape[1]}")


if __name__ == "__main__":
    main()
