# step2_epsilon_event_embedding.py
import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============== Config ==============
PATH_THR = "/content/threshold_timeseries_with_events.csv"
PATH_EVENTS = "/content/threshold_update_events.csv"
PATH_PROFILE = "/content/island_profile.csv"
PATH_ENVERR = "/content/island_env_error.csv"

OUTDIR = "step2_epsilon_event_embedding"
EMB_DIM = 6  # 少数でOK（まずは 4〜8 推奨）

# まずは「安定して効いていそう」なプロファイル特徴を優先（あなたの安定性結果を反映）
# ※列が存在しない場合は自動的にスキップされます
PREFERRED_PROFILE_COLS = [
    "attn_mean_3", "attn_mean_4", "attn_mean_5",
    "attn_mass_0_2", "attn_mass_3_5",
    "y_mix_std",
    "delta_ratio_sun_over_plant",
]

# env/errorは「補助」扱い（必要になったら増やす）
PREFERRED_ENVERR_COLS = [
    "err_abs_mean", "err_abs_p90",
    "err_signed_mean", "err_signed_min", "err_signed_max",
    "plant_mean", "sun_mean", "water_mean",
]

# ====================================

def pivot_wide(df, value_cols, prefix):
    """
    epoch×island の縦持ちを wide 化し、
    island0/island1 と diff_0_minus_1 を作る。
    """
    wide = df.pivot_table(index="epoch", columns="island", values=value_cols, aggfunc="mean")
    wide.columns = [f"{prefix}{col}_island{isl}" for col, isl in wide.columns]
    wide = wide.reset_index()

    # diff を追加
    for col in value_cols:
        c0 = f"{prefix}{col}_island0"
        c1 = f"{prefix}{col}_island1"
        if c0 in wide.columns and c1 in wide.columns:
            wide[f"{prefix}{col}_diff_0_minus_1"] = wide[c0] - wide[c1]
    return wide

def safe_select_existing(cols, df_cols):
    return [c for c in cols if c in df_cols]

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    thr = pd.read_csv(PATH_THR)
    events = pd.read_csv(PATH_EVENTS)
    prof = pd.read_csv(PATH_PROFILE)
    env = pd.read_csv(PATH_ENVERR)

    # --- 1) 使う列を選ぶ（存在するものだけ）
    prof_cols = safe_select_existing(PREFERRED_PROFILE_COLS, prof.columns)
    env_cols  = safe_select_existing(PREFERRED_ENVERR_COLS, env.columns)

    prof_cols = [c for c in prof_cols if c not in ["epoch","island","n"]]
    env_cols  = [c for c in env_cols  if c not in ["epoch","island","n"]]

    if len(prof_cols) == 0 and len(env_cols) == 0:
        raise ValueError("選択した特徴が0です。CSV列名が想定と違う可能性があります。")

    # --- 2) wide化（epoch→特徴ベクトル）
    prof_w = pivot_wide(prof, prof_cols, prefix="prof__")
    env_w  = pivot_wide(env,  env_cols,  prefix="env__")

    feat = prof_w.merge(env_w, on="epoch", how="inner")

    # --- 3) イベントテーブル作成：event_epoch の「直前epoch」を取る
    epochs = np.array(sorted(feat["epoch"].unique()))
    def prev_epoch(e):
        idx = np.searchsorted(epochs, e) - 1
        if idx < 0: return None
        return int(epochs[idx])

    # event epoch の一覧（Trueの行だけでも良いが、events.csv を優先して確実に）
    ev_epochs = sorted(events["epoch"].astype(int).unique().tolist())

    rows = []
    feature_cols = [c for c in feat.columns if c != "epoch"]

    feat_idx = feat.set_index("epoch")

    for e in ev_epochs:
        pre = prev_epoch(e)
        if pre is None or pre not in feat_idx.index or e not in feat_idx.index:
            continue

        # pre/post
        x_pre  = feat_idx.loc[pre, feature_cols]
        x_post = feat_idx.loc[e,   feature_cols]
        x_delta = x_post - x_pre

        ev_row = events.loc[events["epoch"].astype(int) == e].iloc[0].to_dict()

        rows.append({
            "event_epoch": e,
            "pre_epoch": pre,
            "event_sign": int(ev_row.get("event_sign", 0)),
            "threshold_prev": float(ev_row.get("threshold_prev", np.nan)),
            "threshold": float(ev_row.get("threshold", np.nan)),
            "threshold_diff": float(ev_row.get("threshold_diff", np.nan)),
            # 後で展開する
            "_x_pre": x_pre.to_dict(),
            "_x_delta": x_delta.to_dict(),
        })

    ev_df = pd.DataFrame(rows)
    if len(ev_df) == 0:
        raise ValueError("イベントが抽出できませんでした。入力CSVを確認してください。")

    # --- 4) 2用途の行列を作る
    X_pre = pd.DataFrame(ev_df["_x_pre"].tolist())
    X_delta = pd.DataFrame(ev_df["_x_delta"].tolist())

    # 欠損処理（最小：0埋め。必要なら後で改善）
    X_pre = X_pre.fillna(0.0)
    X_delta = X_delta.fillna(0.0)

    X_pre_plus_delta = pd.concat(
        [X_pre.add_prefix("Xprev__"), X_delta.add_prefix("D__")],
        axis=1
    )

    # --- 5) embedding（標準化→PCA）
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_pre_plus_delta.values)

    pca = PCA(n_components=min(EMB_DIM, Z.shape[1], Z.shape[0]))
    E = pca.fit_transform(Z)

    # --- 6) 出力
    emb_cols = [f"emb_{i}" for i in range(E.shape[1])]
    out = ev_df[["pre_epoch","event_epoch","event_sign","threshold_prev","threshold","threshold_diff"]].copy()
    for i,c in enumerate(emb_cols):
        out[c] = E[:, i]

    out.to_csv(os.path.join(OUTDIR, "epsilon_event_embeddings.csv"), index=False)

    # 人間可読（元特徴）も保存
    X_pre_only = X_pre.add_prefix("Xprev__")
    X_pre_only.insert(0, "pre_epoch", ev_df["pre_epoch"].values)
    X_pre_only.insert(1, "event_epoch", ev_df["event_epoch"].values)
    X_pre_only.to_csv(os.path.join(OUTDIR, "epsilon_event_features_pre_only.csv"), index=False)

    X_pre_plus_delta.insert(0, "pre_epoch", ev_df["pre_epoch"].values)
    X_pre_plus_delta.insert(1, "event_epoch", ev_df["event_epoch"].values)
    X_pre_plus_delta.to_csv(os.path.join(OUTDIR, "epsilon_event_features_pre_plus_delta.csv"), index=False)

    meta = {
        "embedding": {
            "method": "StandardScaler + PCA",
            "emb_dim": int(E.shape[1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "features": {
            "profile_cols_used": prof_cols,
            "enverr_cols_used": env_cols,
            "wide_columns_count": int(X_pre.shape[1]),
            "pre_plus_delta_columns_count": int(X_pre_plus_delta.shape[1]-2),
        },
        "notes": [
            "予測用途は epsilon_event_features_pre_only.csv を使う（未来情報を含めない）",
            "記憶/説明用途は pre_plus_delta も使える（起きたことの要約）",
        ]
    }
    with open(os.path.join(OUTDIR, "epsilon_event_embedding_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved to: {OUTDIR}")
    print(f"[ok] events: {len(out)} | emb_dim={E.shape[1]}")
    print("[ok] files:")
    print(" - epsilon_event_embeddings.csv")
    print(" - epsilon_event_features_pre_only.csv")
    print(" - epsilon_event_features_pre_plus_delta.csv")
    print(" - epsilon_event_embedding_meta.json")

if __name__ == "__main__":
    main()