# step_label_lr_proposal.py
import os
import numpy as np
import pandas as pd

ALIGNED_PATH = "/content/aligned_cluster_by_window_with_agreement.csv"
EVENT_TS_PATH = "/content/threshold_timeseries_with_events.csv"

OUTDIR = "label_lr"
os.makedirs(OUTDIR, exist_ok=True)

# ---- knobs (調整ポイント) ----
USE_LABEL = "majority"        # "majority" 推奨（run1..run5 の多数決ラベル）
MIN_AGREE_FRAC = 0.8          # 意味合わせが十分に一致しているwindowだけ使う
LR_BASE_MULT = 1.0
LR_GAIN = 1.5                 # event率が高いクラスタほど lr を上げる強さ
LR_MIN, LR_MAX = 0.5, 2.0     # 安全クリップ
# ----------------------------

def load_events(event_ts_path: str) -> np.ndarray:
    ts = pd.read_csv(event_ts_path)
    if "is_event" not in ts.columns:
        raise ValueError(f"{event_ts_path} must contain 'is_event' column")
    event_epochs = ts.loc[ts["is_event"].astype(bool), "epoch"].astype(int).values
    return np.unique(event_epochs)

def window_has_event(event_epochs: np.ndarray, s: int, e: int) -> bool:
    # window の定義は必要に応じて調整可能:
    # ここでは epoch in [s, e] に event があれば True
    # np.searchsorted で高速化
    i = np.searchsorted(event_epochs, s, side="left")
    return (i < len(event_epochs)) and (event_epochs[i] <= e)

def main():
    aligned = pd.read_csv(ALIGNED_PATH)
    if USE_LABEL not in aligned.columns:
        raise ValueError(f"{ALIGNED_PATH} must contain column '{USE_LABEL}'")
    if "agree_frac" not in aligned.columns:
        raise ValueError(f"{ALIGNED_PATH} must contain column 'agree_frac'")

    # quality filter
    aligned_f = aligned.loc[aligned["agree_frac"] >= MIN_AGREE_FRAC].copy()
    if len(aligned_f) == 0:
        raise ValueError("No windows left after agree_frac filtering. Lower MIN_AGREE_FRAC.")

    event_epochs = load_events(EVENT_TS_PATH)

    # event window 判定
    ev_flags = []
    for _, r in aligned_f.iterrows():
        s, e = int(r["epoch_start"]), int(r["epoch_end"])
        ev_flags.append(window_has_event(event_epochs, s, e))
    aligned_f["is_event_window"] = ev_flags
    aligned_f["cluster"] = aligned_f[USE_LABEL].astype(int)

    # clusterごとの event率
    grp = (aligned_f
           .groupby("cluster", as_index=False)
           .agg(
               n_windows=("cluster","size"),
               event_windows=("is_event_window","sum"),
               event_rate=("is_event_window","mean"),
               mean_agree=("agree_frac","mean"),
           ))

    global_rate = aligned_f["is_event_window"].mean()

    # lr_mult 提案: lr = base * clip(1 + gain*(rate - global_rate))
    def propose(rate: float) -> float:
        mult = LR_BASE_MULT * (1.0 + LR_GAIN * (rate - global_rate))
        return float(np.clip(mult, LR_MIN, LR_MAX))

    grp["lr_mult"] = grp["event_rate"].apply(propose)

    # windowごとの lr_mult（train.pyでそのまま使える）
    w = aligned_f.merge(grp[["cluster","lr_mult"]], on="cluster", how="left")
    w_out = w[["epoch_start","epoch_end","cluster","agree_frac","is_event_window","lr_mult"]].copy()

    grp.to_csv(os.path.join(OUTDIR, "lr_mult_by_cluster.csv"), index=False)
    w_out.to_csv(os.path.join(OUTDIR, "lr_schedule_by_window.csv"), index=False)

    print("[saved]")
    print(" -", os.path.join(OUTDIR, "lr_mult_by_cluster.csv"))
    print(" -", os.path.join(OUTDIR, "lr_schedule_by_window.csv"))
    print("\n[summary]")
    print("global_event_rate:", global_rate)
    print(grp.sort_values("cluster"))

if __name__ == "__main__":
    main()