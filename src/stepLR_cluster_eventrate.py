# stepLR_cluster_eventrate.py
# 固定I/O仕様:
#   input : runs/<RUN_ID>/alpha_log.csv
#   output: runs/<RUN_ID>/lr_mult_by_epoch.csv
#           runs/<RUN_ID>/lr_mult_by_cluster.csv
#
# 実行例:
#   RUN_ID=test1 python stepLR_cluster_eventrate.py
#   python stepLR_cluster_eventrate.py --run_id test1
#   python stepLR_cluster_eventrate.py --run_dir runs/test1
#
# 仕様（この実装を公式仕様として固定する前提）
# 1) 必須列（alpha_log.csv）:
#    - epoch (int)
#    - alpha_used (float)  : そのepochでforwardに使ったalpha
#    - alpha (float)       : そのepochのepsilon gateで更新された「次epoch向けalpha」
#    - epsilon (float)     : モデル差分指標（符号つきでも可）
#    ※他の列(dC等)は任意。あれば補助的に使えるが、この完成版では必須にしない。
#
# 2) cluster_label の定義（alpha_used による3値ビン固定）:
#    - 0: alpha_used < 0.60
#    - 1: 0.60 <= alpha_used < 0.90
#    - 2: alpha_used >= 0.90
#
# 3) is_event の判定（epochごとの“変化イベント”）:
#    - delta_alpha = |alpha - alpha_used|
#    - is_event = (|epsilon| >= eps_thr) OR (delta_alpha >= alpha_thr)
#    デフォルト: eps_thr=0.10, alpha_thr=0.05
#    ※epoch=0などで epsilon/alpha が欠損の行は is_event=False 扱い
#
# 4) lr_mult の計算（clusterごとの event_rate を使う）:
#    - cluster別に event_rate = (#is_event True) / (points)
#    - propose_lr_mult(event_rate) で lr_mult を決定
#      conservative: event多い=不安定 → lrを下げる
#      aggressive  : event多い=学習必要 → lrを上げる
#    - lr_mult は cluster毎に一定値、各epochへ付与
#
# 注意:
# - lr_mult_by_epoch.csv は alpha_log に存在する epoch（通常LOG_EVERY刻み）だけ出します。
#   train側は forward-fill で運用する想定（あなたの train.py の safe_lr_mult と整合）。

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from run_utils import resolve_run_dir, build_run_paths


def propose_lr_mult(
    event_rate: float,
    min_mult: float = 0.5,
    max_mult: float = 1.2,
    mode: str = "conservative",
) -> float:
    """
    event率→lr倍率のヒューリスティック。
    conservative: eventが多い=不安定/切替頻発 → lr下げる
    aggressive:   eventが多い=学習が必要       → lr上げる
    """
    er = float(np.clip(event_rate, 0.0, 1.0))
    if mode == "conservative":
        mult = max_mult - (max_mult - min_mult) * er  # er=0 -> max, er=1 -> min
    elif mode == "aggressive":
        mult = min_mult + (max_mult - min_mult) * er  # er=0 -> min, er=1 -> max
    else:
        raise ValueError("mode must be conservative or aggressive")
    return float(np.clip(mult, min_mult, max_mult))


def load_alpha_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"alpha_log not found: {path}")

    df = pd.read_csv(path)

    # 必須列（固定仕様）
    need = {"epoch", "alpha_used", "alpha", "epsilon"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"alpha_log missing columns: {missing} in {path}")

    out = df.copy()
    out["epoch"] = out["epoch"].astype(int)

    # 数値化（壊れている行は NaN にして後で安全処理）
    for c in ["alpha_used", "alpha", "epsilon"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("epoch").reset_index(drop=True)
    return out

def ensure_epoch0_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    lr_mult_by_epoch の事故防止:
    alpha_log に epoch=0 が無い場合は自動で補完する。
    - alpha_used/alpha/epsilon は NaN にしておき、後段の is_event=False 扱いに落ちる
    - lr_mult は最終的に 1.0（欠損→1.0 ルール）になる
    """
    if len(df) == 0:
        # 空なら epoch=0 だけ作る
        return pd.DataFrame([{"epoch": 0, "alpha_used": np.nan, "alpha": np.nan, "epsilon": np.nan}])

    if (df["epoch"] == 0).any():
        return df

    row0 = {"epoch": 0, "alpha_used": np.nan, "alpha": np.nan, "epsilon": np.nan}
    df2 = pd.concat([pd.DataFrame([row0]), df], ignore_index=True)
    df2 = df2.sort_values("epoch").reset_index(drop=True)
    return df2

def assign_cluster(alpha_used: pd.Series) -> pd.Series:
    """
    固定仕様: alpha_used を 3ビンに分ける（0/1/2）
      0: <0.60, 1: [0.60,0.90), 2: >=0.90
    """
    bins = [-np.inf, 0.60, 0.90, np.inf]
    labels = [0, 1, 2]
    cl = pd.cut(alpha_used, bins=bins, labels=labels, right=False, include_lowest=True)
    # cl は Categorical なので int に直す（欠損は NaN のまま）
    return cl.astype("float")  # いったんfloat（NaN保持）→最後にInt化


def compute_events(df: pd.DataFrame, eps_thr: float, alpha_thr: float) -> pd.Series:
    """
    固定仕様: is_event = (|epsilon|>=eps_thr) OR (|alpha-alpha_used|>=alpha_thr)
    欠損は False
    """
    eps = df["epsilon"].astype(float)
    au = df["alpha_used"].astype(float)
    a = df["alpha"].astype(float)

    delta_alpha = (a - au).abs()
    is_event = (eps.abs() >= eps_thr) | (delta_alpha >= alpha_thr)

    # 欠損を False 扱い
    is_event = is_event.fillna(False).astype(bool)
    return is_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default=None, help="RUN_ID (runs/<RUN_ID>)")
    ap.add_argument("--run_dir", type=str, default=None, help="run directory path (e.g. runs/test1)")
    ap.add_argument("--mode", type=str, default="conservative", choices=["conservative", "aggressive"])
    ap.add_argument("--min_mult", type=float, default=0.5)
    ap.add_argument("--max_mult", type=float, default=1.2)

    # イベント判定閾値（固定仕様のパラメータとして外出し）
    ap.add_argument("--eps_thr", type=float, default=0.10, help="event if |epsilon| >= eps_thr")
    ap.add_argument("--alpha_thr", type=float, default=0.05, help="event if |alpha-alpha_used| >= alpha_thr")
    ap.add_argument("--min_points", type=int, default=3,
                help="clusters with < min_points will be lr_mult=1.0 (guard)")

    args = ap.parse_args()

    # run_dir 決定（固定I/O）
    run_id = args.run_id or os.environ.get("RUN_ID")
    run_dir_arg = args.run_dir or os.environ.get("RUN_DIR")
    run_dir, run_id = resolve_run_dir(run_dir_arg, run_id)
    paths = build_run_paths(run_dir, run_id)

    alpha_log_path = paths.alpha_log
    out_epoch_path = paths.run_dir / "lr_mult_by_epoch.csv"
    out_cluster_path = paths.run_dir / "lr_mult_by_cluster.csv"

    df = load_alpha_log(alpha_log_path)
    df = ensure_epoch0_row(df)

    # cluster_label（固定仕様: alpha_used 3ビン）
    df["cluster_label"] = assign_cluster(df["alpha_used"])

    # is_event（固定仕様）
    df["is_event"] = compute_events(df, eps_thr=args.eps_thr, alpha_thr=args.alpha_thr)

    # cluster別 event率 → lr_mult
    joined = df.dropna(subset=["cluster_label"]).copy()
    joined["cluster_label"] = joined["cluster_label"].astype(int)

    g = joined.groupby("cluster_label")["is_event"]
    summary = g.agg(n_points="count", n_events="sum").reset_index()
    summary["event_rate"] = summary["n_events"] / summary["n_points"]

    summary["lr_mult"] = summary["event_rate"].apply(
        lambda r: propose_lr_mult(
            r,
            min_mult=args.min_mult,
            max_mult=args.max_mult,
            mode=args.mode,
        )
    )
    # ★min_points ガード：点が少ないクラスタは lr を触らない
    summary["guarded"] = summary["n_points"] < int(args.min_points)
    summary.loc[summary["guarded"], "lr_mult"] = 1.0

    # epoch行へ付与
    lr_map = summary.set_index("cluster_label")["lr_mult"].to_dict()
    df["lr_mult"] = df["cluster_label"].map(lr_map)

    # 欠損（clusterに入らない/alpha_used欠損など）は mult=1.0 に倒す（train側defaultと整合）
    df["lr_mult"] = pd.to_numeric(df["lr_mult"], errors="coerce").fillna(1.0)

    # 出力
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_cluster_path, index=False)

    out = df[["epoch", "cluster_label", "is_event", "lr_mult"]].copy()
    # cluster_label は欠損があり得るので、そのまま出す（CSVで空欄になる）
    out["epoch"] = out["epoch"].astype(int)
    out.to_csv(out_epoch_path, index=False)

    print("[saved]")
    print(" -", out_cluster_path)
    print(" -", out_epoch_path)
    print("\n[preview lr_mult_by_cluster]")
    if len(summary) == 0:
        print("  (no clusters found; check alpha_used column)")
    else:
        print(summary.sort_values("event_rate", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
