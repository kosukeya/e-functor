# stepLR_cluster_eventrate.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_threshold_csv(path: Path) -> pd.DataFrame:
    thr = pd.read_csv(path)
    # 必須列チェック
    need = {"epoch", "is_event"}
    missing = need - set(thr.columns)
    if missing:
        raise ValueError(f"threshold csv missing columns: {missing} in {path}")
    thr = thr.copy()
    thr["is_event"] = thr["is_event"].astype(bool)
    return thr

def load_aligned_windows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"epoch_start", "epoch_end"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"aligned windows csv missing columns: {missing} in {path}")

    # cluster列名は出力の揺れがあり得るので吸収
    cluster_col = None
    for cand in ["cluster_aligned", "majority", "run1", "cluster"]:
        if cand in df.columns:
            cluster_col = cand
            break
    if cluster_col is None:
        raise ValueError(
            f"aligned windows csv has no cluster label col. "
            f"expected one of cluster_aligned/majority/run1/cluster. cols={df.columns.tolist()}"
        )

    out = df[["epoch_start", "epoch_end", cluster_col]].rename(columns={cluster_col: "cluster_label"})
    out["epoch_start"] = out["epoch_start"].astype(int)
    out["epoch_end"] = out["epoch_end"].astype(int)
    return out

def map_epoch_to_cluster(thr: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """
    epochがどのwindow(epoch_start..epoch_end)に属するかで cluster_label を付与
    """
    # windowが重ならない前提（あなたのtrajectory window作りはそのはず）
    windows = windows.sort_values(["epoch_start", "epoch_end"]).reset_index(drop=True)

    # epochごとに線形スキャンでも小さいので十分
    labels = []
    w = windows.to_dict("records")

    for e in thr["epoch"].astype(int).tolist():
        lab = None
        for row in w:
            if row["epoch_start"] <= e <= row["epoch_end"]:
                lab = row["cluster_label"]
                break
        labels.append(lab)

    out = thr.copy()
    out["cluster_label"] = labels
    return out

def propose_lr_mult(event_rate: float,
                    min_mult: float = 0.5,
                    max_mult: float = 1.2,
                    mode: str = "conservative") -> float:
    """
    event率→lr倍率のヒューリスティック。
    conservative: eventが多い=不安定/切替頻発 → lr下げる
    aggressive:   eventが多い=学習が必要 → lr上げる
    """
    er = float(np.clip(event_rate, 0.0, 1.0))

    if mode == "conservative":
        # er=0 -> max_mult, er=1 -> min_mult (線形)
        mult = max_mult - (max_mult - min_mult) * er
    elif mode == "aggressive":
        # er=0 -> min_mult, er=1 -> max_mult
        mult = min_mult + (max_mult - min_mult) * er
    else:
        raise ValueError("mode must be conservative or aggressive")

    return float(np.clip(mult, min_mult, max_mult))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="directory containing threshold_timeseries_with_events.csv and aligned_cluster_by_window_with_agreement.csv")
    ap.add_argument("--thr", type=str, default="threshold_timeseries_with_events.csv")
    ap.add_argument("--aligned", type=str, default="aligned_cluster_by_window_with_agreement.csv")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--mode", type=str, default="conservative", choices=["conservative", "aggressive"])
    ap.add_argument("--min_mult", type=float, default=0.5)
    ap.add_argument("--max_mult", type=float, default=1.2)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    thr_path = run_dir / args.thr
    aligned_path = run_dir / args.aligned
    out_dir = Path(args.out_dir) if args.out_dir else run_dir

    thr = load_threshold_csv(thr_path)
    windows = load_aligned_windows(aligned_path)
    joined = map_epoch_to_cluster(thr, windows)

    # cluster別に event率
    g = joined.dropna(subset=["cluster_label"]).groupby("cluster_label")["is_event"]
    summary = g.agg(n_points="count", n_events="sum").reset_index()
    summary["event_rate"] = summary["n_events"] / summary["n_points"]

    summary["lr_mult"] = summary["event_rate"].apply(
        lambda r: propose_lr_mult(r, min_mult=args.min_mult, max_mult=args.max_mult, mode=args.mode)
    )

    # epoch→lr_mult のlookup表も作る（trainで使いやすい）
    lr_map = summary.set_index("cluster_label")["lr_mult"].to_dict()
    joined["lr_mult"] = joined["cluster_label"].map(lr_map)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "lr_mult_by_cluster.csv", index=False)
    joined[["epoch", "cluster_label", "is_event", "lr_mult"]].to_csv(out_dir / "lr_mult_by_epoch.csv", index=False)

    print("[saved]")
    print(" -", out_dir / "lr_mult_by_cluster.csv")
    print(" -", out_dir / "lr_mult_by_epoch.csv")
    print("\n[preview lr_mult_by_cluster]")
    print(summary.sort_values("event_rate", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
# 実行例
# python stepLR_cluster_eventrate.py --run_dir runs_for_analysis/run1 --mode conservative