# detect_critical.py
# Usage:
#   python detect_critical.py runs/C4_noCF_s0/alpha_log.csv
# Options:
#   python detect_critical.py runs/.../alpha_log.csv --baseline 200 600 --z_hard 4 --z_soft 3

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


EPS = 1e-12


def robust_stats(x: np.ndarray) -> Tuple[float, float]:
    """Return (median, robust_sigma) where robust_sigma ~ std using MAD."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    robust_sigma = 1.4826 * (mad + EPS)
    return med, robust_sigma


def robust_z(x: np.ndarray, med: float, sigma: float) -> np.ndarray:
    if not np.isfinite(med) or not np.isfinite(sigma) or sigma <= 0:
        return np.full_like(x, np.nan, dtype=float)
    return (x - med) / sigma


def consecutive_true(mask: np.ndarray, k: int = 2) -> np.ndarray:
    """mask[i] True if there are k consecutive True ending at i."""
    if k <= 1:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= k:
            out[i] = True
    return out


def two_of_three(mask: np.ndarray) -> np.ndarray:
    """mask[i] True if among {i-2,i-1,i} at least 2 are True."""
    out = np.zeros_like(mask, dtype=bool)
    for i in range(mask.size):
        j0 = max(0, i - 2)
        window = mask[j0:i + 1]
        out[i] = (window.sum() >= 2)
    return out


@dataclass
class EventHit:
    epoch: int
    kind: str
    score: float  # primary z-score (or combined)
    reason: str


def pick_baseline(df: pd.DataFrame, b0: int, b1: int) -> pd.DataFrame:
    base = df[(df["epoch"] >= b0) & (df["epoch"] <= b1)].copy()
    return base


def compute_diff_series(x: np.ndarray) -> np.ndarray:
    """First difference with NaN at first element."""
    d = np.empty_like(x, dtype=float)
    d[:] = np.nan
    d[1:] = x[1:] - x[:-1]
    return d


def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def detect_env_break(df: pd.DataFrame, base: pd.DataFrame, z_hard: float, z_soft: float) -> Optional[EventHit]:
    """
    Env-break:
      primary: d_att z > z_hard (2 consecutive) OR z > z_soft (2/3)
      confirm: |Δ env_sum_tr| OR |Δ I_sum_tr| OR |Δ H_I_proxy| large
    Note: alpha_log.csv doesn't contain H_I directly; we approximate confirmation using ΔI_sum_tr or Δenv_sum_tr,
          and optionally Δcorr_val (drop) as a weak proxy.
    """
    require_cols(df, ["epoch", "d_att", "env_sum_tr", "I_sum_tr", "corr_val"])

    x = df["d_att"].to_numpy(dtype=float)
    xb = base["d_att"].to_numpy(dtype=float)
    med, sig = robust_stats(xb)
    z = robust_z(x, med, sig)

    hard = consecutive_true(z > z_hard, k=2)
    soft = two_of_three(z > z_soft)

    # confirmations based on diffs
    env = df["env_sum_tr"].to_numpy(dtype=float)
    Is  = df["I_sum_tr"].to_numpy(dtype=float)
    corr = df["corr_val"].to_numpy(dtype=float)

    d_env = np.abs(compute_diff_series(env))
    d_I   = np.abs(compute_diff_series(Is))
    d_corr_drop = np.maximum(0.0, -compute_diff_series(corr))  # only drops

    # thresholds from baseline diffs (robust)
    d_env_b = np.abs(compute_diff_series(base["env_sum_tr"].to_numpy(dtype=float)))
    d_I_b   = np.abs(compute_diff_series(base["I_sum_tr"].to_numpy(dtype=float)))
    d_cd_b  = np.maximum(0.0, -compute_diff_series(base["corr_val"].to_numpy(dtype=float)))

    med_de, sig_de = robust_stats(d_env_b)
    med_dI, sig_dI = robust_stats(d_I_b)
    med_dc, sig_dc = robust_stats(d_cd_b)

    z_de = robust_z(d_env, med_de, sig_de)
    z_dI = robust_z(d_I, med_dI, sig_dI)
    z_dc = robust_z(d_corr_drop, med_dc, sig_dc)

    confirm = (z_de > 4.0) | (z_dI > 4.0) | (z_dc > 3.0)  # corr drop is weaker

    trigger = (hard | soft) & confirm

    if not np.any(trigger):
        return None

    i = int(np.argmax(trigger))  # first True
    epoch = int(df.loc[i, "epoch"])
    reason = f"d_att z={z[i]:.2f}, confirm(z|Δenv|={z_de[i]:.2f}, z|ΔI|={z_dI[i]:.2f}, zΔcorr_drop={z_dc[i]:.2f})"
    return EventHit(epoch=epoch, kind="env_break", score=float(z[i]), reason=reason)


def detect_self_break(df: pd.DataFrame, base: pd.DataFrame, z_hard: float, z_soft: float,
                      self_target: float) -> Optional[EventHit]:
    """
    Self-break:
      primary: self_err = |self_m_tr - self_target| has z > z_hard (2 consecutive) OR z > z_soft (2/3)
              OR absolute threshold self_err > max(self_target*0.8, med + 4σ)
      confirm: d_self z > 3 (nearby)
    """
    require_cols(df, ["epoch", "self_m_tr", "d_self"])

    self_m = df["self_m_tr"].to_numpy(dtype=float)
    self_err = np.abs(self_m - float(self_target))

    self_m_b = base["self_m_tr"].to_numpy(dtype=float)
    self_err_b = np.abs(self_m_b - float(self_target))

    med, sig = robust_stats(self_err_b)
    z = robust_z(self_err, med, sig)

    hard = consecutive_true(z > z_hard, k=2)
    soft = two_of_three(z > z_soft)

    # absolute gate
    abs_gate = self_err > max(float(self_target) * 0.8, med + 4.0 * sig)

    primary = hard | soft | abs_gate

    # confirm with d_self
    dself = df["d_self"].to_numpy(dtype=float)
    dself_b = base["d_self"].to_numpy(dtype=float)
    med_ds, sig_ds = robust_stats(dself_b)
    z_ds = robust_z(dself, med_ds, sig_ds)
    confirm = z_ds > 3.0

    # allow confirm within +/-1 log step
    confirm_near = confirm.copy()
    confirm_near[:-1] |= confirm[1:]
    confirm_near[1:]  |= confirm[:-1]

    trigger = primary & confirm_near
    if not np.any(trigger):
        return None

    i = int(np.argmax(trigger))
    epoch = int(df.loc[i, "epoch"])
    reason = f"self_err={self_err[i]:.4f} (target={self_target}), z(self_err)={z[i]:.2f}, z(d_self)={z_ds[i]:.2f}"
    score = float(max(z[i], z_ds[i]))
    return EventHit(epoch=epoch, kind="self_break", score=score, reason=reason)


def detect_rule_break(df: pd.DataFrame, base: pd.DataFrame, z_hard: float, z_soft: float) -> Optional[EventHit]:
    """
    Rule-break:
      primary: (d_cf + d_mono) z > z_hard (2 consecutive) OR z > z_soft (2/3)
           OR |Δ ratio_s_proxy| large (we don't have ratio_s in alpha_log.csv columns)
      confirm: corr_val drop OR Val_sem spike (optional)
    Notes:
      - alpha_log.csv provided columns do not include ratio_s/gap_s. So we focus on d_cf + d_mono.
      - confirm uses corr_val drop and/or Val_sem rise.
    """
    require_cols(df, ["epoch", "d_cf", "d_mono", "corr_val", "Val_sem"])

    x = (df["d_cf"].to_numpy(dtype=float) + df["d_mono"].to_numpy(dtype=float))
    xb = (base["d_cf"].to_numpy(dtype=float) + base["d_mono"].to_numpy(dtype=float))
    med, sig = robust_stats(xb)
    z = robust_z(x, med, sig)

    hard = consecutive_true(z > z_hard, k=2)
    soft = two_of_three(z > z_soft)
    primary = hard | soft

    # confirmations
    corr = df["corr_val"].to_numpy(dtype=float)
    val_sem = df["Val_sem"].to_numpy(dtype=float)

    corr_drop = np.maximum(0.0, -compute_diff_series(corr))
    vs_rise = np.maximum(0.0, compute_diff_series(val_sem))

    corr_drop_b = np.maximum(0.0, -compute_diff_series(base["corr_val"].to_numpy(dtype=float)))
    vs_rise_b = np.maximum(0.0, compute_diff_series(base["Val_sem"].to_numpy(dtype=float)))

    med_cd, sig_cd = robust_stats(corr_drop_b)
    med_vs, sig_vs = robust_stats(vs_rise_b)

    z_cd = robust_z(corr_drop, med_cd, sig_cd)
    z_vs = robust_z(vs_rise, med_vs, sig_vs)

    confirm = (z_cd > 3.0) | (z_vs > 4.0)

    trigger = primary & confirm
    if not np.any(trigger):
        return None

    i = int(np.argmax(trigger))
    epoch = int(df.loc[i, "epoch"])
    reason = f"z(d_cf+d_mono)={z[i]:.2f}, confirm(zΔcorr_drop={z_cd[i]:.2f}, zΔVal_sem_rise={z_vs[i]:.2f})"
    return EventHit(epoch=epoch, kind="rule_break", score=float(z[i]), reason=reason)


def pick_first_event(events: List[EventHit]) -> Optional[EventHit]:
    if not events:
        return None
    # earliest epoch wins; tie-break: self > env > rule; then higher score
    prio = {"self_break": 0, "env_break": 1, "rule_break": 2}
    events_sorted = sorted(
        events,
        key=lambda e: (e.epoch, prio.get(e.kind, 9), -e.score),
    )
    return events_sorted[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str, help="Path to alpha_log.csv")
    ap.add_argument("--baseline", type=int, nargs=2, default=[200, 600], metavar=("B0", "B1"))
    ap.add_argument("--baseline_rows", type=int, default=5,
                help="Number of first rows to use as baseline (robust for sparse logs).")
    ap.add_argument("--z_hard", type=float, default=4.0)
    ap.add_argument("--z_soft", type=float, default=3.0)
    ap.add_argument("--self_target", type=float, default=None,
                    help="If omitted, try to infer from early stable self_m_tr (fallback).")
    ap.add_argument("--print_head", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)

    # Ensure numeric columns are numeric (strings or blanks can appear)
    # - Convert what can be converted; keep genuinely text columns as-is.
    TEXT_COLS = {"eps_mode"}  # add more if needed

    for c in df.columns:
        if c in TEXT_COLS:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # If an entire column becomes NaN, it's likely non-numeric -> keep original
        if not s.isna().all():
            df[c] = s

    require_cols(df, ["epoch"])
    df = df.sort_values("epoch").reset_index(drop=True)

    if args.print_head:
        print(df.head(10).to_string(index=False))

    b0, b1 = args.baseline
    K = int(args.baseline_rows)

    # 1) try epoch-window baseline (backward compatible)
    base_win = pick_baseline(df, b0, b1)

    # 2) robust default: first K rows
    base_first = df.iloc[:K].copy()

    # Choose baseline:
    # - If window baseline has enough rows, use it.
    # - Otherwise fall back to first-K baseline (works with sparse logs like every-200 epochs).
    if len(base_win) >= K:
        base = base_win
        baseline_desc = f"epoch in [{b0},{b1}]"
    else:
        base = base_first
        if len(base_win) > 0:
            print(f"[WARN] Baseline window too small (epochs [{b0},{b1}] -> {len(base_win)} rows). "
                  f"Falling back to first {K} rows baseline.")
        else:
            print(f"[WARN] Baseline window empty (epochs [{b0},{b1}]). "
                  f"Using first {K} rows baseline.")
        baseline_desc = f"first {K} rows"

    if len(base) < 3:
        raise ValueError(f"alpha_log too small for baseline: need >=3 rows, got {len(base)}.")


    # infer self_target if not provided:
    # - since alpha_log.csv doesn't contain 'self_target' column, we infer it:
    #   use median of self_m_tr in baseline as "target-ish" (works if self target was enforced strongly).
    if args.self_target is None:
        if "self_m_tr" in df.columns:
            # use baseline, but if baseline is first-K rows, consider using a slightly later slice
            inferred = float(np.median(base["self_m_tr"].to_numpy(dtype=float)))
            self_target = inferred
        else:
            self_target = 0.03
    else:
        self_target = float(args.self_target)

    z_hard = float(args.z_hard)
    z_soft = float(args.z_soft)

    events: List[EventHit] = []

    env_hit = detect_env_break(df, base, z_hard=z_hard, z_soft=z_soft)
    if env_hit:
        events.append(env_hit)

    self_hit = detect_self_break(df, base, z_hard=z_hard, z_soft=z_soft, self_target=self_target)
    if self_hit:
        events.append(self_hit)

    rule_hit = detect_rule_break(df, base, z_hard=z_hard, z_soft=z_soft)
    if rule_hit:
        events.append(rule_hit)

    print("==== Criticality Detector ====")
    print(f"CSV: {args.csv_path}")
    print(f"Baseline: {baseline_desc} (n={len(base)})")
    print(f"z_hard={z_hard} (2 consecutive), z_soft={z_soft} (2 of 3)")
    print(f"self_target used: {self_target:.6f} (inferred={args.self_target is None})")
    print()

    if not events:
        print("No break detected under current thresholds.")
        return

    for e in sorted(events, key=lambda x: x.epoch):
        print(f"- {e.kind}: epoch={e.epoch}, score={e.score:.2f}")
        print(f"  reason: {e.reason}")

    first = pick_first_event(events)
    print()
    print("---- FIRST BREAK (critical epoch) ----")
    print(f"kind={first.kind}  epoch={first.epoch}  score={first.score:.2f}")
    print(f"reason: {first.reason}")


if __name__ == "__main__":
    main()
