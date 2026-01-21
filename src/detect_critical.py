# detect_critical.py
# Usage:
#   python detect_critical.py runs/C4_noCF_s0/alpha_log.csv
# Options:
#   python detect_critical.py runs/.../alpha_log.csv --baseline 200 600 --z_hard 4 --z_soft 3

import argparse
import math
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


EPS = 1e-12
 
def safe_nanmax_stack(arrs: List[np.ndarray], axis: int = 0) -> np.ndarray:
    """
     np.nanmax on a stacked array, but WITHOUT RuntimeWarning when all-NaN slices exist.
    If a slice is all-NaN, result is NaN for that position.
    """
    X = np.stack(arrs, axis=axis)
    # mask: True where every value along 'axis' is NaN
    all_nan = np.all(np.isnan(X), axis=axis)
    # avoid RuntimeWarning: replace all-NaN slices with -inf BEFORE nanmax
    X2 = X.copy()
    # broadcast all_nan mask to X's shape
    if axis == 0:
        X2[:, all_nan] = -np.inf
    else:
        # generic broadcast (rarely needed here)
        idx = [slice(None)] * X2.ndim
        idx[axis] = all_nan
        X2[tuple(idx)] = -np.inf
    out = np.nanmax(X2, axis=axis)
    out[out == -np.inf] = np.nan
    return out

MIN_SIGMA = 1e-4  # ここは調整可（d_selfのスケール次第）

def robust_stats(x: np.ndarray) -> Tuple[float, float]:
    """Return (median, robust_sigma) where robust_sigma ~ std using MAD."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    robust_sigma = 1.4826 * (mad + EPS)
    scale = np.nanmedian(np.abs(x))  # baselineの典型スケール
    robust_sigma = max(robust_sigma, max(1e-4, 0.1 * scale))
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

def first_true_epoch(mask: np.ndarray, epochs: np.ndarray) -> Optional[int]:
    """Return epoch at first True in mask, else None."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return int(epochs[int(idx[0])])

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

def argmax_finite(x: np.ndarray) -> Optional[int]:
    """Return argmax of finite values, or None if no finite values exist."""
    m = np.isfinite(x)
    if not np.any(m):
        return None
    # use -inf for non-finite so argmax works
    y = np.where(m, x, -np.inf)
    i = int(np.argmax(y))
    if not np.isfinite(y[i]):
        return None
    return i

def fmt_stat(name: str, med: float, sig: float) -> str:
    return f"{name}: med={med:.6g}, sigma={sig:.6g}"

def _p_direction_from_series(epochs: np.ndarray, z_primary_valid: np.ndarray, valid: np.ndarray) -> Dict:
    """
    Compute direction of P after its valid-peak.
    Returns dict with:
      p_peak_epoch, p_peak, p_tail_epoch, p_tail, p_tail_minus_peak,
      p_slope_after_peak (optional), p_direction (Recover/Worsen/Plateau/Unknown)
    """
    out: Dict = {
        "p_peak_epoch": None,
        "p_peak": None,
        "p_tail_epoch": None,
        "p_tail": None,
        "p_tail_minus_peak": None,
        "p_slope_after_peak": None,
        "p_direction": "Unknown",
    }
    if z_primary_valid is None or epochs is None or valid is None:
        return out
    if len(z_primary_valid) == 0:
        return out
    zv = np.where(valid, z_primary_valid, np.nan)
    if not np.any(np.isfinite(zv)):
        return out
    ip = int(np.nanargmax(zv))
    out["p_peak_epoch"] = int(epochs[ip])
    out["p_peak"] = float(zv[ip])

    idx_valid = np.where(np.isfinite(zv))[0]
    it = int(idx_valid[-1])
    out["p_tail_epoch"] = int(epochs[it])
    out["p_tail"] = float(zv[it])
    out["p_tail_minus_peak"] = float(zv[it] - zv[ip])

    # slope after peak (only if we have >=2 finite points after peak)
    aft = idx_valid[idx_valid >= ip]
    if aft.size >= 2:
        xa = epochs[aft].astype(float)
        ya = zv[aft].astype(float)
        x0 = xa[0]
        denom = np.sum((xa - x0) ** 2)
        if denom > 0:
            slope = float(np.sum((xa - x0) * (ya - ya[0])) / denom)
            out["p_slope_after_peak"] = slope

    # direction (simple, robust): compare tail vs peak with small epsilon
    eps = 0.25  # 0.25σ相当の差分は「ほぼ同じ」とみなす（必要なら調整）
    d = out["p_tail_minus_peak"]
    if d is None or not np.isfinite(d):
        out["p_direction"] = "Unknown"
    elif d <= -eps:
        out["p_direction"] = "Recover"
    elif d >= eps:
        out["p_direction"] = "Worsen"
    else:
        out["p_direction"] = "Plateau"
    return out

def diagnose_self(df: pd.DataFrame, base_self: pd.DataFrame, self_target: float,
                  z_hard: float, z_soft: float) -> None:
    """
    Print why self_break/self_shock may be absent:
      - baseline med/sigma for self_err, d_self, |Δself_m|
      - max z and its epoch for each channel
      - whether key gates ever become true
    """
    if ("self_m_tr" not in df.columns) or ("d_self" not in df.columns):
        print("[DIAG][self] skipped: missing self_m_tr or d_self columns")
        return
    if len(base_self) < 3:
        print(f"[DIAG][self] skipped: base_self too small (n={len(base_self)})")
        return

    self_m = df["self_m_tr"].to_numpy(dtype=float)
    dself  = df["d_self"].to_numpy(dtype=float)
    epochs = df["epoch"].to_numpy(dtype=int)

    # self_err
    self_err = np.abs(self_m - float(self_target))
    self_m_b = base_self["self_m_tr"].to_numpy(dtype=float)
    self_err_b = np.abs(self_m_b - float(self_target))
    med_e, sig_e = robust_stats(self_err_b)
    z_e = robust_z(self_err, med_e, sig_e)

    # d_self
    dself_b = base_self["d_self"].to_numpy(dtype=float)
    med_ds, sig_ds = robust_stats(dself_b)
    z_ds = robust_z(dself, med_ds, sig_ds)

    # |Δself_m|
    d_selfm = np.abs(compute_diff_series(self_m))
    d_selfm_b = np.abs(compute_diff_series(self_m_b))
    med_dm, sig_dm = robust_stats(d_selfm_b)
    z_dm = robust_z(d_selfm, med_dm, sig_dm)

    # maxima
    ie = argmax_finite(z_e)
    ids = argmax_finite(z_ds)
    idm = argmax_finite(z_dm)

    print("[DIAG][self] baseline(after drop_first): n=", len(base_self))
    print("[DIAG][self] " + fmt_stat("self_err", med_e, sig_e))
    print("[DIAG][self] " + fmt_stat("d_self",   med_ds, sig_ds))
    print("[DIAG][self] " + fmt_stat("|Δself_m|", med_dm, sig_dm))

    if ie is not None:
        print(f"[DIAG][self] max z(self_err)={z_e[ie]:.2f} at epoch={epochs[ie]} (self_err={self_err[ie]:.6g})")
    else:
        print("[DIAG][self] max z(self_err)=NaN (no finite)")
    if ids is not None:
        print(f"[DIAG][self] max z(d_self)={z_ds[ids]:.2f} at epoch={epochs[ids]} (d_self={dself[ids]:.6g})")
    else:
        print("[DIAG][self] max z(d_self)=NaN (no finite)")
    if idm is not None:
        print(f"[DIAG][self] max z(|Δself_m|)={z_dm[idm]:.2f} at epoch={epochs[idm]} (|Δself_m|={d_selfm[idm]:.6g})")
    else:
        print("[DIAG][self] max z(|Δself_m|)=NaN (no finite)")

    # Gate coverage: did we ever exceed soft/hard?
    any_soft_err = bool(np.any(z_e > z_soft))
    any_hard_err = bool(np.any(z_e > z_hard))
    any_soft_dm  = bool(np.any(z_dm > z_soft))
    any_hard_dm  = bool(np.any(z_dm > z_hard))
    any_soft_ds  = bool(np.any(z_ds > z_soft))
    any_hard_ds  = bool(np.any(z_ds > z_hard))

    print(f"[DIAG][self] exceed soft: self_err={any_soft_err}, |Δself_m|={any_soft_dm}, d_self={any_soft_ds}")
    print(f"[DIAG][self] exceed hard: self_err={any_hard_err}, |Δself_m|={any_hard_dm}, d_self={any_hard_ds}")

def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    
def drop_first_baseline_row(base: pd.DataFrame, min_len: int = 4) -> pd.DataFrame:
    """
    For self-related signals, the very first logged row often contains warmup/init artifacts.
    If baseline is long enough, drop the first row to avoid poisoning robust stats.
    """
    if base is None:
        return base
    if len(base) >= min_len:
        return base.iloc[1:].copy()
    return base


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

    primary = (hard | soft)

    # confirm strength (1つのスコアに畳み込む)
    z_conf = safe_nanmax_stack([z_de, z_dI, 0.7 * z_dc], axis=0)
  
    # OR-mix:
    # 1) 従来: primary & confirm
    # 2) 強primaryなら confirmが弱くても拾う（ログ間隔が粗いとき用）
    # 3) 強confirmなら primaryが弱くても拾う（primaryが1点だけ外れるとき用）
    strong_primary = (z > (z_hard + 1.0))          # 例: 5σ相当
    strong_confirm = (z_conf > (z_hard + 1.0))     # 例: 5σ相当

    trigger = (primary & confirm) | strong_primary | (strong_confirm & primary)

    if not np.any(trigger):
        return None

    i = int(np.argmax(trigger))  # first True
    epoch = int(df.loc[i, "epoch"])
    reason = f"d_att z={z[i]:.2f}, confirm(z|Δenv|={z_de[i]:.2f}, z|ΔI|={z_dI[i]:.2f}, zΔcorr_drop={z_dc[i]:.2f})"
    return EventHit(epoch=epoch, kind="env_break", score=float(z[i]), reason=reason)


def _first_true_epoch(df: pd.DataFrame, mask: np.ndarray) -> Optional[int]:
    if mask is None or mask.size == 0 or not np.any(mask):
        return None
    i = int(np.argmax(mask))
    return int(df.loc[i, "epoch"])

def detect_self_break(df: pd.DataFrame, base: pd.DataFrame, z_hard: float, z_soft: float,
                        self_target: float,
                        debug_epoch: Optional[int] = None,
                        debug_rows: int = 2) -> Tuple[Optional[EventHit], Dict]:
    """
    Self-break:
      primary: self_err = |self_m_tr - self_target| has z > z_hard (2 consecutive) OR z > z_soft (2/3)
              OR absolute threshold self_err > max(self_target*0.8, med + 4σ)
      confirm: d_self z > 3 (nearby)
      confirm: d_self z > 3 (nearby)
    Patch(A):
      - Promote d_self anomaly to PRIMARY as well (previously only confirm).
      - Add |Δ self_m_tr| anomaly as PRIMARY (helps with sparse logging).
      - OR-mix: allow strong primary even if confirm is weak, and vice versa (nearby).
    baseline は “学習/初期化の揺れ” を含む可能性があるので、臨界検出には使わない
    self 系イベントは baseline 終端以降のみ評価する
    """
    print(f"[DIAG][self] base len(before drop)={len(base) if base is not None else None}")

    SINGLE_HARD_MARGIN = 1.5   # ←ここが今回の肝。まず 1.5 推奨

    require_cols(df, ["epoch", "self_m_tr", "d_self"])

    epochs = df["epoch"].to_numpy(dtype=int)

    diag: Dict = {}

    # IMPORTANT:
    # `base` passed into detect_self_break() is already "self baseline"
    # (i.e., main() has applied drop_first_baseline_row once).
    # Do NOT drop again here, otherwise baseline shrinks 5->4->3.
    base2 = base
    diag["baseline_after_drop_n"] = int(len(base2)) if base2 is not None else 0

    # --- valid mask: evaluate self-events only AFTER baseline region ---
    baseline_end_epoch = None
    if base2 is not None and ("epoch" in base2.columns) and len(base2) > 0:
        baseline_end_epoch = int(np.nanmax(base2["epoch"].to_numpy(dtype=float)))

    valid = np.ones(len(df), dtype=bool)
    valid[0] = False  # Always drop the very first row (warmup artifact-prone)

    if baseline_end_epoch is not None and np.isfinite(baseline_end_epoch):
        valid &= (epochs > baseline_end_epoch)


    self_m = df["self_m_tr"].to_numpy(dtype=float)
    self_err = np.abs(self_m - float(self_target))

    self_m_b = base2["self_m_tr"].to_numpy(dtype=float)
    self_err_b = np.abs(self_m_b - float(self_target))

    med, sig = robust_stats(self_err_b)
    z = robust_z(self_err, med, sig)

    diag["self_err_med"] = float(med) if np.isfinite(med) else None
    diag["self_err_sigma"] = float(sig) if np.isfinite(sig) else None

    # ---- baseline-only spike logging (NO effect on detection) ----
    # baseline region is where valid==False, but keep the same warmup exclusion (idx 0)
    idx = np.arange(len(df))
    baseline_mask = (~valid) & (idx > 0) & np.isfinite(z)
    diag["baseline_self_err_single_hard_count"] = int(np.nansum((z > z_hard) & baseline_mask)) if z.size else 0
    diag["baseline_self_err_single_strong_count"] = int(np.nansum((z > (z_hard + SINGLE_HARD_MARGIN)) & baseline_mask)) if z.size else 0
    if np.any(baseline_mask):
        zb = np.where(baseline_mask, z, np.nan)
        ib = int(np.nanargmax(zb)) if np.any(np.isfinite(zb)) else None
        diag["baseline_self_err_max_z"] = float(z[ib]) if ib is not None and np.isfinite(z[ib]) else None
        diag["baseline_self_err_max_z_epoch"] = int(df.loc[ib, "epoch"]) if ib is not None else None
        # strong-first epoch (baseline only)
        strong_b = (z > (z_hard + SINGLE_HARD_MARGIN)) & baseline_mask
        diag["baseline_self_err_first_strong_epoch"] = first_true_epoch(strong_b, epochs)
    else:
        diag["baseline_self_err_max_z"] = None
        diag["baseline_self_err_max_z_epoch"] = None
        diag["baseline_self_err_first_strong_epoch"] = None

    # ===== DIAG: persistence / duration of exceedance for self_err =====
    epochs = df["epoch"].to_numpy(dtype=int)

    # pointwise exceed
    # NOTE: diagnostics should reflect the same evaluation region as detection.
    # i.e., count exceedance only within `valid`.
    ex_soft = (z > z_soft) & valid
    ex_hard = (z > z_hard) & valid

    # persistence patterns used by the detector
    ex_soft_2of3 = two_of_three(ex_soft)                 # 2 of 3 window
    ex_hard_2consec = consecutive_true(ex_hard, k=2)     # 2 consecutive

    # simple counts
    n_soft = int(np.nansum(ex_soft))
    n_hard = int(np.nansum(ex_hard))
    n_soft_2of3 = int(np.nansum(ex_soft_2of3))
    n_hard_2consec = int(np.nansum(ex_hard_2consec))

    # first epochs where they become true
    first_soft = first_true_epoch(ex_soft, epochs)
    first_hard = first_true_epoch(ex_hard, epochs)
    first_soft_2of3 = first_true_epoch(ex_soft_2of3, epochs)
    first_hard_2consec = first_true_epoch(ex_hard_2consec, epochs)

    # longest consecutive run length (optional but useful)
    def longest_run(mask: np.ndarray) -> int:
        best = 0
        run = 0
        for v in mask:
            run = run + 1 if bool(v) else 0
            if run > best:
                best = run
        return int(best)

    longest_soft_run = longest_run(ex_soft)
    longest_hard_run = longest_run(ex_hard)

    print(f"[DIAG][self] self_err exceed counts: soft(z>{z_soft})={n_soft}, hard(z>{z_hard})={n_hard}")
    print(f"[DIAG][self] self_err persistence: 2of3_soft={n_soft_2of3}, 2consec_hard={n_hard_2consec}")
    print(f"[DIAG][self] self_err first exceed epoch: soft={first_soft}, hard={first_hard}")
    print(f"[DIAG][self] self_err first persistence epoch: 2of3_soft={first_soft_2of3}, 2consec_hard={first_hard_2consec}")
    print(f"[DIAG][self] self_err longest consecutive run: soft={longest_soft_run}, hard={longest_hard_run}")


    hard = consecutive_true(z > z_hard, k=2)
    soft = two_of_three(z > z_soft)

    # --- diagnostics for persistence (self_err) ---
    z_soft_mask = (z > z_soft)
    z_hard_mask = (z > z_hard)
    z_soft_mask = (z > z_soft) & valid
    z_hard_mask = (z > z_hard) & valid
    soft_2of3 = two_of_three(z_soft_mask)
    hard_2con = consecutive_true(z_hard_mask, k=2)

    def _longest_run(mask: np.ndarray) -> int:
        if mask.size == 0:
            return 0
        best = 0
        cur = 0
        for v in mask:
            cur = (cur + 1) if bool(v) else 0
            if cur > best:
                best = cur
        return int(best)

    diag["self_err_exceed_soft_count"] = int(np.nansum(z_soft_mask)) if z_soft_mask.size else 0
    diag["self_err_exceed_hard_count"] = int(np.nansum(z_hard_mask)) if z_hard_mask.size else 0
    diag["self_err_persist_2of3_soft_count"] = int(np.nansum(soft_2of3)) if soft_2of3.size else 0
    diag["self_err_persist_2consec_hard_count"] = int(np.nansum(hard_2con)) if hard_2con.size else 0
    diag["self_err_first_exceed_soft_epoch"] = _first_true_epoch(df, z_soft_mask)
    diag["self_err_first_exceed_hard_epoch"] = _first_true_epoch(df, z_hard_mask)
    diag["self_err_first_persist_2of3_soft_epoch"] = _first_true_epoch(df, soft_2of3)
    diag["self_err_first_persist_2consec_hard_epoch"] = _first_true_epoch(df, hard_2con)
    diag["self_err_longest_run_soft"] = _longest_run(z_soft_mask)
    diag["self_err_longest_run_hard"] = _longest_run(z_hard_mask)

    # absolute gate
    abs_thr = max(float(self_target) * 0.8, med + 4.0 * sig)
    abs_gate_raw = (self_err > abs_thr)
    # 単発ではなく「2連続」か「2/3」だけ許可
    abs_gate = consecutive_true(abs_gate_raw, k=2) | two_of_three(abs_gate_raw)

    # --- d_self anomaly (was confirm only; now also contributes to primary) ---
    dself = df["d_self"].to_numpy(dtype=float)
    dself_b = base2["d_self"].to_numpy(dtype=float)
    med_ds, sig_ds = robust_stats(dself_b)
    z_ds = robust_z(dself, med_ds, sig_ds)

    diag["d_self_med"] = float(med_ds) if np.isfinite(med_ds) else None
    diag["d_self_sigma"] = float(sig_ds) if np.isfinite(sig_ds) else None

    # --- |Δ self_m_tr| anomaly (sparse logs friendly) ---
    d_selfm = np.abs(compute_diff_series(self_m))
    d_selfm_b = np.abs(compute_diff_series(base2["self_m_tr"].to_numpy(dtype=float)))
    med_dm, sig_dm = robust_stats(d_selfm_b)
    z_dm = robust_z(d_selfm, med_dm, sig_dm)

    diag["d_selfm_med"] = float(med_dm) if np.isfinite(med_dm) else None
    diag["d_selfm_sigma"] = float(sig_dm) if np.isfinite(sig_dm) else None

    single_hard = (z > z_hard) & valid
    # Diagnostics: single hard exceed should also respect `valid`
    single_hard = (z > z_hard) & valid
    single_hard_strong = (z > (z_hard + SINGLE_HARD_MARGIN)) & valid
 
    # ---- small focused debug print around a specific epoch ----
    if debug_epoch is not None:
        # choose the closest row to debug_epoch (works even if epoch not logged exactly)
        i0 = int(np.argmin(np.abs(epochs - int(debug_epoch))))
        lo = max(0, i0 - int(debug_rows))
        hi = min(len(df) - 1, i0 + int(debug_rows))
        print(f"[DEBUG][self] focus_epoch={debug_epoch} closest_epoch={int(epochs[i0])} idx={i0} "
              f"rows=[{lo}..{hi}] baseline_end_epoch={baseline_end_epoch}")
        # header
        print("[DEBUG][self] idx epoch valid  self_m_tr    self_err      z(self_err)   "
              "d_self       z(d_self)    |Δself_m|    z(|Δself_m|)  "
              "primary_err primary_dself primary_dselfm primary confirm_near strongP strongC trigger")
        # placeholders for masks that are computed later in the function:
        # we print what is available now, and later we will print a second block after trigger is built.
        for i in range(lo, hi + 1):
            dm_i = float(d_selfm[i]) if np.isfinite(d_selfm[i]) else float("nan")
            print(f"[DEBUG][self] {i:3d} {int(epochs[i]):4d} {int(bool(valid[i])):5d} "
                  f"{self_m[i]:10.6g} {self_err[i]:12.6g} {z[i]:12.4f} "
                  f"{dself[i]:10.6g} {z_ds[i]:12.4f} {dm_i:10.6g} {z_dm[i]:12.4f} "
                  f"{'-':>10} {'-':>12} {'-':>13} {'-':>7} {'-':>12} {'-':>7} {'-':>7} {'-':>7}")

    # primary components
    primary_err = hard | soft | abs_gate | single_hard_strong

    # d_self primary gate: allow either z_hard-ish OR z_soft-ish (2/3) for robustness
    primary_dself_raw = consecutive_true(z_ds > z_hard, k=2) | two_of_three(z_ds > z_soft)

    # d_self を primary にするには「実体の変化」も要求する
    has_real_self_change = (z > z_soft) | (z_dm > z_soft) | abs_gate
    primary_dself = primary_dself_raw & has_real_self_change

    # Δself_m_tr primary gate: usually more noisy, so keep a slightly stricter default
    primary_dselfm = consecutive_true(z_dm > (z_hard - 0.5), k=2) | two_of_three(z_dm > (z_soft + 0.5))

    # merge to primary
    primary = primary_err | primary_dself | primary_dselfm

    # confirm (keep d_self as confirm signal, but make it "nearby-friendly")
    confirm = (z_ds > 3.0)

    # allow confirm within +/-1 log step
    confirm_near = confirm.copy()
    confirm_near[:-1] |= confirm[1:]
    confirm_near[1:]  |= confirm[:-1]

    # OR-mix (like env_break):
    # - classic: primary & confirm_near
    # - strong primary: very large z on any primary channel
    # - strong confirm: very large z_ds, but primary may be a 1-point miss
    # NOTE:
    # z_ds(d_self) は baseline が小さいと爆発しやすいので、
    # strong_primary/score の主導には使わず「補助トリガ」に回す。
    # IMPORTANT:
    #   Do NOT let d_self alone create strong_primary.
    #   Only allow d_self to contribute when "real self change" gate holds.
    z_primary_core = safe_nanmax_stack([z, z_dm], axis=0)  # self_err / Δself_m only
    z_primary_dself = np.where(primary_dself, z_ds, np.nan)  # d_self only when gated
    z_primary = safe_nanmax_stack([z_primary_core, z_primary_dself], axis=0)
    strong_primary_raw = (z_primary > max(5.0, (z_hard + 1.0)))
    strong_primary = strong_primary_raw & valid

    # d_self が強烈でも「実体変化があるときだけ」強confirm扱い
    strong_confirm_raw = (z_ds > (z_hard + 1.0)) & has_real_self_change
    strong_confirm = strong_confirm_raw & valid

    # NOTE: trigger itself remains valid-gated, but we also keep raw masks for diagnostics.
    trigger = ((primary & confirm_near) | strong_primary_raw | (strong_confirm_raw & primary)) & valid

    # ---- second focused debug block (now that primary/confirm/trigger exist) ----
    if debug_epoch is not None:
        i0 = int(np.argmin(np.abs(epochs - int(debug_epoch))))
        lo = max(0, i0 - int(debug_rows))
        hi = min(len(df) - 1, i0 + int(debug_rows))
        # recompute the three "primary parts" flags for printing (they already exist as arrays)
        # primary_err / primary_dself / primary_dselfm exist above
        for i in range(lo, hi + 1):
            print(f"[DEBUG][self] {i:3d} {int(epochs[i]):4d} {int(bool(valid[i])):5d} "
                  f"{self_m[i]:10.6g} {self_err[i]:12.6g} {z[i]:12.4f} "
                  f"{dself[i]:10.6g} {z_ds[i]:12.4f} {float(d_selfm[i]):10.6g} {z_dm[i]:12.4f} "
                  f"{int(bool(primary_err[i])):10d} {int(bool(primary_dself[i])):12d} {int(bool(primary_dselfm[i])):13d} "
                  f"{int(bool(primary[i])):7d} {int(bool(confirm_near[i])):12d} "
                  f"{int(bool(strong_primary_raw[i])):7d} {int(bool(strong_confirm_raw[i])):7d} {int(bool(trigger[i])):7d}")

    if not np.any(trigger):
        # add a few max stats to diag even when no hit
        if z.size:
            im = int(np.nanargmax(z)) if np.any(np.isfinite(z)) else 0
            diag["self_err_max_z"] = float(z[im]) if np.isfinite(z[im]) else None
            diag["self_err_max_z_epoch"] = int(df.loc[im, "epoch"])
        if z_dm.size:
            im = int(np.nanargmax(z_dm)) if np.any(np.isfinite(z_dm)) else 0
            diag["d_selfm_max_z"] = float(z_dm[im]) if np.isfinite(z_dm[im]) else None
            diag["d_selfm_max_z_epoch"] = int(df.loc[im, "epoch"])
        if z_ds.size:
            im = int(np.nanargmax(z_ds)) if np.any(np.isfinite(z_ds)) else 0
            diag["d_self_max_z"] = float(z_ds[im]) if np.isfinite(z_ds[im]) else None
            diag["d_self_max_z_epoch"] = int(df.loc[im, "epoch"])
        # ---- extra diag: strong_primary/confirm (valid-aware) ----
        diag["strong_primary_raw_count"] = int(np.nansum(strong_primary_raw)) if strong_primary_raw.size else 0
        diag["strong_primary_valid_count"] = int(np.nansum(strong_primary)) if strong_primary.size else 0
        diag["strong_confirm_raw_count"] = int(np.nansum(strong_confirm_raw)) if strong_confirm_raw.size else 0
        diag["strong_confirm_valid_count"] = int(np.nansum(strong_confirm)) if strong_confirm.size else 0

        diag["first_strong_primary_raw_epoch"] = first_true_epoch(strong_primary_raw, epochs)
        diag["first_strong_primary_valid_epoch"] = first_true_epoch(strong_primary, epochs)
        diag["first_strong_confirm_raw_epoch"] = first_true_epoch(strong_confirm_raw, epochs)
        diag["first_strong_confirm_valid_epoch"] = first_true_epoch(strong_confirm, epochs)

        # z_primary maxima (raw / valid)
        if z_primary.size and np.any(np.isfinite(z_primary)):
            ip = int(np.nanargmax(z_primary))
            diag["z_primary_max"] = float(z_primary[ip]) if np.isfinite(z_primary[ip]) else None
            diag["z_primary_max_epoch"] = int(df.loc[ip, "epoch"])
        z_primary_valid = np.where(valid, z_primary, np.nan)
        if z_primary_valid.size and np.any(np.isfinite(z_primary_valid)):
            ipv = int(np.nanargmax(z_primary_valid))
            diag["z_primary_valid_max"] = float(z_primary_valid[ipv]) if np.isfinite(z_primary_valid[ipv]) else None
            diag["z_primary_valid_max_epoch"] = int(df.loc[ipv, "epoch"])

        # ---- P-direction summary (peak -> tail) ----
        pdict = _p_direction_from_series(epochs=epochs, z_primary_valid=z_primary, valid=valid)
        # NOTE: function expects "z_primary_valid-like" array + valid; we pass raw z_primary and gate by valid inside.
        # (ここでは valid で NaN 化するのでOK)
        diag.update(pdict)
        
        return None, diag

    i = int(np.argmax(trigger))
    epoch = int(df.loc[i, "epoch"])
    reason = (
        f"self_err={self_err[i]:.4f} (target={self_target}), "
        f"z(self_err)={z[i]:.2f}, z(d_self)={z_ds[i]:.2f}, z(|Δself_m|)={z_dm[i]:.2f}"
    )
    # score should reflect self target violation / state change.
    # include z_ds only when primary_dself gate is active at i.
    cand = [z[i], z_dm[i]]
    if bool(primary_dself[i]):
        cand.append(z_ds[i])
    score = float(np.nanmax(cand))

    # also store hit-related diag
    diag["self_break_epoch"] = int(epoch)
    diag["self_break_score"] = float(score)
    diag["self_err_single_hard_count"] = int(np.nansum(single_hard))
    diag["self_err_single_hard_strong_count"] = int(np.nansum(single_hard_strong))
    diag["self_err_first_single_hard_epoch"] = first_true_epoch(single_hard, epochs)
    diag["self_err_first_single_hard_strong_epoch"] = first_true_epoch(single_hard_strong, epochs)

    # ---- extra diag: strong_primary/confirm (valid-aware) ----
    diag["strong_primary_raw_count"] = int(np.nansum(strong_primary_raw)) if strong_primary_raw.size else 0
    diag["strong_primary_valid_count"] = int(np.nansum(strong_primary)) if strong_primary.size else 0
    diag["strong_confirm_raw_count"] = int(np.nansum(strong_confirm_raw)) if strong_confirm_raw.size else 0
    diag["strong_confirm_valid_count"] = int(np.nansum(strong_confirm)) if strong_confirm.size else 0

    diag["first_strong_primary_raw_epoch"] = first_true_epoch(strong_primary_raw, epochs)
    diag["first_strong_primary_valid_epoch"] = first_true_epoch(strong_primary, epochs)
    diag["first_strong_confirm_raw_epoch"] = first_true_epoch(strong_confirm_raw, epochs)
    diag["first_strong_confirm_valid_epoch"] = first_true_epoch(strong_confirm, epochs)

    if z_primary.size and np.any(np.isfinite(z_primary)):
        ip = int(np.nanargmax(z_primary))
        diag["z_primary_max"] = float(z_primary[ip]) if np.isfinite(z_primary[ip]) else None
        diag["z_primary_max_epoch"] = int(df.loc[ip, "epoch"])
    z_primary_valid = np.where(valid, z_primary, np.nan)
 
    # ---- store P series into diag (so json_out can carry it without recompute) ----
    # Keep it lightweight: lists of numbers (NaN -> None) + valid mask
    diag["p_series_epoch"] = [int(e) for e in epochs.tolist()]
    diag["p_series_valid"] = [bool(v) for v in valid.tolist()]
    diag["p_series_z_primary_valid"] = [
        (None if not np.isfinite(v) else float(v)) for v in z_primary_valid.tolist()
    ]

    if z_primary_valid.size and np.any(np.isfinite(z_primary_valid)):
        ipv = int(np.nanargmax(z_primary_valid))
        diag["z_primary_valid_max"] = float(z_primary_valid[ipv]) if np.isfinite(z_primary_valid[ipv]) else None
        diag["z_primary_valid_max_epoch"] = int(df.loc[ipv, "epoch"])

    # ---- P-direction summary (peak -> tail) ----
    pdict = _p_direction_from_series(epochs=epochs, z_primary_valid=z_primary, valid=valid)
    diag.update(pdict)

    return EventHit(epoch=epoch, kind="self_break", score=score, reason=reason), diag

def detect_self_shock(df: pd.DataFrame, base: pd.DataFrame, z_hard: float, z_soft: float) -> Optional[EventHit]:
    """
    Self-shock (NOT a self_target violation):
      Detect an abrupt spike in d_self itself (e.g., teacher/student mismatch or stability issue),
      but do NOT let it become FIRST BREAK. This is a diagnostic event.

    Trigger:
      - z(d_self) > z_hard (2 consecutive) OR z(d_self) > z_soft (2/3)
      - and NOT accompanied by strong self_err drift (otherwise self_break covers it)
    """
    require_cols(df, ["epoch", "self_m_tr", "d_self"])
    # `base` is already self-baseline (first row removed in main()).
    # Do NOT drop again here.
    base2 = base

    dself = df["d_self"].to_numpy(dtype=float)
    dself_b = base2["d_self"].to_numpy(dtype=float)
    med_ds, sig_ds = robust_stats(dself_b)
    z_ds = robust_z(dself, med_ds, sig_ds)

    # main shock condition
    hard = consecutive_true(z_ds > z_hard, k=2)
    soft = two_of_three(z_ds > z_soft)
    primary = hard | soft

    # exclude if there's clear drift in self_m_tr away from baseline target-ish (i.e., let self_break handle)
    self_m = df["self_m_tr"].to_numpy(dtype=float)
    targetish = float(np.median(base2["self_m_tr"].to_numpy(dtype=float)))
    self_err = np.abs(self_m - targetish)
    self_err_b = np.abs(base2["self_m_tr"].to_numpy(dtype=float) - targetish)
    med_e, sig_e = robust_stats(self_err_b)
    z_e = robust_z(self_err, med_e, sig_e)

    not_drift = (z_e < (z_soft + 0.5)) | ~np.isfinite(z_e)
    trigger = primary & not_drift
    if not np.any(trigger):
        return None

    i = int(np.argmax(trigger))
    epoch = int(df.loc[i, "epoch"])
    reason = f"z(d_self)={z_ds[i]:.2f} (shock), z(self_err_vs_targetish)={z_e[i]:.2f}"
    return EventHit(epoch=epoch, kind="self_shock", score=float(z_ds[i]), reason=reason)


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
    # NOTE: self_shock is diagnostic and should NOT win FIRST BREAK unless nothing else exists.
    prio = {"self_break": 0, "env_break": 1, "rule_break": 2, "self_shock": 99}
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
    ap.add_argument("--json_out", type=str, default=None,
                    help="If set, write result JSON to this path.")
    ap.add_argument("--include_p_series", action="store_true",
                    help="If set, include z_primary_valid time series in JSON (diag.self.p_series).")
    ap.add_argument("--debug_epoch", type=int, default=None,
                    help="If set, print detailed self-break diagnostics around this epoch (closest logged row).")
    ap.add_argument("--debug_rows", type=int, default=2,
                    help="Number of rows before/after the closest debug_epoch row to print (default=2).")
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
    
    # Self-related baseline: drop the first row (warmup/init artifact) if possible
    base_self = drop_first_baseline_row(base, min_len=4)


    # infer self_target if not provided:
    # - since alpha_log.csv doesn't contain 'self_target' column, we infer it:
    #   use median of self_m_tr in baseline as "target-ish" (works if self target was enforced strongly).
    if args.self_target is None:
        if "self_m_tr" in df.columns:
            # use baseline, but if baseline is first-K rows, consider using a slightly later slice
            inferred = float(np.median(base_self["self_m_tr"].to_numpy(dtype=float)))
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

    diag_self: Dict = {}
    self_hit, diag_self = detect_self_break(
        df, base_self, z_hard=z_hard, z_soft=z_soft, self_target=self_target,
        debug_epoch=args.debug_epoch, debug_rows=args.debug_rows
    )
    if self_hit is not None:
        events.append(self_hit)

    # diagnostic only (should not become FIRST BREAK)
    shock_hit = detect_self_shock(df, base_self, z_hard=z_hard, z_soft=z_soft)
    if shock_hit:
        events.append(shock_hit)

    rule_hit = detect_rule_break(df, base, z_hard=z_hard, z_soft=z_soft)
    if rule_hit:
        events.append(rule_hit)

    print("==== Criticality Detector ====")
    print(f"CSV: {args.csv_path}")
    print(f"Baseline: {baseline_desc} (n={len(base)})")
    print(f"z_hard={z_hard} (2 consecutive), z_soft={z_soft} (2 of 3)")
    print(f"self_target used: {self_target:.6f} (inferred={args.self_target is None})")
    print()

    # Diagnostic: if neither self_break nor self_shock is detected, print why.
    has_self_event = any(e.kind in ("self_break", "self_shock") for e in events)
    if not has_self_event and ("self_m_tr" in df.columns) and ("d_self" in df.columns):
        diagnose_self(df, base_self, self_target=self_target, z_hard=z_hard, z_soft=z_soft)
        print()

    if not events:
        print("No break detected under current thresholds.")
        first = None
    else:
        for e in sorted(events, key=lambda x: x.epoch):
            print(f"- {e.kind}: epoch={e.epoch}, score={e.score:.2f}")
            print(f"  reason: {e.reason}")

        # FIRST BREAK: ignore self_shock unless it is the only event kind present
        events_for_first = [e for e in events if e.kind != "self_shock"]
        if not events_for_first:
            events_for_first = events
        first = pick_first_event(events_for_first)
        print()
        print("---- FIRST BREAK (critical epoch) ----")
        print(f"kind={first.kind}  epoch={first.epoch}  score={first.score:.2f}")
        print(f"reason: {first.reason}")

    # (events が無い場合も first=None のまま続行し、json_out があれば書く)

    # ---- JSON output (optional) ----
    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "csv_path": os.path.abspath(args.csv_path),
           "baseline": {
                "desc": baseline_desc,
                "n": int(len(base)),
                "baseline_args": [int(b0), int(b1)],
                "baseline_rows": int(K),
            },
            "thresholds": {"z_hard": float(z_hard), "z_soft": float(z_soft)},
            "self_target": {"value": float(self_target), "inferred": bool(args.self_target is None)},
            "events": [
                {"kind": e.kind, "epoch": int(e.epoch), "score": float(e.score), "reason": str(e.reason)}
                for e in sorted(events, key=lambda x: x.epoch)
            ],
            "first": {
                "kind": first.kind,
                "epoch": int(first.epoch),
                "score": float(first.score),
                "reason": str(first.reason),
            } if first else None,
             "diag": {"self": diag_self},
        }
 
        # If not requested, remove the series to keep JSON small
        if (not args.include_p_series) and ("diag" in payload) and ("self" in payload["diag"]):
            for k in ["p_series_epoch", "p_series_valid", "p_series_z_primary_valid"]:
                payload["diag"]["self"].pop(k, None)

        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
