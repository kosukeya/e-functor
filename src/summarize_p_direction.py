# summarize_p_direction.py
# Usage:
#   python summarize_p_direction.py \
#     --detect detect_critical.py \
#     --runs runs/C1_targetUp_s0/alpha_log.csv runs/C2_noEnv_s0/alpha_log.csv runs/C4_noCF_s0/alpha_log.csv \
#     --out p_direction_summary.csv \
#     --include_p_series
#
# If --runs is omitted, it will auto-glob runs/*/alpha_log.csv and /content/runs/*/alpha_log.csv

import argparse
import csv
import glob
import json
import os
import subprocess
from typing import List, Dict, Optional


def run_detect(detect_py: str, csv_path: str, json_out: str, include_p_series: bool) -> None:
    cmd = ["python", detect_py, csv_path, "--json_out", json_out]
    if include_p_series:
        cmd.append("--include_p_series")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("==== detect_critical.py FAILED ====")
        print("CMD:", " ".join(cmd))
        print("STDOUT:\n", p.stdout)
        print("STDERR:\n", p.stderr)
        raise RuntimeError(f"detect_critical.py failed for {csv_path}")


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_default_runs() -> List[str]:
    pats = ["runs/*/alpha_log.csv", "/content/runs/*/alpha_log.csv"]
    out: List[str] = []
    for pat in pats:
        out.extend(glob.glob(pat))
    # de-dup
    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", type=str, default="detect_critical.py")
    ap.add_argument("--runs", type=str, nargs="*", default=None,
                    help="alpha_log.csv paths. If omitted, auto-glob runs/*/alpha_log.csv")
    ap.add_argument("--out", type=str, default="p_direction_summary.csv")
    ap.add_argument("--include_p_series", action="store_true")
    ap.add_argument("--recompute", action="store_true",
                    help="Re-run detect even if <run_dir>/critical.json exists")
    # --- Recover minimum conditions (new) ---
    ap.add_argument("--drop_min", type=float, default=0.30,
                    help="Minimum required drop to call Recover: (Ppeak - Ptail_mean) >= drop_min")
    ap.add_argument("--tail_n", type=int, default=2,
                    help="How many tail points to average for Ptail_mean (use >=2 for sparse logs).")
    ap.add_argument("--min_tail_points", type=int, default=2,
                    help="Require at least this many finite tail points to allow Recover.")
    args = ap.parse_args()

    detect_py = os.path.abspath(args.detect)
    runs = args.runs if args.runs else find_default_runs()
    if not runs:
        raise RuntimeError("No alpha_log.csv found. Provide --runs ...")

    rows: List[Dict] = []
    for csv_path in runs:
        csv_path = os.path.abspath(csv_path)
        run_dir = os.path.dirname(csv_path)
        run_name = os.path.basename(run_dir)
        json_out = os.path.join(run_dir, "critical.json")

        if args.recompute or (not os.path.exists(json_out)):
            print(f"[sum] running detect: {csv_path}")
            run_detect(detect_py, csv_path, json_out, include_p_series=args.include_p_series)
        else:
            print(f"[sum] using cached: {json_out}")

        payload = load_json(json_out)
        diag_self = (payload.get("diag") or {}).get("self") or {}
        first = payload.get("first") or {}
        thresholds = payload.get("thresholds") or {}
        baseline = payload.get("baseline") or {}

        # -----------------------------
        # Recompute p_direction with minimum Recover conditions (new)
        # -----------------------------
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        p_peak = _to_float(diag_self.get("p_peak"))
        p_tail = _to_float(diag_self.get("p_tail"))
        slope = _to_float(diag_self.get("p_slope_after_peak"))

        # Try to read p-series from JSON (when --include_p_series is used).
        # We accept several possible key names to be robust.
        p_series = diag_self.get("p_series_valid")
        if p_series is None:
            p_series = diag_self.get("p_series")
        if p_series is None:
            p_series = diag_self.get("z_primary_valid_series")  # if you used another name

        # Compute tail mean from last N finite points if series exists; otherwise fallback to last point.
        tail_n = max(1, int(args.tail_n))
        tail_points = 0
        p_tail_mean = None

        if isinstance(p_series, list) and len(p_series) > 0:
            tail_slice = p_series[-tail_n:]
            finite = []
            for v in tail_slice:
                fv = _to_float(v)
                if fv is not None:
                    # allow NaN check
                    if fv == fv:  # not NaN
                        finite.append(fv)
            tail_points = len(finite)
            if tail_points > 0:
                p_tail_mean = sum(finite) / float(tail_points)
        else:
            # fallback
            if p_tail is not None and (p_tail == p_tail):  # not NaN
                p_tail_mean = p_tail
                tail_points = 1
            else:
                p_tail_mean = None
                tail_points = 0

        # drop and minimum condition
        drop = None
        if (p_peak is not None) and (p_tail_mean is not None):
            drop = p_peak - p_tail_mean

        min_ok = (
            (drop is not None) and (drop >= float(args.drop_min)) and
            (tail_points >= int(args.min_tail_points))
        )

        # Re-judge direction with the new minimum condition.
        # (Keep it conservative: Recover only when slope is negative AND min_ok.)
        p_direction_min = None
        if (slope is not None) and (p_peak is not None) and (p_tail_mean is not None):
            if (slope < 0.0) and min_ok:
                p_direction_min = "Recover"
            elif slope > 0.0 and (p_tail_mean - p_peak) > 0.0:
                p_direction_min = "Worsen"
            else:
                p_direction_min = "Plateau"

        row = {
            "run_name": run_name,
            "csv_path": payload.get("csv_path"),
            "baseline_desc": (baseline.get("desc") if baseline else None),
            "z_hard": (thresholds.get("z_hard") if thresholds else None),
            "z_soft": (thresholds.get("z_soft") if thresholds else None),

            "first_kind": first.get("kind"),
            "first_epoch": first.get("epoch"),
            "first_score": first.get("score"),

            # baseline-internal spike indicators (self_err)
            "baseline_self_err_single_hard_count": diag_self.get("baseline_self_err_single_hard_count"),
            "baseline_self_err_single_strong_count": diag_self.get("baseline_self_err_single_strong_count"),
            "baseline_self_err_max_z": diag_self.get("baseline_self_err_max_z"),
            "baseline_self_err_max_z_epoch": diag_self.get("baseline_self_err_max_z_epoch"),

            # P-direction
            "p_direction": diag_self.get("p_direction"),
            "p_peak_epoch": diag_self.get("p_peak_epoch"),
            "p_peak": diag_self.get("p_peak"),
            "p_tail_epoch": diag_self.get("p_tail_epoch"),
            "p_tail": diag_self.get("p_tail"),
            "p_tail_minus_peak": diag_self.get("p_tail_minus_peak"),
            "p_slope_after_peak": diag_self.get("p_slope_after_peak"),
            # P-direction (re-judged with Recover minimum conditions)
            "p_tail_mean": p_tail_mean,
            "p_tail_points": tail_points,
            "p_drop_peak_to_tail_mean": drop,
            "p_recover_min_ok": min_ok,
            "p_direction_min": p_direction_min,
        }
        rows.append(row)

    # write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print("==== summarize_p_direction done ====")
    print("wrote:", os.path.abspath(args.out))

    # print a compact console summary
    print()
    for r in rows:
        print(f"- {r['run_name']}: {r.get('p_direction_min')} (raw={r.get('p_direction')})  "
              f"Ppeak={r.get('p_peak')}@{r.get('p_peak_epoch')}  "
              f"PtailMean={r.get('p_tail_mean')} (pts={r.get('p_tail_points')})  "
              f"drop={r.get('p_drop_peak_to_tail_mean')}  "
              f"slope={r.get('p_slope_after_peak')}")

if __name__ == "__main__":
    main()
