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
        print(f"- {r['run_name']}: {r.get('p_direction')}  "
              f"Ppeak={r.get('p_peak')}@{r.get('p_peak_epoch')}  "
              f"Ptail={r.get('p_tail')}@{r.get('p_tail_epoch')}  "
              f"Δ={r.get('p_tail_minus_peak')}  "
              f"slope={r.get('p_slope_after_peak')}")


if __name__ == "__main__":
    main()