# collect_runs.py
# Usage:
#   python collect_runs.py --detect detect_critical.py --out summary.csv
#   python collect_runs.py --detect /content/detect_critical.py --out /content/summary.csv
#
# It searches:
#   - runs/*/alpha_log.csv
#   - /content/runs/*/alpha_log.csv
#
# For each run directory, it writes:
#   <run_dir>/critical.json
#
# Then it aggregates into summary.csv and summary.jsonl

import argparse
import glob
import json
import os
import subprocess
from typing import Dict, List, Optional


def find_alpha_logs() -> List[str]:
    patterns = [
        "runs/*/alpha_log.csv",
        "/content/runs/*/alpha_log.csv",
    ]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    # de-dup while keeping order
    seen = set()
    uniq = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def run_detect(detect_py: str, csv_path: str, json_out: str,
               extra_args: Optional[List[str]] = None) -> None:
    cmd = ["python", detect_py, csv_path, "--json_out", json_out]
    if extra_args:
        cmd.extend(extra_args)
    # Capture stdout/stderr for debugging; still show on error.
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


def flatten_record(payload: Dict) -> Dict:
    # Basic identifiers
    csv_path = payload.get("csv_path", "")
    run_dir = os.path.dirname(csv_path)
    run_name = os.path.basename(run_dir)

    first = payload.get("first") or {}
    thresholds = payload.get("thresholds") or {}
    baseline = payload.get("baseline") or {}
    self_target = payload.get("self_target") or {}
    diag_self = (payload.get("diag") or {}).get("self") or {}

    # Find specific events (if any)
    events = payload.get("events") or []
    by_kind = {e.get("kind"): e for e in events if isinstance(e, dict) and e.get("kind")}

    def ev_epoch(kind: str):
        e = by_kind.get(kind)
        return e.get("epoch") if e else None

    def ev_score(kind: str):
        e = by_kind.get(kind)
        return e.get("score") if e else None

    row = {
        "run_name": run_name,
        "run_dir": run_dir,
        "csv_path": csv_path,

        "baseline_desc": baseline.get("desc"),
        "baseline_n": baseline.get("n"),
        "baseline_args": str(baseline.get("baseline_args")),
        "baseline_rows": baseline.get("baseline_rows"),

        "z_hard": thresholds.get("z_hard"),
        "z_soft": thresholds.get("z_soft"),

        "self_target": self_target.get("value"),
        "self_target_inferred": self_target.get("inferred"),

        "first_kind": first.get("kind"),
        "first_epoch": first.get("epoch"),
        "first_score": first.get("score"),

        "env_break_epoch": ev_epoch("env_break"),
        "env_break_score": ev_score("env_break"),
        "self_break_epoch": ev_epoch("self_break"),
        "self_break_score": ev_score("self_break"),
        "self_shock_epoch": ev_epoch("self_shock"),
        "self_shock_score": ev_score("self_shock"),
        "rule_break_epoch": ev_epoch("rule_break"),
        "rule_break_score": ev_score("rule_break"),
    }

    # Add a handful of key self diagnostics (helpful for “why it didn’t fire”)
    key_diag_fields = [
        "baseline_after_drop_n",
        "self_err_med", "self_err_sigma",
        "d_self_med", "d_self_sigma",
        "d_selfm_med", "d_selfm_sigma",

        "self_err_exceed_soft_count",
        "self_err_exceed_hard_count",
        "self_err_persist_2of3_soft_count",
        "self_err_persist_2consec_hard_count",
        "self_err_first_exceed_soft_epoch",
        "self_err_first_exceed_hard_epoch",
        "self_err_first_persist_2of3_soft_epoch",
        "self_err_first_persist_2consec_hard_epoch",
        "self_err_longest_run_soft",
        "self_err_longest_run_hard",

        "self_err_max_z",
        "self_err_max_z_epoch",
        "d_selfm_max_z",
        "d_selfm_max_z_epoch",
        "d_self_max_z",
        "d_self_max_z_epoch",
    ]
    for k in key_diag_fields:
        if k in diag_self:
            row[f"diag_self_{k}"] = diag_self.get(k)
        else:
            row[f"diag_self_{k}"] = None

    return row


def write_csv(rows: List[Dict], out_csv: str) -> None:
    # minimal CSV writer without pandas dependency
    if not rows:
        raise ValueError("No rows to write.")
    # stable header: union of keys
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    s = ""
                else:
                    s = str(v)
                # naive CSV escaping
                if any(ch in s for ch in [",", "\n", "\""]):
                    s = "\"" + s.replace("\"", "\"\"") + "\""
                vals.append(s)
            f.write(",".join(vals) + "\n")


def write_jsonl(payloads: List[Dict], out_jsonl: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in payloads:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", type=str, default="detect_critical.py",
                    help="Path to detect_critical.py")
    ap.add_argument("--out", type=str, default="summary.csv",
                    help="Output CSV path")
    ap.add_argument("--jsonl_out", type=str, default="summary.jsonl",
                    help="Output JSONL path (raw payloads)")
    ap.add_argument("--extra", type=str, nargs="*", default=None,
                    help="Extra args passed to detect_critical.py (e.g., --z_hard 4 --z_soft 3)")
    ap.add_argument("--recompute", action="store_true",
                    help="Re-run detect even if critical.json already exists")
    args = ap.parse_args()

    detect_py = os.path.abspath(args.detect)

    logs = find_alpha_logs()
    if not logs:
        raise RuntimeError("No alpha_log.csv found under runs/* or /content/runs/*")

    payloads: List[Dict] = []
    rows: List[Dict] = []

    for csv_path in logs:
        run_dir = os.path.dirname(csv_path)
        json_out = os.path.join(run_dir, "critical.json")

        if args.recompute or (not os.path.exists(json_out)):
            print(f"[collect] running detect: {csv_path}")
            run_detect(detect_py, csv_path, json_out, extra_args=args.extra)
        else:
            print(f"[collect] using cached: {json_out}")

        payload = load_json(json_out)
        payloads.append(payload)
        rows.append(flatten_record(payload))

    write_jsonl(payloads, args.jsonl_out)
    write_csv(rows, args.out)

    print()
    print("==== collect_runs done ====")
    print(f"runs found: {len(logs)}")
    print(f"wrote: {os.path.abspath(args.out)}")
    print(f"wrote: {os.path.abspath(args.jsonl_out)}")


if __name__ == "__main__":
    main()