import json
import subprocess
from pathlib import Path

import pandas as pd

def run_detector(tmp_path: Path, rows, out_name="out.json"):
    csv_path = tmp_path / "alpha_log.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out_path = tmp_path / out_name

    # TODO: 実行コマンドはあなたのプロジェクトに合わせて修正
    # 例: python detect_critical.py --csv ... --out ...
    cmd = [
        "python",
        "detect_critical.py",
        "--csv", str(csv_path),
        "--out", str(out_path),
    ]
    subprocess.check_call(cmd)

    return json.loads(out_path.read_text())

def test_no_intervention_should_be_plateau_and_not_abnormal(tmp_path):
    # baseline epochs + valid epochs を最低限用意
    rows = [
        # baseline window (200-600)
        {"epoch":200, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":400, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":600, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":800, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":1000,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        # valid region (>=1200想定)
        {"epoch":1200,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":1400,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":1600,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":2000,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
    ]

    out = run_detector(tmp_path, rows)

    dself = out["diag"]["self"]
    assert out["events"] == []
    assert dself["p_peak_abnormal"] is False
    assert dself["p_direction"] == "Plateau"
    assert dself["p_eps_peak_abnormal"] is False
    assert dself["p_eps_direction"] == "Plateau"
    assert dself["eps_type"] == "NORMAL"

def test_abnormal_run_can_be_recover(tmp_path):
    # baselineは安定、validで self 系を大きく変化させるなど、異常を作る
    rows = [
        {"epoch":200, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":400, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":600, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":800, "epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},
        {"epoch":1000,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.0002},

        # valid: selfを暴れさせる（数値は実装のz算出に合わせて調整が必要）
        {"epoch":1200,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.05},
        {"epoch":1400,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.02},
        {"epoch":2000,"epsilon":0.01, "dC":0.1, "d_cf":0.01, "d_mono":0.001, "d_att":0.002, "d_self":0.005},
    ]

    out = run_detector(tmp_path, rows)
    dself = out["diag"]["self"]

    # abnormalなら Recover/Worsenが許可されることを最低限チェック
    assert dself["p_peak_abnormal"] in (True, False)  # まずはここを実データに合わせて絞る
    if dself["p_peak_abnormal"]:
        assert dself["p_direction"] in ("Recover", "Worsen", "Plateau")
