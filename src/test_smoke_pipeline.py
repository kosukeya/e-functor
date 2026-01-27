from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train
import detect_critical


def test_smoke_train_detect(tmp_path):
    run_dir = tmp_path / "run_smoke"
    train.main([
        "--run-dir", str(run_dir),
        "--run-id", "smoke",
        "--epochs", "4",
        "--n", "32",
        "--log-every", "1",
        "--island-every", "2",
        "--device", "cpu",
        "--seed", "0",
    ])

    csv_path = run_dir / "alpha_log.csv"
    assert csv_path.exists()

    out_json = run_dir / "collapse" / "critical.json"
    args = detect_critical.parse_args([str(csv_path), "--json_out", str(out_json)])
    detect_critical.run_detect_critical(args)

    assert out_json.exists()
