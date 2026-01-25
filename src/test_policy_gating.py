import numpy as np
import pytest

from detect_critical import _p_direction_from_series

def test_recover_forced_plateau_when_not_abnormal():
    epochs = np.array([1200, 1400, 2000], dtype=int)
    # peak 5 -> tail 2 なので Recover っぽい
    z_primary = np.array([5.0, 3.0, 2.0], dtype=float)
    valid = np.array([True, True, True], dtype=bool)

    out = _p_direction_from_series(
        epochs=epochs,
        z_primary_valid=z_primary,
        valid=valid,
        recover_requires_abnormal=True,
        abnormal_threshold=3.0,   # z_soft 相当
    )
    # peak=5>=3 なので abnormal=True → Recover 許可
    assert out["p_peak_abnormal"] is True
    assert out["p_direction"] == "Recover"

def test_recover_is_downgraded_when_peak_not_abnormal():
    epochs = np.array([1200, 1400, 2000], dtype=int)
    # peak 2.9 -> tail 2.0 なので Recover っぽいが、peak<3.0なので abnormal=False
    z_primary = np.array([2.9, 2.5, 2.0], dtype=float)
    valid = np.array([True, True, True], dtype=bool)

    out = _p_direction_from_series(
        epochs=epochs,
        z_primary_valid=z_primary,
        valid=valid,
        recover_requires_abnormal=True,
        abnormal_threshold=3.0,
    )
    assert out["p_peak_abnormal"] is False
    assert out["p_direction"] == "Plateau"   # Recover 禁止

def test_recover_not_allowed_if_peak_only_in_invalid_region():
    epochs = np.array([600, 1200, 1400, 2000], dtype=int)
    z_primary = np.array([10.0, 2.0, 1.5, 1.0], dtype=float)  # 最大は 600 の 10.0
    valid = np.array([False, True, True, True], dtype=bool)   # 600 は baseline で無効

    out = _p_direction_from_series(
        epochs=epochs,
        z_primary_valid=z_primary,
        valid=valid,
        recover_requires_abnormal=True,
        abnormal_threshold=3.0,
    )
    # peak は valid領域内の最大(=2.0)が採用される → abnormal False
    assert out["p_peak_epoch"] == 1200
    assert out["p_peak_abnormal"] is False
    assert out["p_direction"] == "Plateau"
