# lr_by_epoch.py
from pathlib import Path
import pandas as pd

class LrByEpochSchedule:
    """
    lr_mult_by_epoch.csv を読み、epoch→lr倍率を返す。
    CSV形式:
      epoch,cluster_label,is_event,lr_mult
    """
    def __init__(self, csv_path: str, default_mult: float = 1.0):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)
        if "epoch" not in df.columns or "lr_mult" not in df.columns:
            raise ValueError(f"bad lr_mult_by_epoch.csv: {self.csv_path}")
        self.map = {int(e): float(m) for e, m in zip(df["epoch"], df["lr_mult"])}
        self.default_mult = float(default_mult)

    def lr_mult(self, epoch: int) -> float:
        return float(self.map.get(int(epoch), self.default_mult))