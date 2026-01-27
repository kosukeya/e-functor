# label_lr_schedule.py
import pandas as pd

class LabelConditionedLR:
    """
    window: [epoch_start, epoch_end] ごとに lr_mult を持つ。
    epoch がどのwindowに入るかで lr_mult を返す。
    """
    def __init__(self, schedule_csv_path: str, default_mult: float = 1.0):
        df = pd.read_csv(schedule_csv_path)
        need = {"epoch_start","epoch_end","lr_mult"}
        if not need.issubset(set(df.columns)):
            raise ValueError(f"{schedule_csv_path} must contain {need}")
        self.df = df.sort_values(["epoch_start","epoch_end"]).reset_index(drop=True)
        self.default_mult = float(default_mult)

    def lr_mult(self, epoch: int) -> float:
        # 単純線形検索（epoch数が小さい想定）。必要なら二分探索に置換可。
        for _, r in self.df.iterrows():
            s, e = int(r["epoch_start"]), int(r["epoch_end"])
            if s <= epoch <= e:
                return float(r["lr_mult"])
        return self.default_mult