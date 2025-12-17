# island_dm_plot.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def _col(df, name):
    if name not in df.columns:
        raise KeyError(f"missing column: {name} (available={list(df.columns)})")
    return df[name]

def main(
    runs_dir="runs",
    in_csv="island_eps_summary.csv",
    out_png="island_dM_components_timeseries.png",
    islands=(0, 1),
    comps=("d_cf", "d_att", "d_self"),
):
    runs_dir = Path(runs_dir)
    in_path = runs_dir / in_csv
    if not in_path.exists():
        raise FileNotFoundError(f"not found: {in_path}")

    df = pd.read_csv(in_path).sort_values("epoch").reset_index(drop=True)
    x = _col(df, "epoch")

    # 3つ並べて見たいのでサブプロット（縦）にする
    fig, axes = plt.subplots(len(comps), 1, figsize=(9, 3.2 * len(comps)), sharex=True)
    if len(comps) == 1:
        axes = [axes]

    for ax, comp in zip(axes, comps):
        # all
        ax.plot(x, _col(df, f"{comp}_all"), marker="o", label=f"{comp}_all")

        # islands
        for k in islands:
            col = f"{comp}_{k}"
            if col in df.columns:
                ax.plot(x, df[col], marker="o", label=col)
            else:
                print(f"[warn] missing {col}; skipped")

        ax.axhline(0.0, linestyle="--")
        ax.set_title(f"{comp} time series")
        ax.set_ylabel(comp)
        ax.legend()

    axes[-1].set_xlabel("epoch")
    fig.suptitle("Island dM component time series", y=1.02)
    fig.tight_layout()

    out_path = runs_dir / out_png
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved:", out_path)

if __name__ == "__main__":
    main()
