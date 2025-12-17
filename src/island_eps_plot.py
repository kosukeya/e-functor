# island_eps_plot.py
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
    out_png="island_eps_timeseries.png",
    islands=(0, 1),
):
    runs_dir = Path(runs_dir)
    in_path = runs_dir / in_csv
    if not in_path.exists():
        raise FileNotFoundError(f"not found: {in_path}")

    df = pd.read_csv(in_path)

    # basic cleanup
    df = df.sort_values("epoch").reset_index(drop=True)
    x = _col(df, "epoch")

    # plot
    plt.figure(figsize=(9, 4))
    plt.plot(x, _col(df, "epsilon_all"), marker="o", label="epsilon_all")

    for k in islands:
        col = f"epsilon_{k}"
        if col in df.columns:
            plt.plot(x, df[col], marker="o", label=col)
        else:
            print(f"[warn] missing {col}; skipped")

    plt.axhline(0.0, linestyle="--")
    plt.title("Island epsilon time series")
    plt.xlabel("epoch")
    plt.ylabel("epsilon")
    plt.legend()
    plt.tight_layout()

    out_path = runs_dir / out_png
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("saved:", out_path)

if __name__ == "__main__":
    main()
