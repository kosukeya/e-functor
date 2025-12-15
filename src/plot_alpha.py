# plot_alpha.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path("runs") / "alpha_log.csv"

# --- robust read (schema changes / broken lines) ---
try:
    # pandas>=1.3: bad lines skip
    df = pd.read_csv(log_path, engine="python", on_bad_lines="skip")
except TypeError:
    # older pandas fallback
    df = pd.read_csv(log_path, engine="python", error_bad_lines=False, warn_bad_lines=True)

# column cleanup
df.columns = [c.strip() for c in df.columns]
# to numeric where possible
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="ignore")

out_dir = Path("runs")
out_dir.mkdir(exist_ok=True, parents=True)

def pick_col(*cands):
    """Return first existing column name among candidates, else None."""
    for c in cands:
        if c in df.columns:
            return c
    return None

def plot_if_exists(ycol, fname, title, ylabel, extra=None, ylim=None):
    """Plot df['epoch'] vs df[ycol] if both columns exist and have some numeric data."""
    if "epoch" not in df.columns or ycol is None or ycol not in df.columns:
        print(f"[skip] missing columns: epoch or {ycol}")
        return
    x = pd.to_numeric(df["epoch"], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() == 0:
        print(f"[skip] no numeric data for: {ycol}")
        return

    plt.figure(figsize=(8,4))
    plt.plot(x[m], y[m], marker="o")
    if extra:
        extra()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()
    print("saved:", out_dir / fname)

# --- α と ε ---
alpha_col = pick_col("alpha_used", "alpha")   # forwardで使ったαを優先
plot_if_exists(alpha_col, "alpha.png", "alpha (gate) over epochs", "alpha", ylim=(-0.05, 1.05))

plot_if_exists(
    "epsilon", "epsilon.png", "epsilon over epochs", "epsilon",
    extra=lambda: plt.axhline(0.0, linestyle="--")
)

# --- dM vs dC ---
if "epoch" in df.columns:
    x = pd.to_numeric(df["epoch"], errors="coerce")
else:
    x = None

def plot_two_if_exists(y1, y2, fname, title):
    if x is None or y1 not in df.columns or y2 not in df.columns:
        print(f"[skip] missing columns for {fname}: epoch/{y1}/{y2}")
        return
    a = pd.to_numeric(df[y1], errors="coerce")
    b = pd.to_numeric(df[y2], errors="coerce")
    m1 = x.notna() & a.notna()
    m2 = x.notna() & b.notna()
    if m1.sum() == 0 and m2.sum() == 0:
        print(f"[skip] no numeric data for {fname}")
        return

    plt.figure(figsize=(8,4))
    if m1.sum() > 0: plt.plot(x[m1], a[m1], marker="o", label=y1)
    if m2.sum() > 0: plt.plot(x[m2], b[m2], marker="o", label=y2)
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()
    print("saved:", out_dir / fname)

plot_two_if_exists("dM", "dC", "dM_dC.png", "dM and dC over epochs")

# --- dM 内訳 ---
# あるものだけ描く（欠けててもOK）
components = [c for c in ["d_cf", "d_mono", "d_att", "d_self"] if c in df.columns]
if x is None or len(components) == 0:
    print("[skip] dM_components.png: missing epoch or all component columns")
else:
    plt.figure(figsize=(8,4))
    any_plotted = False
    for c in components:
        y = pd.to_numeric(df[c], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() > 0:
            plt.plot(x[m], y[m], marker="o", label=c)
            any_plotted = True
    if any_plotted:
        plt.title("dM components over epochs")
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "dM_components.png", dpi=150)
        print("saved:", out_dir / "dM_components.png")
    else:
        print("[skip] dM_components.png: no numeric data")
    plt.close()
