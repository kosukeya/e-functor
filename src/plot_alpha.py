# plot_alpha.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path("runs") / "alpha_log.csv"
df = pd.read_csv(log_path)

out_dir = Path("runs")
out_dir.mkdir(exist_ok=True, parents=True)

# --- α と ε ---
plt.figure(figsize=(8,4))
plt.plot(df["epoch"], df["alpha"], marker="o")
plt.ylim(-0.05, 1.05)
plt.title("alpha (gate) over epochs")
plt.xlabel("epoch")
plt.ylabel("alpha")
plt.tight_layout()
plt.savefig(out_dir / "alpha.png", dpi=150)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(df["epoch"], df["epsilon"], marker="o")
plt.axhline(0.0, linestyle="--")
plt.title("epsilon over epochs")
plt.xlabel("epoch")
plt.ylabel("epsilon")
plt.tight_layout()
plt.savefig(out_dir / "epsilon.png", dpi=150)
plt.close()

# --- dM vs dC ---
plt.figure(figsize=(8,4))
plt.plot(df["epoch"], df["dM"], marker="o", label="dM")
plt.plot(df["epoch"], df["dC"], marker="o", label="dC")
plt.title("dM and dC over epochs")
plt.xlabel("epoch")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "dM_dC.png", dpi=150)
plt.close()

# --- dM 内訳 ---
plt.figure(figsize=(8,4))
plt.plot(df["epoch"], df["d_cf"], marker="o", label="d_cf")
plt.plot(df["epoch"], df["d_mono"], marker="o", label="d_mono")
plt.plot(df["epoch"], df["d_att"], marker="o", label="d_att")
plt.plot(df["epoch"], df["d_self"], marker="o", label="d_self")
plt.title("dM components over epochs")
plt.xlabel("epoch")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "dM_components.png", dpi=150)
plt.close()

print("saved:", (out_dir / "alpha.png"), (out_dir / "epsilon.png"),
      (out_dir / "dM_dC.png"), (out_dir / "dM_components.png"))
