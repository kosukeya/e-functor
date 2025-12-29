# island_min_stats.py
# Usage:
#   python island_min_stats.py --run_dir runs/test1 --epochs 200 400 600 1400 1600
# or
#   python island_min_stats.py --island_dir runs/test1/islands

import argparse
from pathlib import Path
import math

import torch
import pandas as pd


EPS = 1e-8

def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.view(-1).float()
    b = b.view(-1).float()
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std(unbiased=False) * b0.std(unbiased=False) + eps)
    return float(((a0 * b0).mean() / denom).item())

def attn_entropy(p: torch.Tensor, eps: float = 1e-8) -> float:
    """
    p: (..., K) attention prob-like. if not normalized, we normalize on last dim.
    returns mean entropy over leading dims.
    """
    p = p.float()
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    h = -(p * (p + eps).log()).sum(dim=-1)
    return float(h.mean().item())

def load_pt(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu")

def compute_stats(payload: dict) -> dict:
    # --- basics ---
    epoch = int(payload.get("epoch", -1))
    alpha_used = float(payload.get("alpha_used", float("nan")))

    env = payload["env"]               # (n,3) : plant,sun,water stacked in your code
    y = payload["y_true"].view(-1)     # (n,)
    y_stat = payload["y_stat"].view(-1)
    y_sem  = payload["y_sem"].view(-1)

    # y_mix may exist; if not, compute
    if "y_mix" in payload:
        y_mix = payload["y_mix"].view(-1)
    else:
        y_mix = (1.0 - alpha_used) * y_stat + alpha_used * y_sem

    # --- mix/stat/sem summary ---
    def summarize(name, t):
        return {
            f"{name}_mean": float(t.mean().item()),
            f"{name}_std":  float(t.std(unbiased=False).item()),
            f"corr_{name}_y": pearson_corr(t, y),
        }

    out = {
        "epoch": epoch,
        "alpha_used": alpha_used,
        **summarize("stat", y_stat),
        **summarize("sem",  y_sem),
        **summarize("mix",  y_mix),
    }

    # --- counterfactual deltas (mix) ---
    # delta = y_mix - y_cf
    if "y_sun0_mix" in payload:
        d_sun = (y_mix - payload["y_sun0_mix"].view(-1))
        out.update({
            "delta_sun0_mean": float(d_sun.mean().item()),
            "delta_sun0_abs_mean": float(d_sun.abs().mean().item()),
        })
    else:
        out.update({"delta_sun0_mean": float("nan"), "delta_sun0_abs_mean": float("nan")})

    if "y_plant0_mix" in payload:
        d_plt = (y_mix - payload["y_plant0_mix"].view(-1))
        out.update({
            "delta_plant0_mean": float(d_plt.mean().item()),
            "delta_plant0_abs_mean": float(d_plt.abs().mean().item()),
        })
    else:
        out.update({"delta_plant0_mean": float("nan"), "delta_plant0_abs_mean": float("nan")})

    # --- attention: growth row mean vector + entropy ---
    # your payload has attn: (n,7) already (growth row, head-avg)
    attn = payload.get("attn", payload.get("attn_growth_row"))
    if attn is not None:
        attn = attn.float()
        attn_mean = attn.mean(dim=0)  # (7,)
        out["attn_entropy"] = attn_entropy(attn)
        # store as compact string (or you can split columns)
        out["attn_mean_vec"] = "[" + ", ".join(f"{x:.4f}" for x in attn_mean.tolist()) + "]"
        out["attn_self_mass_proxy"] = float(attn_mean[6].item())  # growth->growth weight (rough proxy)
    else:
        out["attn_entropy"] = float("nan")
        out["attn_mean_vec"] = ""
        out["attn_self_mass_proxy"] = float("nan")

    # --- I–env correlation ---
    # payload["I"] is (n,3,D) in your code.
    I = payload.get("I", None)
    if I is not None:
        # ensure (n,3,D)
        if I.dim() == 2:
            # unexpected; can't do factor-wise
            out["corr_Ienv_plant"] = float("nan")
            out["corr_Ienv_sun"]   = float("nan")
            out["corr_Ienv_water"] = float("nan")
        else:
            # factor-wise norm: norm(I[:,i,:]) vs env[:,i]
            env_f = env.float()
            I_f = I.float()
            norms = I_f.norm(dim=-1)  # (n,3)

            out["corr_Ienv_plant"] = pearson_corr(norms[:, 0], env_f[:, 0])
            out["corr_Ienv_sun"]   = pearson_corr(norms[:, 1], env_f[:, 1])
            out["corr_Ienv_water"] = pearson_corr(norms[:, 2], env_f[:, 2])

            # optional: "disentanglement-ish" quick check
            # corr of plant env vs sun/water norms etc (off-diagonal)
            out["corr_plant_vs_sunI"]   = pearson_corr(env_f[:, 0], norms[:, 1])
            out["corr_plant_vs_waterI"] = pearson_corr(env_f[:, 0], norms[:, 2])
            out["corr_sun_vs_plantI"]   = pearson_corr(env_f[:, 1], norms[:, 0])
            out["corr_water_vs_plantI"] = pearson_corr(env_f[:, 2], norms[:, 0])
    else:
        out["corr_Ienv_plant"] = float("nan")
        out["corr_Ienv_sun"]   = float("nan")
        out["corr_Ienv_water"] = float("nan")
        out["corr_plant_vs_sunI"]   = float("nan")
        out["corr_plant_vs_waterI"] = float("nan")
        out["corr_sun_vs_plantI"]   = float("nan")
        out["corr_water_vs_plantI"] = float("nan")

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="", help="e.g., runs/test1")
    ap.add_argument("--island_dir", type=str, default="", help="e.g., runs/test1/islands")
    ap.add_argument("--epochs", type=int, nargs="*", default=[200, 400, 600, 1400, 1600])
    ap.add_argument("--save_csv", type=str, default="", help="optional output csv path")
    args = ap.parse_args()

    if args.island_dir:
        island_dir = Path(args.island_dir)
    elif args.run_dir:
        island_dir = Path(args.run_dir) / "islands"
    else:
        raise SystemExit("Provide --run_dir or --island_dir")

    rows = []
    missing = []
    for e in args.epochs:
        pt = island_dir / f"island_epoch{e:05d}.pt"
        if not pt.exists():
            missing.append(str(pt))
            continue
        payload = load_pt(pt)
        rows.append(compute_stats(payload))

    if missing:
        print("[WARN] missing files:")
        for m in missing:
            print("  ", m)

    df = pd.DataFrame(rows).sort_values("epoch")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)

    print("\n=== island min stats (selected epochs) ===")
    print(df.to_string(index=False))

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
