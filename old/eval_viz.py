# eval_viz.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
import config as C
from data import generate_world_data, make_split, pack_data
from model import MultiIWorldModel
from run_utils import resolve_run_dir, build_run_paths

_EPS = 1e-8


@torch.no_grad()
def _pearson_corr(a, b, eps=1e-8):
    a = a.view(-1)
    b = b.view(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std(unbiased=False) * b0.std(unbiased=False) + eps)
    return float(((a0 * b0).mean() / denom).item())


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    out_dir = Path(args.out_dir) if args.out_dir else (paths.figures_dir / "viz_basic")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path) if args.model_path else paths.model_last

    plant, sun, water, growth = generate_world_data(C.N)
    train_idx, _ = make_split(C.N, C.TRAIN_RATIO)
    train_data = pack_data(plant, sun, water, growth, train_idx, C.device)

    model = MultiIWorldModel(d_model=C.D_MODEL, n_heads=C.N_HEADS).to(C.device)
    model.load_state_dict(torch.load(model_path, map_location=C.device))
    model.eval()

    p_tr, s_tr, w_tr, y_tr = train_data

    alpha = getattr(C, "ALPHA_EVAL", 0.5)
    with torch.no_grad():
        y_stat, y_sem, _, _ = model(p_tr, s_tr, w_tr, return_both=True)
        y_mix = (1.0 - alpha) * y_stat + alpha * y_sem

        s0 = torch.zeros_like(s_tr)
        p0 = torch.zeros_like(p_tr)
        y_stat_s0, y_sem_s0, _, _ = model(p_tr, s0, w_tr, return_both=True)
        y_stat_p0, y_sem_p0, _, _ = model(p0, s_tr, w_tr, return_both=True)
        y_mix_s0 = (1.0 - alpha) * y_stat_s0 + alpha * y_sem_s0
        y_mix_p0 = (1.0 - alpha) * y_stat_p0 + alpha * y_sem_p0

        corr_stat = _pearson_corr(y_stat, y_tr)
        corr_sem = _pearson_corr(y_sem, y_tr)
        corr_mix = _pearson_corr(y_mix, y_tr)

        c_stat = ((1.0 - alpha) * y_stat).abs().mean()
        c_sem = (alpha * y_sem).abs().mean()
        contrib_sem = float((c_sem / (c_stat + c_sem + _EPS)).item())
        contrib_stat = 1.0 - contrib_sem

    print("\n=== Branch summary (y_stat / y_sem / y_mix) ===")
    print(f"alpha(eval)={alpha:.3f}")
    print(f"mean stat={y_stat.mean().item():.4f}  sem={y_sem.mean().item():.4f}  mix={y_mix.mean().item():.4f}")
    print(f"corr stat={corr_stat:.4f}  sem={corr_sem:.4f}  mix={corr_mix:.4f}")
    print(f"CF sun0  stat={y_stat_s0.mean().item():.4f} sem={y_sem_s0.mean().item():.4f} mix={y_mix_s0.mean().item():.4f}")
    print(f"CF plant0 stat={y_stat_p0.mean().item():.4f} sem={y_sem_p0.mean().item():.4f} mix={y_mix_p0.mean().item():.4f}")
    print(f"contrib stat={contrib_stat:.3f} sem={contrib_sem:.3f}")

    plt.figure(figsize=(4, 3))
    plt.bar(["stat", "sem"], [contrib_stat, contrib_sem])
    plt.ylim(0, 1.0)
    plt.title("Contribution ratio (mean abs)")
    plt.tight_layout()
    out_path2 = out_dir / "branch_contrib.png"
    plt.savefig(out_path2, dpi=150)
    print(f"Saved figure to: {out_path2}")

    with torch.no_grad():
        _, y_sem2, _, _ = model(p_tr, s_tr, w_tr, return_both=True)
        y_real = y_sem2
        y_s0 = model(p_tr, torch.zeros_like(s_tr), w_tr)
        y_p0 = model(torch.zeros_like(p_tr), s_tr, w_tr)

    print("\n=== Counterfactual check (Multi-I + F2 + self-reg) ===")
    print("mean growth (real data) : ", y_real.mean().item())
    print("mean growth (sun=0)     : ", y_s0.mean().item())
    print("mean growth (plant=0)   : ", y_p0.mean().item())

    with torch.no_grad():
        _, attn, _ = model(p_tr, s_tr, w_tr, return_attn=True)

    attn_mean = attn.mean(dim=1).mean(dim=0)
    growth_row = attn_mean[6]
    labels = ["plant", "sun", "water", "I1", "I2", "I3", "growth_self"]

    print("\nAvg attention from [growth] query:")
    for name, val in zip(labels, growth_row.tolist()):
        print(f"{name:>12}: {val:.4f}")

    plt.figure(figsize=(5, 4))
    plt.bar(labels, growth_row.detach().cpu())
    plt.ylim(0, 1.0)
    plt.title("Avg attention from [growth] query")
    plt.tight_layout()

    out_path = out_dir / "attn_bar.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
