# eval_viz.py
import torch
import matplotlib.pyplot as plt

import config as C
from data import generate_world_data, make_split, pack_data
from model import MultiIWorldModel

def main():
    plant, sun, water, growth = generate_world_data(C.N)
    train_idx, _ = make_split(C.N, C.TRAIN_RATIO)
    train_data = pack_data(plant, sun, water, growth, train_idx, C.device)

    model = MultiIWorldModel(d_model=C.D_MODEL, n_heads=C.N_HEADS).to(C.device)
    model.load_state_dict(torch.load("model_last.pt", map_location=C.device))
    model.eval()

    p_tr, s_tr, w_tr, _ = train_data

    with torch.no_grad():
        _, y_sem, _, _ = model(p_tr, s_tr, w_tr, return_both=True)
        y_real = y_sem
        y_s0 = model(p_tr, torch.zeros_like(s_tr), w_tr)
        y_p0 = model(torch.zeros_like(p_tr), s_tr, w_tr)

    print("\n=== 反事実チェック (Multi-I + F2 + self-reg) ===")
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

    plt.figure(figsize=(5,4))
    plt.bar(labels, growth_row.detach().cpu())
    plt.ylim(0, 1.0)
    plt.title("Avg attention from [growth] query")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
